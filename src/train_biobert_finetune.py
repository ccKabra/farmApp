"""
Etapa 5: Fine-tuning BioBERT como clasificador multi-label.
Entrada: texto "drug: X. indication: Y" -> 98 etiquetas binarias
GPU: RTX 4070 Ti SUPER (CUDA)

Output: models/biobert_finetuned/, outputs/figures/finetune_metrics.png
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
from config import DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR, BIOBERT_MODEL, DEVICE, RANDOM_SEED, TEST_SIZE

# ── Hiperparámetros ───────────────────────────────────────────────────────────
MAX_LEN   = 128
BATCH_TRAIN = 32
BATCH_EVAL  = 64
EPOCHS    = 5
LR        = 2e-5
THRESHOLD = 0.5   # umbral para convertir probabilidad a etiqueta binaria

# ── Dataset ───────────────────────────────────────────────────────────────────
class FAERSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         self.labels[idx]
        }

# ── Modelo ────────────────────────────────────────────────────────────────────
class BioBERTClassifier(nn.Module):
    def __init__(self, bert_model, num_labels, dropout=0.3):
        super().__init__()
        self.bert = bert_model
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # token [CLS]
        return self.classifier(cls)

# ── Cargar datos ──────────────────────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)
df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")

MIN_REACTION_FREQ = 50
all_reac = [r for row in df["reactions"].dropna() for r in row.split("|")]
freq_reac = {r for r, n in Counter(all_reac).items() if n >= MIN_REACTION_FREQ}

df["reaction_list"] = df["reactions"].apply(
    lambda s: [r for r in s.split("|") if r in freq_reac] if pd.notna(s) else []
)
mask = df["reaction_list"].apply(len) > 0
df = df[mask].reset_index(drop=True)

label_names = sorted(freq_reac)
label2idx = {l: i for i, l in enumerate(label_names)}
num_labels = len(label_names)
print(f"Casos: {len(df):,}  |  Etiquetas: {num_labels}")

# Texto de entrada
texts = (
    "drug: " + df["drug"].fillna("unknown").str[:100] +
    ". indication: " + df["indications"].fillna("unknown").str[:200]
).tolist()

# Matriz Y
Y = np.zeros((len(df), num_labels), dtype=np.float32)
for i, reactions in enumerate(df["reaction_list"]):
    for r in reactions:
        if r in label2idx:
            Y[i, label2idx[r]] = 1.0

# Split 70/30
idx = list(range(len(texts)))
train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_SEED)
texts_train = [texts[i] for i in train_idx]
texts_test  = [texts[i] for i in test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]
print(f"Train: {len(train_idx):,}  |  Test: {len(test_idx):,}")

# ── Tokenizer y DataLoaders ───────────────────────────────────────────────────
print(f"\nCargando tokenizer BioBERT...")
tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)

train_ds = FAERSDataset(texts_train, Y_train, tokenizer, MAX_LEN)
test_ds  = FAERSDataset(texts_test,  Y_test,  tokenizer, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_EVAL,  shuffle=False, num_workers=0)

# ── Modelo y optimizador ──────────────────────────────────────────────────────
print("Cargando BioBERT base...")
bert_base = AutoModel.from_pretrained(BIOBERT_MODEL)
model = BioBERTClassifier(bert_base, num_labels).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
)

# Pos weight para clases desbalanceadas
pos_counts = Y_train.sum(axis=0)
neg_counts = len(Y_train) - pos_counts
pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ── Entrenamiento ─────────────────────────────────────────────────────────────
def evaluate(model, loader, threshold=THRESHOLD):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labs = batch["labels"].to(DEVICE)
            logits = model(ids, mask)
            loss = criterion(logits, labs)
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= threshold).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labs.cpu().numpy())
    Y_pred = np.vstack(all_preds)
    Y_true = np.vstack(all_labels)
    return (
        total_loss / len(loader),
        f1_score(Y_true, Y_pred, average="macro",   zero_division=0),
        f1_score(Y_true, Y_pred, average="micro",   zero_division=0),
        f1_score(Y_true, Y_pred, average="samples", zero_division=0),
        Y_pred, Y_true
    )

history = {"train_loss": [], "val_f1_macro": [], "val_f1_micro": [], "val_f1_samples": []}

print(f"\nIniciando fine-tuning: {EPOCHS} epocas, LR={LR}, batch={BATCH_TRAIN}")
print(f"Device: {DEVICE.upper()} — {torch.cuda.get_device_name(0)}\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labs = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 20 == 0:
            print(f"  Epoca {epoch} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}")

    avg_train_loss = total_loss / len(train_loader)
    val_loss, f1_mac, f1_mic, f1_sam, _, _ = evaluate(model, test_loader)

    history["train_loss"].append(avg_train_loss)
    history["val_f1_macro"].append(f1_mac)
    history["val_f1_micro"].append(f1_mic)
    history["val_f1_samples"].append(f1_sam)

    print(f"\n[Epoca {epoch}] train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | "
          f"F1_macro={f1_mac:.4f} | F1_micro={f1_mic:.4f} | F1_samples={f1_sam:.4f}\n")

# ── Evaluación final ──────────────────────────────────────────────────────────
print("=== EVALUACION FINAL ===")
_, f1_mac, f1_mic, f1_sam, Y_pred_final, Y_true_final = evaluate(model, test_loader)

precision = precision_score(Y_true_final, Y_pred_final, average="macro", zero_division=0)
recall    = recall_score(Y_true_final, Y_pred_final, average="macro",    zero_division=0)
h_loss    = hamming_loss(Y_true_final, Y_pred_final)

print(f"  F1 macro   : {f1_mac:.4f}   (RF+BioBERT: 0.1296)")
print(f"  F1 micro   : {f1_mic:.4f}   (RF+BioBERT: 0.1498)")
print(f"  F1 samples : {f1_sam:.4f}  (RF+BioBERT: 0.1267)")
print(f"  Precision  : {precision:.4f}  (RF+BioBERT: 0.0990)")
print(f"  Recall     : {recall:.4f}  (RF+BioBERT: 0.2096)")
print(f"  Hamming loss: {h_loss:.4f}  (RF+BioBERT: 0.0586)")

f1_per_label = f1_score(Y_true_final, Y_pred_final, average=None, zero_division=0)
label_metrics = pd.DataFrame({
    "label": label_names, "f1": f1_per_label,
    "support": Y_true_final.sum(axis=0)
}).sort_values("f1", ascending=False)

print("\n=== TOP 15 ETIQUETAS ===")
print(label_metrics.head(15).to_string(index=False))

# ── Guardar modelo ────────────────────────────────────────────────────────────
save_dir = MODELS_DIR / "biobert_finetuned"
save_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), save_dir / "model.pt")
tokenizer.save_pretrained(save_dir)
pd.DataFrame({"label": label_names}).to_csv(save_dir / "label_names.csv", index=False)
print(f"\nModelo guardado en: {save_dir}")

# ── Gráficos ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fine-tuning BioBERT — Resultados", fontsize=13)

epochs_range = range(1, EPOCHS + 1)
axes[0].plot(epochs_range, history["train_loss"], marker="o", color="steelblue", label="Train Loss")
axes[0].set_title("Loss de entrenamiento"); axes[0].set_xlabel("Epoca"); axes[0].legend()

axes[1].plot(epochs_range, history["val_f1_macro"],   marker="o", label="F1 macro")
axes[1].plot(epochs_range, history["val_f1_micro"],   marker="s", label="F1 micro")
axes[1].plot(epochs_range, history["val_f1_samples"], marker="^", label="F1 samples")
axes[1].set_title("F1 en validacion"); axes[1].set_xlabel("Epoca"); axes[1].legend()

top30 = label_metrics.head(30)
axes[2].barh(top30["label"][::-1], top30["f1"][::-1], color="mediumseagreen")
axes[2].axvline(f1_mac, color="red", linestyle="--", label=f"F1 macro={f1_mac:.3f}")
axes[2].set_title("F1 por etiqueta (Top 30)"); axes[2].legend()

plt.tight_layout()
out = OUTPUTS_DIR / "figures" / "finetune_metrics.png"
plt.savefig(out, dpi=150)
print(f"Grafico guardado: {out}")
