"""
Etapa 5b: Optimizar umbral de decisión por etiqueta y re-entrenar con más épocas.
El fine-tuning base tiene recall alto pero precision baja -> umbral 0.5 no es óptimo.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter
import matplotlib.pyplot as plt
from config import DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR, BIOBERT_MODEL, DEVICE, RANDOM_SEED, TEST_SIZE

MAX_LEN     = 128
BATCH_TRAIN = 32
BATCH_EVAL  = 64
EPOCHS      = 10
LR          = 2e-5

class FAERSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], padding="max_length", truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": self.labels[idx]}

class BioBERTClassifier(nn.Module):
    def __init__(self, bert_model, num_labels, dropout=0.3):
        super().__init__()
        self.bert = bert_model
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_labels)
        )
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

# ── Datos ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)
df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
MIN_REACTION_FREQ = 50
all_reac = [r for row in df["reactions"].dropna() for r in row.split("|")]
freq_reac = {r for r, n in Counter(all_reac).items() if n >= MIN_REACTION_FREQ}
df["reaction_list"] = df["reactions"].apply(
    lambda s: [r for r in s.split("|") if r in freq_reac] if pd.notna(s) else [])
mask = df["reaction_list"].apply(len) > 0
df = df[mask].reset_index(drop=True)

label_names = sorted(freq_reac)
label2idx = {l: i for i, l in enumerate(label_names)}
num_labels = len(label_names)

# Texto más rico: incluir sexo y edad
def build_text(row):
    age = f"age {int(row['age_years'])} years" if pd.notna(row.get("age_years")) else "age unknown"
    sex = {"M": "male", "F": "female"}.get(str(row.get("sex", "U")).strip(), "unknown sex")
    drug = str(row.get("drug", "unknown"))[:120]
    indi = str(row.get("indications", "unknown"))[:250]
    return f"patient {age} {sex}. drug: {drug}. indication: {indi}"

texts = df.apply(build_text, axis=1).tolist()
Y = np.zeros((len(df), num_labels), dtype=np.float32)
for i, reactions in enumerate(df["reaction_list"]):
    for r in reactions:
        if r in label2idx:
            Y[i, label2idx[r]] = 1.0

idx = list(range(len(texts)))
train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_SEED)
# Usar 10% del train como val para tunear umbral
val_size = int(len(train_idx) * 0.1)
val_idx, train_idx = train_idx[:val_size], train_idx[val_size:]

texts_train = [texts[i] for i in train_idx]
texts_val   = [texts[i] for i in val_idx]
texts_test  = [texts[i] for i in test_idx]
Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]
print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
train_ds = FAERSDataset(texts_train, Y_train, tokenizer, MAX_LEN)
val_ds   = FAERSDataset(texts_val,   Y_val,   tokenizer, MAX_LEN)
test_ds  = FAERSDataset(texts_test,  Y_test,  tokenizer, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_EVAL,  shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_EVAL,  shuffle=False, num_workers=0)

# ── Modelo ────────────────────────────────────────────────────────────────────
bert_base = AutoModel.from_pretrained(BIOBERT_MODEL)
model = BioBERTClassifier(bert_base, num_labels).to(DEVICE)

# Cargar pesos previos si existen (continuar entrenamiento)
prev_weights = MODELS_DIR / "biobert_finetuned" / "model.pt"
if prev_weights.exists():
    model.load_state_dict(torch.load(prev_weights, map_location=DEVICE))
    print("Pesos previos cargados, continuando entrenamiento...")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

pos_counts = Y_train.sum(axis=0)
neg_counts = len(Y_train) - pos_counts
pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def get_logits(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labs = batch["labels"]
            logits = model(ids, mask)
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labs.numpy())
    return np.vstack(all_logits), np.vstack(all_labels)

def find_best_thresholds(probs, labels, thresholds=np.arange(0.1, 0.9, 0.05)):
    best_t = np.full(probs.shape[1], 0.5)
    for j in range(probs.shape[1]):
        best_f1, best = 0, 0.5
        for t in thresholds:
            pred = (probs[:, j] >= t).astype(int)
            f1 = f1_score(labels[:, j], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best = f1, t
        best_t[j] = best
    return best_t

# ── Entrenamiento extendido ───────────────────────────────────────────────────
print(f"\nFine-tuning extendido: {EPOCHS} epocas adicionales")
history = {"train_loss": [], "val_f1_macro": []}
best_val_f1 = 0
best_state = None

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

    avg_loss = total_loss / len(train_loader)

    # Validacion con umbral optimizado
    val_probs, val_labels = get_logits(model, val_loader)
    thresholds = find_best_thresholds(val_probs, val_labels)
    val_pred = (val_probs >= thresholds).astype(int)
    val_f1 = f1_score(val_labels, val_pred, average="macro", zero_division=0)

    history["train_loss"].append(avg_loss)
    history["val_f1_macro"].append(val_f1)
    print(f"[Epoca {epoch:2d}] loss={avg_loss:.4f} | val_F1_macro(opt_thresh)={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_thresholds = thresholds.copy()
        print(f"  ** Mejor modelo guardado (F1={best_val_f1:.4f}) **")

# ── Evaluacion final con mejor modelo ────────────────────────────────────────
print("\n=== EVALUACION FINAL (mejor modelo, umbral optimo) ===")
model.load_state_dict(best_state)
test_probs, Y_true = get_logits(model, test_loader)
Y_pred = (test_probs >= best_thresholds).astype(int)

f1_mac  = f1_score(Y_true, Y_pred, average="macro",   zero_division=0)
f1_mic  = f1_score(Y_true, Y_pred, average="micro",   zero_division=0)
f1_sam  = f1_score(Y_true, Y_pred, average="samples", zero_division=0)
prec    = precision_score(Y_true, Y_pred, average="macro", zero_division=0)
rec     = recall_score(Y_true, Y_pred, average="macro",    zero_division=0)

print(f"  F1 macro   : {f1_mac:.4f}   (RF+BioBERT: 0.1296)")
print(f"  F1 micro   : {f1_mic:.4f}   (RF+BioBERT: 0.1498)")
print(f"  F1 samples : {f1_sam:.4f}  (RF+BioBERT: 0.1267)")
print(f"  Precision  : {prec:.4f}  (RF+BioBERT: 0.0990)")
print(f"  Recall     : {rec:.4f}  (RF+BioBERT: 0.2096)")

# ── Guardar ───────────────────────────────────────────────────────────────────
save_dir = MODELS_DIR / "biobert_finetuned"
save_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), save_dir / "model.pt")
tokenizer.save_pretrained(save_dir)
np.save(save_dir / "thresholds.npy", best_thresholds)
pd.DataFrame({"label": label_names, "threshold": best_thresholds}).to_csv(
    save_dir / "label_thresholds.csv", index=False)
print(f"\nModelo y umbrales guardados en: {save_dir}")

# ── Grafico ───────────────────────────────────────────────────────────────────
f1_per_label = f1_score(Y_true, Y_pred, average=None, zero_division=0)
label_metrics = pd.DataFrame({
    "label": label_names, "f1": f1_per_label, "threshold": best_thresholds
}).sort_values("f1", ascending=False)
print("\n=== TOP 15 ETIQUETAS ===")
print(label_metrics.head(15).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("BioBERT Fine-tuning con Umbral Optimizado", fontsize=13)
axes[0].plot(history["train_loss"], marker="o", color="steelblue"); axes[0].set_title("Train Loss")
axes[1].plot(history["val_f1_macro"], marker="o", color="darkorange"); axes[1].set_title("Val F1 macro")
top30 = label_metrics.head(30)
axes[2].barh(top30["label"][::-1], top30["f1"][::-1], color="mediumseagreen")
axes[2].axvline(f1_mac, color="red", linestyle="--", label=f"F1 macro={f1_mac:.3f}")
axes[2].set_title("F1 por etiqueta (Top 30)"); axes[2].legend()
plt.tight_layout()
out = OUTPUTS_DIR / "figures" / "finetune_optimized.png"
plt.savefig(out, dpi=150)
print(f"Grafico guardado: {out}")
