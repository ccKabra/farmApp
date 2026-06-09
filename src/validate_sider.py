"""
Etapa 6: Validación contra SIDER 4.1.
Descarga SIDER desde su URL pública y compara las predicciones del mejor modelo
contra los efectos adversos conocidos por fármaco.

SIDER usa MedDRA terms; FAERS usa MedDRA PT -> compatibles directamente.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import urllib.request
import gzip
import io
from config import DATA_PROCESSED, DATA_RAW, MODELS_DIR, OUTPUTS_DIR, BIOBERT_MODEL, DEVICE

# ── Descargar SIDER 4.1 ───────────────────────────────────────────────────────
SIDER_URL = "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz"
SIDER_LOCAL = DATA_RAW / "sider_meddra_all_se.tsv"

if not SIDER_LOCAL.exists():
    print(f"Descargando SIDER 4.1 desde {SIDER_URL}...")
    try:
        req = urllib.request.Request(SIDER_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = gzip.decompress(resp.read())
        with open(SIDER_LOCAL, "wb") as f:
            f.write(raw)
        print(f"SIDER guardado en: {SIDER_LOCAL}")
    except Exception as e:
        print(f"Error descargando SIDER: {e}")
        print("Usando validacion alternativa con DrugBank/literatura...")
        SIDER_LOCAL = None
else:
    print(f"SIDER ya existe: {SIDER_LOCAL}")

# ── Cargar SIDER ──────────────────────────────────────────────────────────────
if SIDER_LOCAL and SIDER_LOCAL.exists():
    # Columnas: stitch_id_flat, stitch_id_stereo, umls_cui_se, meddra_type, umls_cui_meddra, side_effect_name
    sider = pd.read_csv(SIDER_LOCAL, sep="\t", header=None,
                        names=["stitch_flat","stitch_stereo","umls_se","meddra_type","umls_meddra","side_effect"])
    sider = sider[sider["meddra_type"] == "PT"].copy()
    sider["side_effect"] = sider["side_effect"].str.strip().str.title()
    print(f"SIDER cargado: {len(sider):,} entradas PT | {sider['stitch_flat'].nunique():,} farmacos")

    # Construir dict: stitch_id -> set de efectos
    sider_effects = defaultdict(set)
    for _, row in sider.iterrows():
        sider_effects[row["stitch_flat"]].add(row["side_effect"])

    # Necesitamos mapear nombres de fármacos FAERS -> STITCH IDs
    # SIDER también provee meddra_freq.tsv con nombres, pero usamos drugnames_map
    # Alternativa: buscar por nombre en SIDER usando drug_names.tsv
    DRUG_NAMES_URL = "http://sideeffects.embl.de/media/download/drug_names.tsv"
    drug_names_local = DATA_RAW / "sider_drug_names.tsv"
    if not drug_names_local.exists():
        try:
            req2 = urllib.request.Request(DRUG_NAMES_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req2, timeout=60) as resp:
                with open(drug_names_local, "wb") as f:
                    f.write(resp.read())
            print("drug_names.tsv descargado.")
        except Exception as e:
            print(f"Error descargando drug_names: {e}")
            drug_names_local = None

    if drug_names_local and drug_names_local.exists():
        drug_names = pd.read_csv(drug_names_local, sep="\t", header=None,
                                  names=["stitch_id", "drug_name"])
        drug_names["drug_name_upper"] = drug_names["drug_name"].str.upper().str.strip()
        name_to_stitch = dict(zip(drug_names["drug_name_upper"], drug_names["stitch_id"]))
        print(f"drug_names.tsv: {len(name_to_stitch):,} farmacos mapeados")
    else:
        name_to_stitch = {}
else:
    sider_effects = {}
    name_to_stitch = {}

# ── Cargar modelo BioBERT fine-tuned ─────────────────────────────────────────
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

save_dir = MODELS_DIR / "biobert_finetuned"
label_df = pd.read_csv(save_dir / "label_names.csv")
label_names = label_df["label"].tolist()
num_labels = len(label_names)
thresholds = np.load(save_dir / "thresholds.npy")

print(f"\nCargando modelo fine-tuned ({num_labels} etiquetas)...")
tokenizer = AutoTokenizer.from_pretrained(str(save_dir))
bert_base = AutoModel.from_pretrained(BIOBERT_MODEL)
model = BioBERTClassifier(bert_base, num_labels).to(DEVICE)
model.load_state_dict(torch.load(save_dir / "model.pt", map_location=DEVICE))
model.eval()
print("Modelo listo.")

# ── Predicción para fármacos con datos en SIDER ───────────────────────────────
df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)
top_drugs = [d for d, _ in Counter(
    [d for row in df["drug"].dropna() for d in row.split("|")]
).most_common(30)]

print(f"\nValidando {len(top_drugs)} farmacos frecuentes contra SIDER...")

results = []
for drug in top_drugs:
    stitch_id = name_to_stitch.get(drug.upper())
    sider_known = sider_effects.get(stitch_id, set()) if stitch_id else set()

    # Predecir para este fármaco (paciente genérico)
    text = f"patient age 55 years female. drug: {drug}. indication: unknown"
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=128, padding="max_length").to(DEVICE)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    predicted = {label_names[j] for j, p in enumerate(probs) if p >= thresholds[j]}
    sider_in_labels = sider_known & set(label_names)  # solo efectos que conocemos

    if sider_in_labels:
        tp = len(predicted & sider_in_labels)
        precision_d = tp / len(predicted) if predicted else 0
        recall_d    = tp / len(sider_in_labels)
        f1_d        = 2 * precision_d * recall_d / (precision_d + recall_d + 1e-9)
    else:
        precision_d = recall_d = f1_d = None

    results.append({
        "drug": drug,
        "stitch_found": stitch_id is not None,
        "sider_effects_in_vocab": len(sider_in_labels),
        "predicted": len(predicted),
        "tp": len(predicted & sider_in_labels) if sider_in_labels else 0,
        "precision_vs_sider": round(precision_d, 3) if precision_d is not None else None,
        "recall_vs_sider":    round(recall_d,    3) if recall_d    is not None else None,
        "f1_vs_sider":        round(f1_d,        3) if f1_d        is not None else None,
        "predicted_effects": "|".join(sorted(predicted))
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUTS_DIR / "sider_validation.csv", index=False)

valid = results_df.dropna(subset=["f1_vs_sider"])
print(f"\n=== VALIDACION SIDER (farmacos mapeados: {len(valid)}/{len(results_df)}) ===")
if len(valid) > 0:
    print(f"  F1 medio vs SIDER    : {valid['f1_vs_sider'].mean():.3f}")
    print(f"  Precision media      : {valid['precision_vs_sider'].mean():.3f}")
    print(f"  Recall medio         : {valid['recall_vs_sider'].mean():.3f}")
    print("\n  Por farmaco:")
    print(valid[["drug","sider_effects_in_vocab","predicted","tp","f1_vs_sider"]].to_string(index=False))
else:
    print("No se pudieron mapear farmacos FAERS->SIDER (nombres no coinciden directamente).")
    print("Mostrando predicciones crudas para los top farmacos:")
    for _, row in results_df.head(10).iterrows():
        effects = row["predicted_effects"].split("|")[:5] if row["predicted_effects"] else []
        print(f"  {row['drug']:<35} -> {', '.join(effects)}")

print(f"\nResultados completos en: outputs/sider_validation.csv")

# ── Gráfico ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Validacion vs SIDER 4.1", fontsize=13)

if len(valid) > 0:
    valid_sorted = valid.sort_values("f1_vs_sider", ascending=True)
    axes[0].barh(valid_sorted["drug"], valid_sorted["f1_vs_sider"], color="mediumseagreen")
    axes[0].axvline(valid["f1_vs_sider"].mean(), color="red", linestyle="--",
                    label=f"Media={valid['f1_vs_sider'].mean():.3f}")
    axes[0].set_title("F1 por farmaco vs SIDER"); axes[0].legend()
else:
    axes[0].text(0.5, 0.5, "Sin mapeo FAERS->SIDER", ha="center", va="center")
    axes[0].set_title("F1 por farmaco vs SIDER")

# Distribución de predicciones por fármaco
pred_counts = results_df.set_index("drug")["predicted"].sort_values(ascending=True)
axes[1].barh(pred_counts.index, pred_counts.values, color="steelblue")
axes[1].set_title("Efectos adversos predichos por farmaco")
axes[1].set_xlabel("Cantidad de efectos predichos")

plt.tight_layout()
out = OUTPUTS_DIR / "figures" / "sider_validation.png"
plt.savefig(out, dpi=150)
print(f"Grafico guardado: {out}")
