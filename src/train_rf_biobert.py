"""
Etapa 4b: RF con embeddings BioBERT. Compara vs baseline TF-IDF.
Output: models/rf_biobert.pkl, outputs/figures/rf_biobert_metrics.png
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from config import DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED, TEST_SIZE

print("Cargando X_biobert e Y...")
X = pd.read_csv(DATA_PROCESSED / "X_biobert.csv")
Y = pd.read_csv(DATA_PROCESSED / "Y.csv")
label_names = Y.columns.tolist()
print(f"X: {X.shape}  |  Y: {Y.shape}")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

print("\nEntrenando RF + BioBERT embeddings...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced"
)
model = MultiOutputClassifier(rf, n_jobs=-1)
model.fit(X_train, Y_train)
print("Entrenamiento completo.")

Y_pred = model.predict(X_test)

f1_macro   = f1_score(Y_test, Y_pred, average="macro",   zero_division=0)
f1_micro   = f1_score(Y_test, Y_pred, average="micro",   zero_division=0)
f1_samples = f1_score(Y_test, Y_pred, average="samples", zero_division=0)
precision  = precision_score(Y_test, Y_pred, average="macro", zero_division=0)
recall     = recall_score(Y_test, Y_pred, average="macro",    zero_division=0)
h_loss     = hamming_loss(Y_test, Y_pred)

print("\n=== RF + BioBERT EMBEDDINGS ===")
print(f"  F1 macro   : {f1_macro:.4f}   (baseline: 0.1070)")
print(f"  F1 micro   : {f1_micro:.4f}   (baseline: 0.1039)")
print(f"  F1 samples : {f1_samples:.4f}  (baseline: 0.1052)")
print(f"  Precision  : {precision:.4f}  (baseline: 0.0636)")
print(f"  Recall     : {recall:.4f}  (baseline: 0.4669)")
print(f"  Hamming loss: {h_loss:.4f}  (baseline: 0.1803)")

f1_per_label = f1_score(Y_test, Y_pred, average=None, zero_division=0)
label_metrics = pd.DataFrame({
    "label": label_names, "f1": f1_per_label,
    "support": np.array(Y_test).sum(axis=0)
}).sort_values("f1", ascending=False)

print("\n=== TOP 15 ETIQUETAS ===")
print(label_metrics.head(15).to_string(index=False))

MODELS_DIR.mkdir(parents=True, exist_ok=True)
with open(MODELS_DIR / "rf_biobert.pkl", "wb") as f:
    pickle.dump({"model": model, "label_names": label_names, "feature_names": X.columns.tolist()}, f)
print(f"\nModelo guardado: models/rf_biobert.pkl")

# Grafico comparativo
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("RF + BioBERT Embeddings vs Baseline TF-IDF", fontsize=13)

top30 = label_metrics.head(30)
axes[0].barh(top30["label"][::-1], top30["f1"][::-1], color="darkorange")
axes[0].set_title("F1 por etiqueta (Top 30)")
axes[0].set_xlabel("F1 Score")
axes[0].axvline(f1_macro, color="red", linestyle="--", label=f"F1 macro={f1_macro:.3f}")
axes[0].legend()

categories = ["F1 macro", "F1 micro", "F1 samples", "Precision", "Recall"]
baseline   = [0.1070,     0.1039,     0.1052,       0.0636,      0.4669]
biobert    = [f1_macro,   f1_micro,   f1_samples,   precision,   recall]
x = np.arange(len(categories))
w = 0.35
axes[1].bar(x - w/2, baseline, w, label="Baseline TF-IDF", color="steelblue")
axes[1].bar(x + w/2, biobert,  w, label="RF + BioBERT",    color="darkorange")
axes[1].set_xticks(x); axes[1].set_xticklabels(categories)
axes[1].set_ylim(0, 1); axes[1].set_title("Comparativa de metricas")
axes[1].legend()
for i, v in enumerate(biobert):
    axes[1].text(i + w/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

plt.tight_layout()
out = OUTPUTS_DIR / "figures" / "rf_biobert_metrics.png"
plt.savefig(out, dpi=150)
print(f"Grafico guardado: {out}")
