"""
Etapa 2b: Construcción de features y matriz multi-label.

Input:  data/processed/dataset.csv
Output: data/processed/X.csv, data/processed/Y.csv, data/processed/label_names.txt
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from config import DATA_PROCESSED, RANDOM_SEED

MIN_REACTION_FREQ = 50  # solo etiquetas con al menos N casos (evita ruido)
MIN_DRUG_FREQ = 10      # solo fármacos con al menos N apariciones

df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)
df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")

# ── Features ──────────────────────────────────────────────────────────────────

# 1. Sexo → one-hot
sex_dummies = pd.get_dummies(df["sex"], prefix="sex")

# 2. Edad → normalizada 0-1 (imputar mediana si falta)
age_median = df["age_years"].median()
df["age_norm"] = df["age_years"].fillna(age_median) / 100.0

# 3. Fármaco principal → one-hot (top fármacos frecuentes, resto = "OTHER")
from collections import Counter
all_drugs = [d for row in df["drug"].dropna() for d in row.split("|")]
freq_drugs = {d for d, n in Counter(all_drugs).items() if n >= MIN_DRUG_FREQ}

def encode_drugs(drug_str):
    drugs = drug_str.split("|") if pd.notna(drug_str) else []
    known = [d for d in drugs if d in freq_drugs]
    return known if known else ["OTHER"]

df["drug_list"] = df["drug"].apply(encode_drugs)

mlb_drug = MultiLabelBinarizer()
drug_matrix = mlb_drug.fit_transform(df["drug_list"])
drug_df = pd.DataFrame(drug_matrix, columns=[f"drug_{d}" for d in mlb_drug.classes_])

# 4. Indicaciones → bag-of-words simple (top palabras en indicaciones)
from sklearn.feature_extraction.text import TfidfVectorizer

indi_texts = df["indications"].fillna("unknown")
tfidf = TfidfVectorizer(max_features=100, min_df=5, token_pattern=r"[A-Za-z][A-Za-z ]{2,}")
indi_matrix = tfidf.fit_transform(indi_texts).toarray()
indi_df = pd.DataFrame(indi_matrix, columns=[f"indi_{t}" for t in tfidf.get_feature_names_out()])

# Combinar todo en X
X = pd.concat([
    df[["age_norm"]].reset_index(drop=True),
    sex_dummies.reset_index(drop=True),
    drug_df.reset_index(drop=True),
    indi_df.reset_index(drop=True),
], axis=1)

print(f"Features (X): {X.shape[0]:,} filas × {X.shape[1]:,} columnas")

# ── Etiquetas (Y) ─────────────────────────────────────────────────────────────

# Filtrar reacciones poco frecuentes
from collections import Counter
all_reac = [r for row in df["reactions"].dropna() for r in row.split("|")]
freq_reac = {r for r, n in Counter(all_reac).items() if n >= MIN_REACTION_FREQ}

print(f"Etiquetas únicas con freq >= {MIN_REACTION_FREQ}: {len(freq_reac)}")

df["reaction_list"] = df["reactions"].apply(
    lambda s: [r for r in s.split("|") if r in freq_reac] if pd.notna(s) else []
)

# Eliminar filas sin ninguna etiqueta frecuente
mask = df["reaction_list"].apply(len) > 0
X = X[mask].reset_index(drop=True)
df = df[mask].reset_index(drop=True)
print(f"Casos con al menos una etiqueta frecuente: {mask.sum():,}")

mlb_reac = MultiLabelBinarizer()
Y = mlb_reac.fit_transform(df["reaction_list"])
Y_df = pd.DataFrame(Y, columns=mlb_reac.classes_)

print(f"Etiquetas (Y): {Y_df.shape[0]:,} filas × {Y_df.shape[1]:,} etiquetas")
print(f"Densidad de la matriz Y: {Y.mean():.4f}  (promedio reacciones activas por caso)")

# ── Guardar ───────────────────────────────────────────────────────────────────
X.to_csv(DATA_PROCESSED / "X.csv", index=False)
Y_df.to_csv(DATA_PROCESSED / "Y.csv", index=False)

with open(DATA_PROCESSED / "label_names.txt", "w") as f:
    f.write("\n".join(mlb_reac.classes_))

print("\nArchivos guardados:")
print(f"  X.csv          : {X.shape}")
print(f"  Y.csv          : {Y_df.shape}")
print(f"  label_names.txt: {len(mlb_reac.classes_)} etiquetas")
print(f"\nTop 10 etiquetas más frecuentes:")
print(Y_df.sum().sort_values(ascending=False).head(10).to_string())
