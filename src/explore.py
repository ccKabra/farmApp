"""
Etapa 2a: Análisis exploratorio del dataset procesado.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from config import DATA_PROCESSED, OUTPUTS_DIR

df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)

print("=== SHAPE ===")
print(f"Filas: {len(df):,}  |  Columnas: {list(df.columns)}\n")

print("=== VALORES NULOS ===")
print(df.isnull().sum(), "\n")

print("=== SEXO ===")
print(df["sex"].value_counts(), "\n")

# Edad
df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
print("=== EDAD (años) ===")
print(df["age_years"].describe().round(1), "\n")

# Reacciones: cuántas por caso
df["n_reactions"] = df["reactions"].str.split("|").apply(len)
print("=== REACCIONES POR CASO ===")
print(df["n_reactions"].describe().round(1), "\n")

# Top 20 reacciones más frecuentes
all_reac = [r for row in df["reactions"].dropna() for r in row.split("|")]
top_reac = Counter(all_reac).most_common(20)
print("=== TOP 20 REACCIONES ===")
for r, n in top_reac:
    print(f"  {r:<45} {n:>5}")

# Top 20 fármacos más frecuentes
all_drugs = [d for row in df["drug"].dropna() for d in row.split("|")]
top_drugs = Counter(all_drugs).most_common(20)
print("\n=== TOP 20 FÁRMACOS ===")
for d, n in top_drugs:
    print(f"  {d:<45} {n:>5}")

# --- Gráficos ---
fig_dir = OUTPUTS_DIR / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Análisis exploratorio FAERS Q1 2026 — muestra 8k", fontsize=14)

# 1. Distribución de edad
axes[0, 0].hist(df["age_years"].dropna(), bins=30, color="steelblue", edgecolor="white")
axes[0, 0].set_title("Distribución de edad")
axes[0, 0].set_xlabel("Años")
axes[0, 0].set_ylabel("Frecuencia")

# 2. Sexo
sex_counts = df["sex"].value_counts()
axes[0, 1].bar(sex_counts.index, sex_counts.values, color=["steelblue", "salmon", "gray"])
axes[0, 1].set_title("Distribución por sexo")

# 3. Número de reacciones por caso
axes[1, 0].hist(df["n_reactions"], bins=30, color="darkorange", edgecolor="white")
axes[1, 0].set_title("Reacciones adversas por caso")
axes[1, 0].set_xlabel("Cantidad de reacciones")

# 4. Top 15 reacciones
reac_names = [r for r, _ in top_reac[:15]]
reac_counts = [n for _, n in top_reac[:15]]
axes[1, 1].barh(reac_names[::-1], reac_counts[::-1], color="mediumseagreen")
axes[1, 1].set_title("Top 15 reacciones más frecuentes")

plt.tight_layout()
out = fig_dir / "eda.png"
plt.savefig(out, dpi=150)
print(f"\nGráfico guardado en: {out}")
