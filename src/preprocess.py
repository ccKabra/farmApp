"""
Etapa 1: Carga, limpieza y sampleo de datos FAERS Q1 2026.
Produce: data/processed/dataset.csv con 8000 casos listos para modelar.
"""

import pandas as pd
import numpy as np
from config import (
    FAERS_DEMO, FAERS_DRUG, FAERS_REAC, FAERS_INDI, FAERS_OUTC,
    DATA_PROCESSED, SAMPLE_SIZE, RANDOM_SEED
)

SEP = "$"


def load_demo():
    df = pd.read_csv(FAERS_DEMO, sep=SEP, dtype=str, on_bad_lines="skip", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    # Conservar solo columnas útiles
    cols = ["primaryid", "age", "age_cod", "sex", "wt", "wt_cod", "reporter_country"]
    df = df[[c for c in cols if c in df.columns]].copy()

    # Normalizar edad a años
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    age_map = {"DEC": 10, "YR": 1, "MON": 1/12, "WK": 1/52, "DY": 1/365, "HR": 1/8760}
    df["age_cod"] = df["age_cod"].str.strip().str.upper()
    df["age_years"] = df.apply(
        lambda r: r["age"] * age_map.get(r["age_cod"], 1) if pd.notna(r["age"]) else np.nan,
        axis=1
    )
    df = df.drop(columns=["age", "age_cod"])

    # Normalizar sexo
    df["sex"] = df["sex"].str.strip().str.upper().map({"M": "M", "F": "F"}).fillna("U")
    return df.drop_duplicates("primaryid")


def load_drugs():
    df = pd.read_csv(FAERS_DRUG, sep=SEP, dtype=str, on_bad_lines="skip", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    # Solo fármacos primarios sospechosos (role_cod PS = Primary Suspect)
    df = df[df["role_cod"].str.strip().str.upper() == "PS"].copy()
    # Usar ingrediente activo si existe, sino nombre comercial
    df["drug"] = df["prod_ai"].str.strip().fillna(df["drugname"].str.strip())
    df["drug"] = df["drug"].str.upper().str.strip()
    # Agrupar múltiples drogas por caso (puede haber más de una PS)
    drugs = (
        df.groupby("primaryid")["drug"]
        .apply(lambda x: "|".join(sorted(set(x.dropna()))))
        .reset_index()
    )
    return drugs


def load_reactions():
    df = pd.read_csv(FAERS_REAC, sep=SEP, dtype=str, on_bad_lines="skip", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df["pt"] = df["pt"].str.strip().str.title()
    reac = (
        df.groupby("primaryid")["pt"]
        .apply(lambda x: "|".join(sorted(set(x.dropna()))))
        .reset_index()
        .rename(columns={"pt": "reactions"})
    )
    return reac


def load_indications():
    df = pd.read_csv(FAERS_INDI, sep=SEP, dtype=str, on_bad_lines="skip", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df["indi_pt"] = df["indi_pt"].str.strip().str.title()
    indi = (
        df.groupby("primaryid")["indi_pt"]
        .apply(lambda x: "|".join(sorted(set(x.dropna()))))
        .reset_index()
        .rename(columns={"indi_pt": "indications"})
    )
    return indi


def build_dataset():
    print("Cargando archivos FAERS...")
    demo  = load_demo()
    drugs = load_drugs()
    reac  = load_reactions()
    indi  = load_indications()

    print(f"  DEMO:  {len(demo):,} pacientes")
    print(f"  DRUGS: {len(drugs):,} casos con fármaco PS")
    print(f"  REAC:  {len(reac):,} casos con reacciones")
    print(f"  INDI:  {len(indi):,} casos con indicaciones")

    # Join por primaryid
    df = demo.merge(drugs, on="primaryid", how="inner")
    df = df.merge(reac,  on="primaryid", how="inner")
    df = df.merge(indi,  on="primaryid", how="left")

    # Eliminar filas sin información mínima
    df = df.dropna(subset=["drug", "reactions"])
    df = df[df["drug"].str.len() > 0]
    df = df[df["reactions"].str.len() > 0]

    print(f"\nCasos válidos antes de samplear: {len(df):,}")

    # Samplear
    n = min(SAMPLE_SIZE, len(df))
    df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Muestra final: {n:,} casos")

    # Guardar
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "dataset.csv"
    df.to_csv(out, index=False)
    print(f"\nGuardado en: {out}")
    print(df.head(3).to_string())
    return df


if __name__ == "__main__":
    build_dataset()
