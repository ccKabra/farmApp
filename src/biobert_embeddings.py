"""
Etapa 4a: Generar embeddings BioBERT para las indicaciones de cada caso.
Reemplaza el TF-IDF basico por representaciones semanticas densas.

Output: data/processed/X_biobert.csv
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from config import DATA_PROCESSED, BIOBERT_MODEL, DEVICE

BATCH_SIZE = 64

def mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def get_embeddings(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
    all_embeddings = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        embeddings = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        all_embeddings.append(embeddings.cpu().numpy())
        if (i // batch_size) % 5 == 0:
            print(f"  Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
    return np.vstack(all_embeddings)

print(f"Cargando BioBERT desde HuggingFace: {BIOBERT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
model = AutoModel.from_pretrained(BIOBERT_MODEL).to(DEVICE)
print(f"Modelo cargado en {DEVICE.upper()}.")

df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)
Y = pd.read_csv(DATA_PROCESSED / "Y.csv")

# Alinear df con Y (mismos indices que prepare_features.py)
df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
from collections import Counter
MIN_REACTION_FREQ = 50
all_reac = [r for row in df["reactions"].dropna() for r in row.split("|")]
freq_reac = {r for r, n in Counter(all_reac).items() if n >= MIN_REACTION_FREQ}
df["reaction_list"] = df["reactions"].apply(
    lambda s: [r for r in s.split("|") if r in freq_reac] if pd.notna(s) else []
)
mask = df["reaction_list"].apply(len) > 0
df = df[mask].reset_index(drop=True)

# Texto a embeber: "drug: X. indication: Y"
texts = (
    "drug: " + df["drug"].fillna("unknown").str[:100] +
    ". indication: " + df["indications"].fillna("unknown").str[:200]
).tolist()

print(f"\nGenerando embeddings para {len(texts):,} casos...")
embeddings = get_embeddings(texts, tokenizer, model, DEVICE)
print(f"Embeddings shape: {embeddings.shape}")

# Reconstruir X combinando features demograficos + embeddings BioBERT
X_old = pd.read_csv(DATA_PROCESSED / "X.csv")

# Tomar solo columnas no-indi del X anterior (edad, sexo, fármaco)
non_indi_cols = [c for c in X_old.columns if not c.startswith("indi_")]
X_demo_drug = X_old[non_indi_cols]

emb_df = pd.DataFrame(embeddings, columns=[f"bb_{i}" for i in range(embeddings.shape[1])])
X_biobert = pd.concat([X_demo_drug.reset_index(drop=True), emb_df], axis=1)

out = DATA_PROCESSED / "X_biobert.csv"
X_biobert.to_csv(out, index=False)
print(f"X_biobert guardado: {X_biobert.shape} en {out}")
