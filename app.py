"""
Demo: Prediccion de efectos adversos de farmacos
Ejecutar: venv\Scripts\streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import plotly.graph_objects as go

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models" / "biobert_finetuned"
DATA_DIR   = ROOT / "data" / "processed"
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Modelo ────────────────────────────────────────────────────────────────────
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

@st.cache_resource
def load_model():
    label_df = pd.read_csv(MODELS_DIR / "label_names.csv")
    label_names = label_df["label"].tolist()
    thresholds = np.load(MODELS_DIR / "thresholds.npy")
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR))
    bert_base = AutoModel.from_pretrained(BIOBERT_MODEL)
    model = BioBERTClassifier(bert_base, len(label_names)).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / "model.pt", map_location=DEVICE))
    model.eval()
    return model, tokenizer, label_names, thresholds

@st.cache_data
def get_top_drugs():
    df = pd.read_csv(DATA_DIR / "dataset.csv", dtype=str)
    from collections import Counter
    all_drugs = [d for row in df["drug"].dropna() for d in row.split("|")]
    return [d for d, _ in Counter(all_drugs).most_common(50)]

def predict(model, tokenizer, label_names, thresholds, drug, age, sex, indication=""):
    sex_str = {"Masculino": "male", "Femenino": "female", "No especificado": "unknown"}[sex]
    text = f"patient age {age} years {sex_str}. drug: {drug}. indication: {indication or 'unknown'}"
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=128, padding="max_length").to(DEVICE)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    results = [
        {"effect": label_names[i], "probability": float(probs[i]), "predicted": probs[i] >= thresholds[i]}
        for i in range(len(label_names))
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results, text

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FarmApp - Prediccion de Efectos Adversos",
                   page_icon="💊", layout="wide")

st.title("💊 Sistema de Prediccion de Efectos Adversos")
st.caption("Modelo BioBERT fine-tuned sobre FDA FAERS Q1 2026 — Proyecto Mineria de Texto y Aprendizaje Automatico")

with st.spinner("Cargando modelo BioBERT..."):
    model, tokenizer, label_names, thresholds = load_model()
    top_drugs = get_top_drugs()

st.success(f"Modelo listo | {len(label_names)} efectos adversos | Device: {DEVICE.upper()}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Datos del paciente")

drug_input = st.sidebar.selectbox("Farmaco", options=[""] + top_drugs + ["Otro (escribir abajo)"])
if drug_input == "Otro (escribir abajo)":
    drug_input = st.sidebar.text_input("Nombre del farmaco (ingrediente activo)").upper()

age_input = st.sidebar.slider("Edad (anos)", min_value=0, max_value=100, value=55)
sex_input = st.sidebar.selectbox("Sexo", ["No especificado", "Masculino", "Femenino"])
indication_input = st.sidebar.text_input("Indicacion (opcional)", placeholder="ej: Type 2 Diabetes Mellitus")

predict_btn = st.sidebar.button("Predecir efectos adversos", type="primary")

# ── Prediccion ────────────────────────────────────────────────────────────────
if predict_btn and drug_input:
    with st.spinner("Analizando..."):
        results, input_text = predict(model, tokenizer, label_names, thresholds,
                                       drug_input, age_input, sex_input, indication_input)

    predicted = [r for r in results if r["predicted"]]
    not_predicted = [r for r in results if not r["predicted"]][:20]

    col1, col2, col3 = st.columns(3)
    col1.metric("Farmaco", drug_input)
    col2.metric("Efectos adversos predichos", len(predicted))
    col3.metric("Confianza maxima", f"{results[0]['probability']:.1%}")

    st.markdown("---")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("Efectos adversos predichos")
        if predicted:
            pred_df = pd.DataFrame(predicted)[["effect", "probability"]]
            pred_df["probability"] = pred_df["probability"].map("{:.1%}".format)
            pred_df.columns = ["Efecto adverso", "Probabilidad"]
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
        else:
            st.info("No se predijeron efectos adversos con los umbrales actuales.")

    with col_b:
        st.subheader("Top 15 probabilidades")
        top15 = results[:15]
        fig = go.Figure(go.Bar(
            x=[r["probability"] for r in top15],
            y=[r["effect"] for r in top15],
            orientation="h",
            marker_color=["#e74c3c" if r["predicted"] else "#3498db" for r in top15],
            text=[f"{r['probability']:.1%}" for r in top15],
            textposition="outside"
        ))
        fig.update_layout(
            height=450, margin=dict(l=0, r=40, t=10, b=10),
            xaxis_title="Probabilidad", yaxis=dict(autorange="reversed"),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Texto de entrada al modelo"):
        st.code(input_text)

    with st.expander("Tabla completa de probabilidades"):
        all_df = pd.DataFrame(results)[["effect", "probability", "predicted"]]
        all_df["probability"] = all_df["probability"].map("{:.3f}".format)
        all_df.columns = ["Efecto adverso", "Probabilidad", "Predicho"]
        st.dataframe(all_df, use_container_width=True, hide_index=True)

elif predict_btn and not drug_input:
    st.warning("Por favor selecciona o escribe un farmaco.")
else:
    st.info("Selecciona un farmaco y presiona 'Predecir efectos adversos' para comenzar.")
    st.image(str(ROOT / "outputs" / "figures" / "heatmap_drug_reaction.png"),
             caption="Co-ocurrencia fármaco x efecto adverso en FAERS Q1 2026", use_container_width=True)
