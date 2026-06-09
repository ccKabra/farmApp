"""
Etapa 7: Visualizaciones — grafo fármaco<->efecto, heatmap, red de co-ocurrencias.
Output: outputs/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter, defaultdict
from config import DATA_PROCESSED, OUTPUTS_DIR

df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)
df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")

MIN_REACTION_FREQ = 50
all_reac = [r for row in df["reactions"].dropna() for r in row.split("|")]
freq_reac = {r for r, n in Counter(all_reac).items() if n >= MIN_REACTION_FREQ}

# Pares fármaco-reacción
drug_reac_pairs = []
for _, row in df.iterrows():
    drugs = row["drug"].split("|") if pd.notna(row["drug"]) else []
    reacs = [r for r in row["reactions"].split("|") if r in freq_reac] if pd.notna(row["reactions"]) else []
    for d in drugs:
        for r in reacs:
            drug_reac_pairs.append((d, r))

pair_counts = Counter(drug_reac_pairs)

# Top fármacos y reacciones para visualizaciones manejables
top_drugs = [d for d, _ in Counter(df["drug"].dropna().str.split("|").explode()).most_common(15)]
top_reacs = [r for r, _ in Counter(all_reac).most_common(20) if r in freq_reac]

# ── 1. Heatmap fármaco × reacción ─────────────────────────────────────────────
print("Generando heatmap...")
matrix = pd.DataFrame(0, index=top_drugs, columns=top_reacs)
for (d, r), n in pair_counts.items():
    if d in matrix.index and r in matrix.columns:
        matrix.loc[d, r] = n

# Normalizar por fármaco (frecuencia relativa)
matrix_norm = matrix.div(matrix.sum(axis=1) + 1e-9, axis=0)

fig, ax = plt.subplots(figsize=(18, 8))
sns.heatmap(matrix_norm, annot=False, cmap="YlOrRd", ax=ax,
            linewidths=0.3, cbar_kws={"label": "Frecuencia relativa"})
ax.set_title("Co-ocurrencia fármaco × efecto adverso (frecuencia relativa)", fontsize=13)
ax.set_xlabel("Efecto adverso"); ax.set_ylabel("Fármaco")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "figures" / "heatmap_drug_reaction.png", dpi=150)
plt.close()
print("  heatmap_drug_reaction.png")

# ── 2. Grafo bipartito fármaco <-> reacción (networkx + matplotlib) ────────────
print("Generando grafo bipartito...")
G = nx.Graph()

# Nodos
for d in top_drugs[:10]:
    G.add_node(d, node_type="drug")
for r in top_reacs[:15]:
    G.add_node(r, node_type="reaction")

# Aristas con peso
for (d, r), n in pair_counts.items():
    if d in G.nodes and r in G.nodes and n >= 5:
        G.add_edge(d, r, weight=n)

pos = nx.bipartite_layout(G, nodes=[n for n, d in G.nodes(data=True) if d["node_type"] == "drug"])

drug_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "drug"]
reac_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "reaction"]
edge_weights = [G[u][v]["weight"] / 20 for u, v in G.edges()]

fig, ax = plt.subplots(figsize=(14, 10))
nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color="steelblue",
                       node_size=800, ax=ax, label="Farmaco")
nx.draw_networkx_nodes(G, pos, nodelist=reac_nodes, node_color="darkorange",
                       node_size=500, ax=ax, label="Reaccion")
nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color="gray", ax=ax)
nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
ax.set_title("Grafo bipartito: Farmacos <-> Efectos adversos", fontsize=13)
ax.legend(scatterpoints=1); ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "figures" / "grafo_bipartito.png", dpi=150)
plt.close()
print("  grafo_bipartito.png")

# ── 3. Red de co-ocurrencia de reacciones ─────────────────────────────────────
print("Generando red de co-ocurrencia de reacciones...")
cooc = defaultdict(int)
for _, row in df.iterrows():
    reacs = [r for r in row["reactions"].split("|") if r in freq_reac] if pd.notna(row["reactions"]) else []
    reacs = [r for r in reacs if r in top_reacs]
    for i in range(len(reacs)):
        for j in range(i+1, len(reacs)):
            key = tuple(sorted([reacs[i], reacs[j]]))
            cooc[key] += 1

G2 = nx.Graph()
for r in top_reacs:
    G2.add_node(r)
for (r1, r2), n in cooc.items():
    if n >= 10:
        G2.add_edge(r1, r2, weight=n)

pos2 = nx.spring_layout(G2, seed=42, k=2)
degrees = dict(G2.degree(weight="weight"))
node_sizes = [degrees.get(n, 1) * 3 + 200 for n in G2.nodes()]
edge_widths = [G2[u][v]["weight"] / 15 for u, v in G2.edges()]

fig, ax = plt.subplots(figsize=(13, 10))
nx.draw_networkx_nodes(G2, pos2, node_size=node_sizes, node_color=node_sizes,
                       cmap=cm.YlOrRd, ax=ax)
nx.draw_networkx_edges(G2, pos2, width=edge_widths, alpha=0.5, edge_color="gray", ax=ax)
nx.draw_networkx_labels(G2, pos2, font_size=7, ax=ax)
ax.set_title("Red de co-ocurrencia de efectos adversos", fontsize=13)
ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "figures" / "red_coocurrencia_reacciones.png", dpi=150)
plt.close()
print("  red_coocurrencia_reacciones.png")

# ── 4. Grafo interactivo Plotly (HTML) ────────────────────────────────────────
print("Generando grafo interactivo Plotly...")
edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]; x1, y1 = pos[v]
    edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

node_x = [pos[n][0] for n in G.nodes()]
node_y = [pos[n][1] for n in G.nodes()]
node_text = list(G.nodes())
node_colors = ["steelblue" if G.nodes[n]["node_type"] == "drug" else "darkorange" for n in G.nodes()]

fig_plotly = go.Figure()
fig_plotly.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                line=dict(width=0.8, color="gray"), hoverinfo="none"))
fig_plotly.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                marker=dict(size=14, color=node_colors),
                                text=node_text, textposition="top center",
                                hoverinfo="text"))
fig_plotly.update_layout(
    title="Grafo interactivo: Farmacos (azul) <-> Efectos adversos (naranja)",
    showlegend=False, hovermode="closest",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=700
)
out_html = OUTPUTS_DIR / "grafo_interactivo.html"
fig_plotly.write_html(str(out_html))
print(f"  grafo_interactivo.html -> {out_html}")

print("\nTodas las visualizaciones generadas.")
