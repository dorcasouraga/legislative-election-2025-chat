from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# ENV
# ============================================================

load_dotenv()

DATA_PATH = Path("data/processed/results_clean.parquet")
OUT_PATH = Path("data/rag/rag_index.parquet")

client = OpenAI()

# ============================================================
# Row → Text (CRITIQUE POUR RAG)
# ============================================================

def row_to_text(row) -> str:
    parts = []

    if pd.notna(row.get("region")):
        parts.append(f"Région : {row['region']}.")

    if pd.notna(row.get("circonscription")):
        parts.append(f"Circonscription : {row['circonscription']}.")

    if pd.notna(row.get("parti")):
        parts.append(f"Parti : {row['parti']}.")

    if pd.notna(row.get("candidat")):
        parts.append(f"Candidat : {row['candidat']}.")

    if pd.notna(row.get("voix")):
        parts.append(f"Voix : {int(row['voix'])}.")

    if pd.notna(row.get("pourcentage_voix")):
        parts.append(f"Pourcentage : {row['pourcentage_voix']}%.")

    if pd.notna(row.get("taux_participation")):
        parts.append(f"Taux de participation : {row['taux_participation']}%.")

    # 🔥 INDISPENSABLE pour répondre à "Qui a gagné ?"
    if "elu" in row:
        parts.append(f"Élu : {'Oui' if bool(row['elu']) else 'Non'}.")

    # 🔎 Provenance page
    if "page_source" in row and pd.notna(row["page_source"]):
        parts.append(f"Page : {int(row['page_source'])}.")

    return " ".join(parts)

# ============================================================
# Build RAG index
# ============================================================

def build_index():
    print("📥 Loading data:", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)

    # Sécurité
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)

    # Texte pour RAG
    print("🧱 Building RAG text...")
    df["text"] = df.apply(row_to_text, axis=1)

    # Embeddings
    texts = df["text"].tolist()
    embeddings = []

    BATCH = 64
    print("🧠 Generating embeddings...")
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        embeddings.extend([d.embedding for d in resp.data])

    df["embedding"] = embeddings

    # Colonnes finales — NE PAS RÉDUIRE
    keep_cols = [
        "row_id",
        "region",
        "circonscription",
        "candidat",
        "parti",
        "elu",
        "page_source",
        "text",
        "embedding",
    ]
    df = df[keep_cols].reset_index(drop=True)
    df["row_id"] = df.index.astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"✅ RAG index saved to {OUT_PATH}")
    print("Rows:", df.shape[0])
if __name__ == "__main__":
    build_index()
