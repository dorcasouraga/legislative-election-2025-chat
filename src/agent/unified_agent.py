
from __future__ import annotations

import json
import os
import re
import unicodedata
import time
from difflib import get_close_matches
from typing import Dict, Optional, List, Tuple, Any
from contextlib import contextmanager

import duckdb
import pandas as pd
import altair as alt

from dotenv import load_dotenv
from langfuse import Langfuse
import numpy as np
from openai import OpenAI
from pathlib import Path


# ============================================================
# 🔭 OBSERVABILITY HELPERS (AJOUT SEUL)
# ============================================================

@contextmanager
def timed(step: str, bag: Dict):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        bag["timings_ms"][step] = round((time.perf_counter() - t0) * 1000, 2)


def safe_llm_usage(resp) -> Dict:
    try:
        u = resp.usage
        if not u:
            return {}
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        }
    except Exception:
        return {}


# ============================================================
# INIT
# ============================================================

load_dotenv()

lf = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

DB_PATH = "src/db/results.duckdb"
con = duckdb.connect(DB_PATH, read_only=True)

client = OpenAI()

RAG_INDEX_PATH = Path("data/rag/rag_index.parquet")
RAG_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

RAG_TOPK = 20
RAG_DF = None

RESOLVE_THRESHOLD = 0.78


def exact_norm_match(question: str, pool: Dict[str, str]) -> Optional[str]:
    """
    Exact match after strong normalization.
    Ex:
      question: "top 5 candidats dans la mé"
      key: "la me"
    """
    q = " " + norm(question) + " "
    for key, value in pool.items():
        if re.search(rf"(^|\s){re.escape(key)}(\s|$)", q):
            return value
    return None



# ============================================================
# NORMALIZATION
# ============================================================

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def norm(text: str) -> str:
    if not text:
        return ""
    text = strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


# ============================================================
# ENTITY POOLS
# ============================================================

STOPWORDS = {
    "qui","a","au","aux","de","des","du","la","le","les","un","une","et","est",
    "dans","pour","par","gagne","gagné","vainqueur","top","combien","nombre",
    "total","taux","participation","classement","rang","siege","sieges"
}


def build_pool(values: List[str]) -> Dict[str, str]:
    return {norm(v): v for v in values if isinstance(v, str) and v.strip()}


def load_pools():
    regions = con.execute("SELECT DISTINCT region FROM resultats").fetchdf()["region"].dropna().tolist()
    circos  = con.execute("SELECT DISTINCT circonscription FROM resultats").fetchdf()["circonscription"].dropna().tolist()
    partis  = con.execute("SELECT DISTINCT parti FROM resultats").fetchdf()["parti"].dropna().tolist()
    return build_pool(regions), build_pool(circos), build_pool(partis)


REGIONS, CIRCOS, PARTIS = load_pools()


# ============================================================
# ENTITY RESOLUTION (ROBUST)
# ============================================================

def extract_tokens(question: str) -> List[str]:
    return [
        t for t in norm(question).split()
        if t not in STOPWORDS and len(t) >= 3
    ]
    
def resolve_by_inclusion(question: str, pool: Dict[str, str]) -> Optional[str]:
    tokens = extract_tokens(question)

    for label_norm, original in pool.items():
        for tok in tokens:
            if tok in label_norm:
                return original

    return None


def resolve(question: str, pool: Dict[str, str]) -> Tuple[Optional[str], float]:
    tokens = [t for t in norm(question).split() if t not in STOPWORDS and len(t) >= 3]

    best_val, best_score = None, 0.0

    for n in range(min(6, len(tokens)), 0, -1):
        for g in ngrams(tokens, n):
            match = get_close_matches(g, pool.keys(), n=1, cutoff=0.6)
            if match:
                key = match[0]
                score = len(g) / max(len(key), 1)
                if score > best_score:
                    best_val = pool[key]
                    best_score = score

    if best_val is None:
        q = norm(question)
        for k, v in pool.items():
            if k in q:
                return v, 0.55

    return best_val, best_score

def resolve_entity(question: str, pool: Dict[str, str]) -> Optional[str]:
    # 1️⃣ Match exact normalisé
    exact = exact_norm_match(question, pool)
    if exact:
        return exact

    # 2️⃣ Match par inclusion de token
    inc = resolve_by_inclusion(question, pool)
    if inc:
        return inc

    # 3️⃣ Fuzzy (dernier recours)
    val, score = resolve(question, pool)
    if score >= 0.6:
        return val

    return None



# ============================================================
# INTENT
# ============================================================


def analyze_intent(question: str) -> Dict[str, bool]:
    q = norm(question)
    return {
        "analytics": any(k in q for k in [
            "combien","nombre","total","taux","top","classement","rang","siege","sieges"
        ]),
        "top": any(k in q for k in ["top","classement","rang"]),
        "taux": any(k in q for k in ["taux","participation"]),
        "winner": any(k in q for k in ["gagnant","gagne","elu","élu","vainqueur"]),
        "chart": any(k in q for k in ["graph","graphique","histogramme","diagramme"]),
    }
    
    
 
# ============================================================
# GUARDRAILS
# ============================================================
def is_malicious(q: str) -> bool:
    nq = norm(q)

    # helper: match mots entiers seulement
    def has_word(word: str) -> bool:
        return re.search(rf"\b{re.escape(word)}\b", nq) is not None

    # 1) SQL destructive
    for w in ["drop", "delete", "truncate", "alter", "insert", "update", "api", "cle api"]:
        if has_word(w):
            return True

    # 2) Secrets / prompt injection (STRICT word match)
    for w in ["api", "key", "token", "secret", "prompt", "cle", "cle api"]:
        if has_word(w):
            return True

    # 3) Exfiltration / bypass
    for w in [
        "ignore tes regles",
        "ignore tes règles",
        "sans limit",
        "no limit",
        "toute la base",
        "dump",
        "leak",
        "show me all",
        "all rows",
    ]:
        if w in nq:
            return True

    return False



def is_out_of_scope(q: str) -> bool:
    return any(k in norm(q) for k in [
        "meteo","weather","temps","president","gouvernement"
    ])


# ============================================================
# SQL HELPERS
# ============================================================

def render_sql(sql: str, params: Optional[List[Any]]) -> str:
    if not params:
        return sql.strip()
    rendered = sql
    for p in params:
        if isinstance(p, str):
            v = "'" + p.replace("'", "''") + "'"
        else:
            v = str(p)
        rendered = rendered.replace("?", v, 1)
    return rendered.strip()

# ============================================================
# PROVENANCE HELPERS (BONUS)
# ============================================================

def sql_provenance(
    table_id: str,
    query_type: str,
    filters: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    Provenance for SQL answers (aggregation / ranking).
    """
    return {
        "type": "sql",
        "table_id": table_id,
        "query_type": query_type,
        "filters": filters or {},
    }

# ============================================================
# SQL QUERIES
# ============================================================

def seats_by_party(party: str):
    sql = """
    SELECT parti, COUNT(*) AS sieges
    FROM resultats
    WHERE elu = TRUE AND parti = ?
    GROUP BY parti
    """
    return sql, [party], con.execute(sql, [party]).fetchdf()


def top_candidates_region(region: str):
    sql = """
    SELECT candidat, SUM(voix) AS total_voix
    FROM resultats
    WHERE region = ?
    GROUP BY candidat
    ORDER BY total_voix DESC
    LIMIT 5
    """
    return sql, [region], con.execute(sql, [region]).fetchdf()


def turnout_by_region():
    sql = """
    SELECT region, AVG(taux_participation) AS taux_participation
    FROM resultats
    GROUP BY region
    ORDER BY taux_participation DESC
    """
    return sql, [], con.execute(sql).fetchdf()


def winners_by_party():
    sql = """
    SELECT parti, COUNT(*) AS winners
    FROM resultats
    WHERE elu = TRUE
    GROUP BY parti
    ORDER BY winners DESC
    """
    return sql, [], con.execute(sql).fetchdf()


# ============================================================
# CHART FACTORY
# ============================================================

def make_chart(df: pd.DataFrame, title: str):
    if df is None or df.empty or len(df.columns) < 2:
        return None

    x, y = df.columns[:2]

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x, sort="-y"),
            y=y,
            tooltip=[x, y],
        )
        .properties(title=title, width=700, height=350)
    )

def is_safe_commune_token(tok: str) -> bool:
    return len(tok) >= 4 and tok not in {"me", "la", "du", "de"}

def ask(question: str) -> Dict:
    # trace = lf.start_span(name="ask")
    # trace.input = question
    trace = lf.start_span(name="ask")
    trace.input = question
    trace.metadata = {
        "eval_mode": os.getenv("EVAL_MODE", "false")
    }


    # -------- SECURITY --------
    if is_malicious(question):
        trace.end(); lf.flush()
        return {
            "status": "refused",
            "engine": "none",
            "text": "Requête refusée",
            "sql": None,
            "rag_allowed": False
        }

    if is_out_of_scope(question):
        trace.end(); lf.flush()
        return {
            "status": "not_found",
            "engine": "none",
            "text": "Information non disponible",
            "sql": None,
             "rag_allowed": False
        }
    intent = analyze_intent(question)

    region = resolve_entity(question, REGIONS)
    circo  = resolve_entity(question, CIRCOS)
    party  = resolve_entity(question, PARTIS)
    r_score = 1.0 if region else 0.0
    c_score = 1.0 if circo else 0.0



    exact_region = exact_norm_match(question, REGIONS)
    if exact_region:
        region = exact_region
        r_score = 1.0


    # -------- SEATS --------
    if intent["analytics"] and party:
        sql, params, df = seats_by_party(party)
        if not df.empty:
            trace.end(); lf.flush()
            return {
                "status": "ok",
                "engine": "sql",
                "text": df.to_string(index=False),
                "sql": render_sql(sql, params),
                "chart": make_chart(df, f"Sièges – {party}"),
                "provenance": sql_provenance(
                table_id="resultats",
                query_type="aggregation",
                filters={"parti": party, "elu": True}
    )
            }
    # --- CHART : SIEGES PAR PARTI ---
    if intent["analytics"] and intent["chart"] and "siege" in norm(question) and "parti" in norm(question):
        sql = """
        SELECT parti, COUNT(*) AS sieges
        FROM resultats
        WHERE elu = TRUE
        GROUP BY parti
        ORDER BY sieges DESC
        """
        df = con.execute(sql).fetchdf()

        if not df.empty:
            return {
                "status": "ok",
                "engine": "sql",
                "text": df.to_string(index=False),
                "sql": sql.strip(),
                "chart": make_chart(df, "Sièges par parti"),
                "provenance": sql_provenance(
                table_id="resultats",
                query_type="aggregation",
                filters={"parti": party, "elu": True})
            }


    # -------- TOP --------
    if intent["analytics"] and intent["top"] and region and r_score >= RESOLVE_THRESHOLD:
        sql, params, df = top_candidates_region(region)
        if not df.empty:
            trace.end(); lf.flush()
            return {
                "status": "ok",
                "engine": "sql",
                "text": df.to_string(index=False),
                "sql": render_sql(sql, params),
                "chart": make_chart(df, f"Top candidats – {region}"),
                "provenance": sql_provenance(
                table_id="resultats",
                query_type="ranking",
                filters={"region": region}
)

            }

    # -------- TURNOUT --------
    if intent["analytics"] and intent["taux"]:
        sql, params, df = turnout_by_region()
        if not df.empty:
            trace.end(); lf.flush()
            return {
                "status": "ok",
                "engine": "sql",
                "text": df.to_string(index=False),
                "sql": render_sql(sql, params),
                "chart": make_chart(df, "Taux de participation par région"),
                "provenance": sql_provenance(
                table_id="resultats",
                query_type="aggregation",
                filters={"metric": "taux_participation"}
)

            }

    # -------- HISTOGRAMME GAGNANTS --------
    if intent["chart"] and intent["winner"] and "parti" in norm(question):
        sql, params, df = winners_by_party()
        if not df.empty:
            trace.end(); lf.flush()
            return {
                "status": "ok",
                "engine": "sql",
                "text": df.to_string(index=False),
                "sql": render_sql(sql, params),
                "chart": make_chart(df, "Histogramme des gagnants par parti"),
                "provenance": sql_provenance(
                table_id="resultats",
                query_type="aggregation",
                filters={"elu": True}
)

            }
 
    trace.end(); lf.flush()
    
    return {
    "status": "not_found",
    "engine": "none",
    "text": "Information non disponible",
    "sql": None,
    "rag_allowed": True,   # 👈 clé
}



ask_level1 = ask


def contains_entity(question: str, value: str) -> bool:
    """
    Check if a short entity (e.g. 'tiapoum')
    is contained in a longer label
    """
    return norm(question) in norm(value) or norm(value).find(norm(question)) != -1

# ============================================================
# LEVEL 2 — RAG (Hybrid Router)
# ============================================================


def load_rag_index() -> pd.DataFrame:
    if not RAG_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"RAG index not found at {RAG_INDEX_PATH}. "
            f"Run: python build_index.py"
        )
    df = pd.read_parquet(RAG_INDEX_PATH)

    # Sécurité minimale
    required = {"text", "embedding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"RAG index missing columns: {missing}")

    return df

RAG_DF = load_rag_index()


# ---------------------------
# Embeddings
# ---------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    r = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([x.embedding for x in r.data], dtype="float32")


# ---------------------------
# RAG Search
# ---------------------------
def rag_search(question: str, k: int = RAG_TOPK) -> List[Dict]:
    if RAG_DF.empty:
        return []

    # -------------------------------------------------
    # 1️⃣ Extraire entités (clé du bug)
    # # -------------------------------------------------
    # region, r_score = resolve(question, REGIONS)
    # circo, c_score  = resolve(question, CIRCOS)
    
    region = resolve_entity(question, REGIONS)
    circo  = resolve_entity(question, CIRCOS)


    # -------------------------------------------------
    # 2️⃣ Filtrage prioritaire par entité
    # -------------------------------------------------
    df = RAG_DF
    
    if circo:
        df = df[
            df["circonscription"]
            .fillna("")
            .apply(lambda x: norm(circo) in norm(x))
        ]

    elif region:
        df = df[
            df["region"]
            .fillna("")
            .apply(lambda x: norm(region) in norm(x))
        ]


    if df.empty:
        df = RAG_DF

    # -------------------------------------------------
    # 4️⃣ Embedding similarity
    # -------------------------------------------------
    qv = embed_texts([question])[0]
    qv = qv / (np.linalg.norm(qv) + 1e-9)

    mat = np.vstack(df["embedding"].to_numpy())
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    sims = mat @ qv
    idx = np.argsort(-sims)[:k]

    out = []
    for i in idx:
        row = df.iloc[int(i)]
        out.append({
            "score": float(sims[int(i)]),
            "text": row["text"],
            "row_id": row["row_id"],
            "page": row["page_source"],
        })

    return out



# ---------------------------
# LLM grounded answer
# ---------------------------
def rag_answer(question: str, chunks: List[Dict]) -> Dict:
    if not chunks:
        return {
            "status": "not_found",
            "engine": "rag",
            "text": "Information non disponible",
            "sql": None,
        }

    context = "\n\n".join(
        f"[page={c['page']} row_id={c['row_id']}]\n{c['text']}"
        for c in chunks
    )
    prompt = f"""Tu es un assistant d’analyse électorale.

Tu dois répondre UNIQUEMENT à partir des SOURCES fournies.
Toute information absente des sources doit être refusée.

====================
RÈGLES DE RAISONNEMENT
====================

1) Résolution géographique (AUTORISÉE)
- La question peut mentionner :
  - une commune (ex: "Tiapoum", "Adzopé")
  - une circonscription
  - une région (ex: "LA ME")
- Une source est pertinente si :
  - la circonscription mentionnée dans la source CONTIENT explicitement
    la commune demandée
  - OU si la région demandée correspond exactement à la région de la source

2) Condition d’élection (OBLIGATOIRE)
- Tu dois vérifier explicitement :
  Élu : Oui / True
- Un candidat sans "Élu : Oui / True" NE PEUT PAS être considéré comme gagnant.

3) Cas avec PLUSIEURS candidats élus
- Si plusieurs candidats ÉLUS correspondent à la zone demandée :
  → retourne UNIQUEMENT celui ayant le PLUS GRAND NOMBRE DE VOIX.

4) Interdictions ABSOLUES
Tu ne dois JAMAIS :
- inventer un gagnant
- extrapoler à partir d’une autre région
- déduire un élu sans "Élu : Oui / True"
- répondre si la zone demandée ne correspond pas clairement aux sources

5) Cas sans résultat valide
- Si AUCUN candidat ÉLU ne correspond exactement à la zone demandée :
  → réponds STRICTEMENT :
  
  "Information non disponible."

====================
FORMAT DE RÉPONSE
====================

Si un gagnant est trouvé, réponds de maniere concise et naturel avec les details

====================
SOURCES
====================
{context}

====================
QUESTION
====================
{question}
"""


    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
    "status": "ok",
    "engine": "rag",
    "text": r.choices[0].message.content.strip(),
    "sql": None,
    "sources": chunks,
    "provenance": {
        "type": "rag",
        "index": str(RAG_INDEX_PATH),
        "top_k": len(chunks),
        "rows": [
            {
                "row_id": c["row_id"],
                "source_page": c["page"],
                "score": round(c["score"], 4),
                "excerpt": c["text"][:200]
            }
            for c in chunks
        ]
    }
}


def resolve_entity_candidates(
    question: str,
    pool: Dict[str, str],
    max_candidates: int = 8
) -> List[str]:
    """
    Return plausible entity matches from DB only.
    Used ONLY to detect ambiguity.
    """
    q = norm(question)
    candidates = []

    # inclusion simple
    for k, v in pool.items():
        if k in q or q in k:
            candidates.append(v)

    # fallback fuzzy
    if not candidates:
        matches = get_close_matches(
            q,
            pool.keys(),
            n=max_candidates,
            cutoff=0.55
        )
        candidates = [pool[m] for m in matches]

    return list(dict.fromkeys(candidates))[:max_candidates]
def is_explicitly_scoped(question: str, entity_value: str) -> bool:
    """
    Returns True only if the entity is explicitly mentioned
    in the user question (not inferred).
    """
    return exact_norm_match(question, {norm(entity_value): entity_value}) is not None

def detect_ambiguity(question: str) -> Optional[Dict]:
    intent = analyze_intent(question)

    region = resolve_entity(question, REGIONS)
    circo  = resolve_entity(question, CIRCOS)

    # ============================
    # 1️⃣ WINNER / ELU
    # ============================
    if intent["winner"]:
        if not region and not circo:
            return {
                "type": "zone",
                "candidates": list(REGIONS.values())[:8]
            }

        # ⚠️ même si détecté → pas explicitement demandé
        if circo and not is_explicitly_scoped(question, circo):
            return {
                "type": "circonscription",
                "candidates": resolve_entity_candidates(question, CIRCOS)
            }

    # ============================
    # 2️⃣ ANALYTICS (top / taux)
    # ============================
    if intent["analytics"]:
        if not region:
            return {
                "type": "région",
                "candidates": list(REGIONS.values())[:8]
            }

        if region and not is_explicitly_scoped(question, region):
            return {
                "type": "région",
                "candidates": resolve_entity_candidates(question, REGIONS)
            }

    return None


def build_ambiguity_context(question: str) -> Dict:
    intent = analyze_intent(question)

    region = resolve_entity(question, REGIONS)
    circo  = resolve_entity(question, CIRCOS)
    party  = resolve_entity(question, PARTIS)

    return {
        "intent": intent,
        "resolved_entities": {
            "region": region,
            "circonscription": circo,
            "party": party
        },
        "question": question
    }
def needs_clarification(question: str) -> bool:
    intent = analyze_intent(question)

    region = resolve_entity(question, REGIONS)
    circo  = resolve_entity(question, CIRCOS)
    party  = resolve_entity(question, PARTIS)

    # Cas 1 : gagnant sans périmètre clair
    if intent["winner"] and not region and not circo:
        return True

    # Cas 2 : top sans périmètre
    if intent["top"] and not region:
        return True

    # Cas 3 : taux sans périmètre
    if intent["taux"] and not region:
        return True

    # Cas 4 : mot vague détecté (sans liste hardcodée)
    tokens = extract_tokens(question)
    if any(len(t) <= 4 for t in tokens) and not region and not circo:
        return True

    return False
def llm_clarify_entity(context: Dict) -> Dict:
    """
    LLM asks a NATURAL clarification question
    WITHOUT EVER asking election type, date or event.
    """

    prompt = f"""
Tu es un assistant conversationnel spécialisé dans
les résultats des élections CEI 2025.

CONTEXTE FIXE (NE JAMAIS QUESTIONNER) :
- même élection
- même scrutin
- mêmes résultats officiels

QUESTION UTILISATEUR :
"{context["question"]}"

INTENTION DÉTECTÉE :
{context["intent"]}

ENTITÉS DÉTECTÉES :
- Région : {context["resolved_entities"]["region"]}
- Circonscription : {context["resolved_entities"]["circonscription"]}
- Parti : {context["resolved_entities"]["party"]}

RÈGLES ABSOLUES :
- Ne demande JAMAIS :
  - le type d’élection
  - la date
  - l’événement
- Ne reformule pas la question de façon administrative
- Pose UNE seule question courte et naturelle
- Demande uniquement l’information manquante
- Parle comme un humain

FORMAT JSON STRICT :
{{
  "clarification_question": "...",
  "options": []
}}
"""

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.25,
        messages=[{"role": "user", "content": prompt}],
    )

    return json.loads(r.choices[0].message.content)
def llm_verbalize_sql_text(
    question: str,
    sql_text: str
) -> str:
    """
    Reformule un résultat SQL BRUT (texte) en réponse humaine.
    Aucun hardcode métier. Aucune supposition.
    """

    prompt = f"""
Tu es un assistant officiel de la CEI 2025.

Ta mission est de transformer un RÉSULTAT SQL BRUT
en une réponse humaine, claire et professionnelle.

QUESTION DE L’UTILISATEUR :
"{question}"

RÉSULTAT SQL BRUT :
\"\"\"
{sql_text}
\"\"\"

RÈGLES :
- Réponds en français
- 1 à 2 phrases maximum
- Ton clair, naturel et institutionnel
- Ne répète PAS les noms de colonnes
- Si le résultat contient un nombre → parle de quantité
- Si le résultat contient un taux → parle de participation
- Si le résultat est un classement → parle de top / leaders
- Ne mets PAS de tableau
- N’invente RIEN
- Ne pose PAS de question

RÉPONSE :
"""

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )

    return r.choices[0].message.content.strip()



def llm_is_followup_question(user_message: str, last_assistant_message: str | None):
    """
    Ask the LLM if the user message is a contextual follow-up.
    ALWAYS returns a safe dict.
    """

    system = """
You are a conversational router.

Determine if the user message refers to the previous assistant response.

If YES:
- is_followup = true
- infer the expected action

Possible actions:
- table
- chart
- text
- null

Respond ONLY with valid JSON.
No explanations. No markdown.
"""

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"""
User message:
"{user_message}"

Previous assistant message:
"{last_assistant_message or ''}"
"""
        }
    ]

    try:
        resp = llm_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            messages=messages,
        )

        raw = resp.choices[0].message.content.strip()

        # 🔥 sécurité : réponse vide
        if not raw:
            return {
                "is_followup": False,
                "action": None
            }

        # 🔥 nettoyage si ```json``` ou texte parasite
        raw = raw.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw)

        # 🔥 validation minimale
        return {
            "is_followup": bool(data.get("is_followup", False)),
            "action": data.get("action")
        }

    except Exception:
        # 🛟 FAIL-SAFE ABSOLU
        return {
            "is_followup": False,
            "action": None
        }
def ask_level2(question: str) -> Dict:
    """
    Routeur conversationnel principal.
    IDENTIQUE AU COMPORTEMENT ACTUEL
    + observability
    + sources SQL / RAG exposées
    """

    trace = lf.start_span(name="ask_level2")
    trace.input = question

    obs = {
        "route": None,
        "intent": None,
        "entities": {},
        "timings_ms": {},
        "sql": None,
        "sources": None,
        "token_usage": {},
    }

    try:
        with timed("intent", obs):
            obs["intent"] = analyze_intent(question)

        with timed("entity_resolution", obs):
            obs["entities"] = {
                "region": resolve_entity(question, REGIONS),
                "circonscription": resolve_entity(question, CIRCOS),
                "party": resolve_entity(question, PARTIS),
            }

        # ============================
        # 🧠 NIVEAU 1 — SQL / LOGIQUE
        # ============================
        with timed("ask_level1", obs):
            res = ask_level1(question)

        # 🚫 Sécurité
        if res["status"] == "refused":
            obs["route"] = "refused"
            res["_obs"] = obs
            trace.output = {"status": "refused"}
            return res

        # ============================
        # ✅ SQL → VERBALISATION
        # ============================
        if res["status"] == "ok" and res["engine"] == "sql":
            obs["route"] = "sql"
            obs["sql"] = res.get("sql")

            raw_sql_text = res.get("text", "")

            with timed("llm_verbalize_sql", obs):
                verbalized = llm_verbalize_sql_text(
                    question=question,
                    sql_text=raw_sql_text
                )

            # 🔹 sources SQL exposées comme RAG
            obs["sources"] = {
                "type": "sql",
                "table": "resultats",
                "query": res.get("sql"),
                "provenance": res.get("provenance"),
            }

            res["text"] = verbalized
            res["sources"] = obs["sources"]
            res["_obs"] = obs

            trace.output = {"status": "ok", "engine": "sql"}
            return res

        # ============================
        # ❌ HORS SCOPE (RAG INTERDIT)
        # ============================
        if res["status"] == "not_found" and res.get("rag_allowed") is False:
            obs["route"] = "out_of_scope"
            res["_obs"] = obs
            trace.output = {"status": "not_found"}
            return res

        # ============================
        # ❓ CLARIFICATION
        # ============================
        with timed("needs_clarification", obs):
            if needs_clarification(question):
                obs["route"] = "clarify"
                context = build_ambiguity_context(question)

                with timed("llm_clarify", obs):
                    clarif = llm_clarify_entity(context)

                res = {
                    "status": "clarify",
                    "engine": "llm",
                    "text": clarif["clarification_question"],
                    "options": clarif.get("options", []),
                    "sql": None,
                    "rag_allowed": False,
                    "_obs": obs,
                }
                trace.output = {"status": "clarify"}
                return res

        # ============================
        # 📚 RAG
        # ============================
        obs["route"] = "rag"

        with timed("rag_search", obs):
            chunks = rag_search(question)

        with timed("rag_answer", obs):
            rag_res = rag_answer(question, chunks)

        # 🔹 sources RAG déjà existantes → on les expose
        obs["sources"] = rag_res.get("provenance")

        rag_res["_obs"] = obs
        trace.output = {"status": rag_res.get("status"), "engine": "rag"}
        return rag_res

    finally:
        trace.end()
        lf.flush()


# alias final
ask = ask_level2

