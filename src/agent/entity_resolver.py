import duckdb
import unicodedata
import re
from rapidfuzz import process, fuzz
from functools import lru_cache
from db.duck import get_conn


DB_PATH = "src/db/results.duckdb"

def normalize_basic(x: str) -> str:
    if not x:
        return ""
    x = ''.join(
        c for c in unicodedata.normalize("NFD", x)
        if unicodedata.category(c) != "Mn"
    )
    x = x.lower()
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def load_entities():
    con = get_conn()
    regions = con.execute("SELECT DISTINCT region FROM resultats").fetchdf()['region']
    circs = con.execute("SELECT DISTINCT circonscription FROM resultats").fetchdf()['circonscription']
    parties = con.execute("SELECT DISTINCT parti FROM resultats").fetchdf()['parti']

    return {
        "regions": [normalize_basic(x) for x in regions if x],
        "circonscriptions": [normalize_basic(x) for x in circs if x],
        "parties": [normalize_basic(x) for x in parties if x],
    }


ENTITIES = load_entities()

def resolve_entity(user_text: str, category: str, threshold=80):
    normalized = normalize_basic(user_text)
    choices = ENTITIES[category]

    best, score, _ = process.extractOne(
        normalized,
        choices,
        scorer=fuzz.WRatio
    )

    if score >= threshold:
        return best
    return None

