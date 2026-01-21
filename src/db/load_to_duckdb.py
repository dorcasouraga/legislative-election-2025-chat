import duckdb
import pandas as pd
from pathlib import Path

PARQUET_PATH = Path("data/processed/results_clean.parquet")
DB_PATH = Path("src/db/results.duckdb")

print("📦 Loading parquet:", PARQUET_PATH)

df = pd.read_parquet(PARQUET_PATH)

DB_PATH.parent.mkdir(exist_ok=True, parents=True)
con = duckdb.connect(str(DB_PATH))

# Drop old tables
con.execute("DROP TABLE IF EXISTS resultats;")
con.execute("DROP TABLE IF EXISTS dim_circonscription;")
con.execute("DROP TABLE IF EXISTS dim_parti;")
con.execute("DROP TABLE IF EXISTS dim_candidat;")

# Load raw table
con.register("df", df)
con.execute("CREATE TABLE resultats AS SELECT * FROM df;")

# ========= DIM CIRC =========
con.execute("""
CREATE TABLE dim_circonscription AS
SELECT DISTINCT
    region,
    code_circonscription,
    circonscription,
    nb_bv,
    inscrits,
    votants,
    taux_participation,
    bulletins_nuls,
    suffrages_exprimes,
    bulletins_blancs
FROM resultats
ORDER BY code_circonscription;
""")

# ========= DIM PARTI =========
con.execute("""
CREATE TABLE dim_parti AS
SELECT
    parti,
    COUNT(*) AS total_lignes,
    SUM(voix) AS total_voix,
    SUM(CASE WHEN elu THEN 1 ELSE 0 END) AS elus
FROM resultats
GROUP BY parti
ORDER BY total_voix DESC;
""")

# ========= DIM CANDIDAT =========
con.execute("""
CREATE TABLE dim_candidat AS
SELECT
    candidat,
    parti,
    region,
    SUM(voix) AS total_voix,
    MAX(pourcentage_voix) AS meilleur_score_pct,
    MAX(CASE WHEN elu THEN 1 ELSE 0 END) AS elu
FROM resultats
GROUP BY candidat, parti, region
ORDER BY total_voix DESC;
""")

# ========= USEFUL VIEWS =========

# winners only
con.execute("""
CREATE OR REPLACE VIEW vw_elus AS
SELECT *
FROM resultats
WHERE elu = TRUE;
""")

# participation by region
con.execute("""
CREATE OR REPLACE VIEW vw_participation_region AS
SELECT
    region,
    AVG(taux_participation) AS taux_moyen
FROM dim_circonscription
GROUP BY region
ORDER BY taux_moyen DESC;
""")

# ranking per circonscription
con.execute("""
CREATE OR REPLACE VIEW vw_ranking AS
SELECT
    region,
    code_circonscription,
    circonscription,
    candidat,
    parti,
    voix,
    pourcentage_voix,
    ROW_NUMBER() OVER (
        PARTITION BY code_circonscription
        ORDER BY voix DESC
    ) AS rank_in_circ
FROM resultats;
""")

# party share
con.execute("""
CREATE OR REPLACE VIEW vw_parti_summary AS
SELECT *
FROM dim_parti;
""")

con.close()

print(f"DONE — Loaded {len(df)} rows + 3 dim tables + views into DuckDB!")
