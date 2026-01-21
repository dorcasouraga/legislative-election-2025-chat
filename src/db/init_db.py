from pathlib import Path
import duckdb

DB_PATH = Path("src/db/results.duckdb")
DATA_PATH = Path("data/processed/results_clean.parquet")

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB_PATH.as_posix())

    # Drop and recreate table safely
    con.execute("DROP TABLE IF EXISTS resultats;")
    con.execute("""
        CREATE TABLE resultats AS
        SELECT * FROM read_parquet(?)
    """, [DATA_PATH.as_posix()])

    # Vue des élus
    con.execute("""
        CREATE OR REPLACE VIEW vw_elus AS
        SELECT *
        FROM resultats
        WHERE elu = TRUE
    """)

    # Vue taux de participation par circonscription
    con.execute("""
        CREATE OR REPLACE VIEW vw_participation AS
        SELECT DISTINCT
            region,
            code_circonscription,
            circonscription,
            nb_bv,
            inscrits,
            votants,
            suffrages_exprimes,
            bulletins_nuls,
            bulletins_blancs,
            taux_participation
        FROM resultats
    """)

    con.close()
    print(f"🎉 Base créée avec succès → {DB_PATH}")

if __name__ == "__main__":
    init_db()
