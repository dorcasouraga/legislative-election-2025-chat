import re
from pathlib import Path

import pdfplumber
import pandas as pd
import unicodedata


PDF_PATH = Path("data/raw/EDAN_2025_RESULTAT_NATIONAL_DETAILS.pdf")
OUT_BASENAME = Path("data/processed/results_clean")


# ========= REGION MAP ========= #
def region_for(code):
    c = int(code)

def region_for(code: str) -> str:
    c = int(code)

    if   1  <= c <= 8:    return "AGNEBY-TIASSA"
    elif 9  <= c <= 13:   return "BAFING"
    elif 14 <= c <= 18:   return "BAGOUE"
    elif 19 <= c <= 22:   return "BELIER"
    elif 23 <= c <= 27:   return "BERE"
    elif 28 <= c <= 35:   return "BOUNKANI"
    elif 36 <= c <= 38:   return "CAVALLY"
    elif 39 <= c <= 50:   return "DISTRICT AUTONOME D'ABIDJAN"
    elif 51 <= c <= 53:   return "DISTRICT AUTONOME DE YAMOUSSOUKRO"
    elif 54<= c <= 55 : return "FOLON"
    elif 56 <= c <= 63:   return "GBEKE"
    elif 64 <= c <= 66:   return "GBOKLE"
    elif 67 <= c <= 77:   return "GOH"
    
    elif 78 <= c <= 82:   return "GONTOUGO"
    elif 82 <= c <= 85:   return "GRANDS-PONTS"
    elif 86 <= c <= 92:   return "GUEMON"
    elif 93 <= c <= 98:   return "HAMBOL"
    elif 99 <= c <= 108:  return "HAUT-SASSANDRA"
    elif 109 <= c <= 113:  return "IFFOU"
    elif 114 <= c <= 118: return "INDENIE-DJUABLIN"
    elif 119 <= c <= 124: return "KABADOUGOU"
    elif 125 <= c <= 132: return "LOH-DJIBOUA"
    elif 133 <= c <= 140: return "MARAHOUE"
    elif 141 <= c <= 147: return "LA ME"
    elif 148 <= c <= 153: return "NAWA"
    elif 135 <= c <= 138: return "NAWA"
    elif c in [154,157 , 158, 161, 162]: return "MORONOU"
    elif c in [155,156 , 159, 160 ]: return "NZI"
    elif 163 <= c <= 172: return "PORO"
    elif 173 <= c <= 177: return "SAN-PEDRO"
    elif 178 <= c <= 184: return "SUD-COMOE"
    elif 185 <= c <= 189: return "TCHOLOGO"
    elif 190 <= c <= 199: return "TONKPI"
    elif 200 <= c <= 205: return "WORODOUGOU"
    else:                 return "UNKNOWN"



#========= CLEANING HELPERS ========= #
def normalize(x: str) -> str:
    """Remove accents, uppercase, keep A–Z0–9 + spaces."""
    if not x:
        return ""
    x = ''.join(
        c for c in unicodedata.normalize("NFD", x)
        if unicodedata.category(c) != "Mn"
    )
    x = x.upper()
    x = re.sub(r"[^A-Z0-9 ]+", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()




def to_int(x: str) -> int:
    if not x:
        return 0
    cleaned = re.sub(r"[^0-9]", "", x)
    return int(cleaned) if cleaned else 0


def to_float(x: str) -> float:
    if not x:
        return 0.0
    cleaned = x.replace(",", ".")
    cleaned = re.sub(r"[^0-9.]", "", cleaned)
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def looks_int(x: str) -> bool:
    return bool(re.sub(r"[^0-9]", "", x)) if x else False


# ========= MAIN PARSER ========= #
def parse_pdf(path: Path) -> pd.DataFrame:
    records = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            table = page.extract_table()
            if not table:
                continue

            current_code = None
            current_circ = None
            turnout = {}

            for row in table:
                row = [c.strip() if c else "" for c in row]

                # Skip TOTAL summary lines
                if row[0] == "TOTAL":
                    continue

                # FIRST candidate row
                if (
                    len(row) > 14
                    and looks_int(row[1])
                    and looks_int(row[3])
                    and looks_int(row[4])
                ):
                    current_code = row[1]
                    current_circ = normalize(row[2])

                    turnout = {
                        "nb_bv": to_int(row[3]),
                        "inscrits": to_int(row[4]),
                        "votants": to_int(row[5]),
                        "taux_participation": to_float(row[6]),
                        "bulletins_nuls": to_int(row[7]),
                        "suffrages_exprimes": to_int(row[8]),
                        "bulletins_blancs": to_int(row[9]),
                    }

                    records.append({
                        "region": region_for(current_code),
                        "code_circonscription": current_code,
                        "circonscription": current_circ,
                        **turnout,
                        "parti": normalize(row[11]),
                        "candidat": normalize(row[12]),
                        "voix": to_int(row[13]),
                        "pourcentage_voix": to_float(row[14]),
                        "elu": "ELU" in row[-1].upper(),
                        "page_source": page_num,
                    })
                    continue

                # SUBSEQUENT candidate rows
                if len(row) > 12 and row[11] and looks_int(row[13]):
                    records.append({
                        "region": region_for(current_code),
                        "code_circonscription": current_code,
                        "circonscription": current_circ,
                        **turnout,
                        "parti": normalize(row[11]),
                        "candidat": normalize(row[12]),
                        "voix": to_int(row[13]),
                        "pourcentage_voix": to_float(row[14]),
                        "elu": "ELU" in row[-1].upper(),
                        "page_source": page_num,
                    })

    return pd.DataFrame(records)


from rapidfuzz import fuzz

def cluster_names(series, threshold=90):
    """
    Auto-group similar names using fuzzy matching (Levenshtein ratio).
    Returns a mapping: raw_candidate -> canonical representative.
    No hardcoded name list.
    """
    mapping = {}
    representatives = []

    uniques = list(series.unique())

    for name in uniques:
        matched = False
        for rep in representatives:
            if fuzz.ratio(name, rep) >= threshold:
                mapping[name] = rep
                matched = True
                break
        if not matched:
            representatives.append(name)
            mapping[name] = name

    return mapping


# === MANUAL CANONICAL FIXES ===
REPLACE_MAP = {
    "UNE COTE D IVOIRE EN PAIX PROSPERE ET SOLIDAIRE": "UNE COTE DIVOIRE EN PAIX PROSPERE ET SOLIDAIRE",
    "UNE COTE DIVOIRE EN PAIX PROSPERE ET SOLIDARITE": "UNE COTE DIVOIRE EN PAIX PROSPERE ET SOLIDAIRE",
    "UNE COTE DIVOIRE EN PAIX PROPERE ET SOLIDAIRE": "UNE COTE DIVOIRE EN PAIX PROSPERE ET SOLIDAIRE",
    "UNE COTE IVOIRE EN PAIX PROSPERE ET SOLIDAIRE": "UNE COTE DIVOIRE EN PAIX PROSPERE ET SOLIDAIRE",
}



if __name__ == "__main__":
    df = parse_pdf(PDF_PATH)

    # 1) core normalization
    for col in ["region", "circonscription", "parti", "candidat"]:
        df[col] = df[col].fillna("").apply(normalize)

    # 2) drop empty candidate rows
    df["candidat"] = df["candidat"].replace(REPLACE_MAP)
    df = df[df["candidat"] != ""].copy()



    # 3) build fuzzy mapping on normalized candidates
    mapping = cluster_names(df["candidat"])

    # 4) apply cluster
    df["candidat_canon"] = df["candidat"].map(mapping)

    OUT_BASENAME.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUT_BASENAME.with_suffix(".parquet"), index=False)
    df.to_csv(OUT_BASENAME.with_suffix(".csv"), index=False, encoding="utf-8")
    df.to_excel(OUT_BASENAME.with_suffix(".xlsx"), index=False)

    print("SAVED:", df.shape)
    print(df.head())

