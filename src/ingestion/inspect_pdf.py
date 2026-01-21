from pathlib import Path
import pdfplumber

PDF_PATH = Path("data/raw/EDAN_2025_RESULTAT_NATIONAL_DETAILS.pdf")

def inspect_first_page():
    with pdfplumber.open(PDF_PATH) as pdf:
        print("Pages:", len(pdf.pages))
        page0 = pdf.pages[0]
        table = page0.extract_table()
        print(table[:3])

if __name__ == "__main__":
    inspect_first_page()
