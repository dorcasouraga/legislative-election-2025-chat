import unicodedata
import re
from dataclasses import dataclass
from typing import Optional


def normalize_question_text(text: str) -> str:
    """
    Normalise un texte pour matching et recherche.
    """
    if not text:
        return ""

    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class LocationEntity:
    raw_text: str
    normalized: str
    circonscription: Optional[str] = None
