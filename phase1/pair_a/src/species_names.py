"""Offline English names for the full iWildCam v2 class vocabulary.

Lookup uses scientific names, never numeric IDs (which differ in previews).
See the repository README for name-data provenance and generic-label notes.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

NAME_TABLE = Path(__file__).resolve().parents[1] / "data/iwildcam_english_names.csv"


def normalize_species_name(name: str) -> str:
    text = " ".join(unicodedata.normalize("NFKC", str(name)).casefold().split())
    return re.sub(r"\bsp\.$", "sp", text)


# Keep genus-level and unidentified labels at their actual level of specificity.
# The two additional supported species are referenced in the repository README.
DISPLAY_NAMES = {
    "empty": "Empty / no animal",
    "motorcycle": "Motorcycle",
    "unknown bird": "Unidentified bird",
    "ave desconocida": "Unidentified bird",
    "unknown bat": "Unidentified bat",
    "unknown dove": "Unidentified dove",
    "bos taurus": "Cattle",
    "geotrygon sp": "Quail-dove (species unspecified)",
    "didelphis sp": "Opossum (species unspecified)",
    "phaetornis sp": "Hummingbird (species unspecified)",
    "brotogeris sp": "Parakeet (species unspecified)",
    "macaca sp": "Macaque (species unspecified)",
    "mazama sp": "Brocket deer (species unspecified)",
    "tragulus sp": "Mouse-deer (species unspecified)",
    "lophura sp": "Pheasant (species unspecified)",
    "peromyscus sp": "Deer mouse (species unspecified)",
    "sciurus sp": "Squirrel (species unspecified)",
    "cercopithecus ascanius": "Red-tailed monkey",
    "potamochoerus porcus": "Red river hog",
}


@lru_cache(maxsize=1)
def english_names() -> dict[str, str]:
    result = {}
    with NAME_TABLE.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            scientific = normalize_species_name(row["name"])
            common = row["english"].strip()
            if not scientific or not common:
                raise ValueError("Empty species name in English-name table")
            if scientific in result and result[scientific].casefold() != common.casefold():
                raise ValueError(f"Conflicting English names for {scientific}")
            result[scientific] = common[0].upper() + common[1:]
    result.update(DISPLAY_NAMES)
    return result


def english_name(scientific_name: str) -> str | None:
    """Return a known English name; unfamiliar future labels stay unguessed."""
    return english_names().get(normalize_species_name(scientific_name))


def format_species_name(scientific_name: str) -> str:
    """Both names for searchable selectors and species detail lists."""
    common = english_name(scientific_name)
    if not common or normalize_species_name(common) == normalize_species_name(scientific_name):
        return common or scientific_name
    return f"{common} ({scientific_name})"
