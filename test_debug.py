# test_names.py
from src.extractors.names import NameExtractor
from pathlib import Path

ROOT = Path(__file__).parent
ext = NameExtractor(
    institution_keywords_path=str(ROOT / "data" / "institution_keywords.json"),
    geo_json_path=str(ROOT / "data" / "geo_canonical.json"),
    snomed_known_terms=frozenset(),
)

tests = [
    "lung cancer research on new york by dr holtz in phase 2 or 3",
    "phase ii studies by dr holmes",
    "dr holmes",
    "studies by dr holmes",
    "cancer studies by dr holmes",
    "cancer by dr holmes",
]

for q in tests:
    r = ext.extract(q)
    print(f"\nQuery: {q!r}")
    print(f"  investigator: {r.investigator_name!r} ({r.investigator_confidence})")
    print(f"  site:         {r.site_name!r} ({r.site_confidence})")
    print(f"  ambiguous:    {[a.text for a in r.ambiguous_names]}")