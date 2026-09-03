# Operators don't type country codes — they type "Dubai", "KSA", or "the Gulf".
# These lookup tables let the agent translate natural language into the codes
# our scraper actually understands. Codes match the DB markets table (e.g.
# "UAE" not "AE").

from typing import Dict, List


COUNTRY_ALIASES: Dict[str, List[str]] = {
    "UAE": ["uae", "united arab emirates", "emirates", "dubai", "abu dhabi"],
    "TR": ["turkey", "turkiye", "türkiye"],
    "SA": ["saudi", "saudi arabia", "ksa"],
    "BR": ["brazil", "brasil"],
    "MX": ["mexico", "méxico"],
    "ZA": ["south africa"],
    "AM": ["armenia"],
}


REGION_MAPPINGS: Dict[str, List[str]] = {
    "Gulf / GCC":          ["UAE", "SA", "KW"],
    "MENA":                ["UAE", "SA", "KW", "EG", "MA"],
    "North Africa":        ["EG", "MA"],
    "West Africa":         ["NG"],
    "East/Southern Africa": ["NG", "ZA"],
    "Latin America":       ["BR", "MX", "CO", "AR"],
    "Southeast Asia":      ["ID", "PH", "TH", "MY"],
    "South Asia":          ["IN"],
    "East Asia":           ["JP"],
}
