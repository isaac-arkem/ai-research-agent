# These are the markets where our scraper can discover creators.
# Codes match what the DB markets table stores — e.g. "UAE" not "AE".
# This list is the FALLBACK when Supabase is unreachable.

from typing import List, NamedTuple


class Market(NamedTuple):
    code: str
    iso: str
    name: str


CREATOR_MARKETS: List[Market] = [
    Market("UAE", "UAE", "United Arab Emirates"),
    Market("SA",  "SA",  "Saudi Arabia"),
    Market("KW",  "KW",  "Kuwait"),
    Market("EG",  "EG",  "Egypt"),
    Market("TR",  "TR",  "Turkey"),
    Market("NG",  "NG",  "Nigeria"),
    Market("ZA",  "ZA",  "South Africa"),
    Market("MA",  "MA",  "Morocco"),
    Market("BR",  "BR",  "Brazil"),
    Market("MX",  "MX",  "Mexico"),
    Market("CO",  "CO",  "Colombia"),
    Market("AR",  "AR",  "Argentina"),
    Market("IN",  "IN",  "India"),
    Market("ID",  "ID",  "Indonesia"),
    Market("PH",  "PH",  "Philippines"),
    Market("TH",  "TH",  "Thailand"),
    Market("MY",  "MY",  "Malaysia"),
    Market("JP",  "JP",  "Japan"),
    Market("AM",  "AM",  "Armenia"),
]

VALID_ISO_CODES: set = {m.iso for m in CREATOR_MARKETS}
