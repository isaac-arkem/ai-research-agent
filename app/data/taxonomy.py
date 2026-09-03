# The topic taxonomy is the list of "niche" categories we organise creators into.
# When someone says "find fashion creators", the agent maps that to "fashion_beauty".
#
# Each slug has a list of aliases — words that should trigger that category.
# For example, if the operator types "cooking" or "recipe", the agent knows
# they mean the "cooking_mum" niche.
#
# In production this will come from Supabase. This hardcoded version is the
# fallback for local development and the validation script.

from typing import Dict, List


TAG_ALIASES: Dict[str, List[str]] = {
    "music_dance":           ["dance", "dancing", "dancer", "music", "song", "choreo", "choreography"],
    "cooking_mum":           ["cooking", "cook", "food", "recipe", "kitchen", "baking", "mum", "mom", "mother"],
    "comedy_skits":          ["comedy", "comedian", "funny", "humour", "humor", "skit", "skits", "sketch"],
    "yerevan_lifestyle":     ["yerevan", "armenia", "armenian"],
    "adhd_wellness":         ["adhd", "wellness", "neurodivergent", "mental health", "therapy", "selfcare"],
    "spirituality":          ["spiritual", "spirituality", "astrology", "tarot", "manifestation", "faith", "religion", "zodiac"],
    "heritage_diaspora":     ["heritage", "diaspora", "culture", "cultural", "immigrant", "roots", "tradition"],
    "trading":               ["trading", "trader", "crypto", "cryptocurrency", "bitcoin", "forex", "stocks", "investing", "finance"],
    "polymarket":            ["polymarket", "prediction market", "betting", "odds"],
    "buchona_lifestyle":     ["buchona", "buchona lifestyle"],
    "buchona_beauty":        ["buchona beauty"],
    "fashion_beauty":        ["fashion", "beauty", "makeup", "style", "outfit", "ootd", "skincare", "hair"],
    "ai_content":            ["ai", "artificial intelligence", "genai"],
    "maga":                  ["maga", "politics", "political"],
    "travel":                ["travel", "travelling", "holiday", "trip"],
    "beauty":                ["beauty", "makeup", "skincare"],
    "chinese_student":       ["chinese", "china"],
    "wealthy_russian_youth": ["russian", "russia"],
    "armenian_creators":     ["armenian", "armenia", "yerevan"],
}
