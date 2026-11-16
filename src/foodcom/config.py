"""Paths, seed and shared constants (FROZEN: docs/CONTRACT.md 3.2).

Import from here instead of hard-coding values. Changing this file needs a `contract` PR.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"

KAGGLE_DATASET = "shuyangli94/food-com-recipes-and-user-interactions"
RAW_RECIPES = DATA_RAW / "RAW_recipes.csv"
RAW_INTERACTIONS = DATA_RAW / "RAW_interactions.csv"

RECIPES = DATA_PROCESSED / "recipes.parquet"
INTERACTIONS = DATA_PROCESSED / "interactions.parquet"
RECIPES_SAMPLE = DATA_PROCESSED / "recipes_sample.parquet"
INTERACTIONS_SAMPLE = DATA_PROCESSED / "interactions_sample.parquet"

# Files exchanged between roles (schemas in schemas.py)
DUPLICATES = DATA_PROCESSED / "duplicates.parquet"       # R2 -> R3, R5
NEIGHBOURS = DATA_PROCESSED / "neighbours.parquet"       # R2 -> R5
PAIRS = DATA_PROCESSED / "pairs.parquet"                 # R3 -> R4
RULES = DATA_PROCESSED / "rules.parquet"                 # R3 (results, no consumer)

SEED = 42
SAMPLE_FRAC = 0.01

# Order of the 7 values in the raw "nutrition" column (percent daily value except calories)
NUTRITION_COLS = ["calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
                  "protein_pdv", "saturated_fat_pdv", "carbohydrates_pdv"]

# Values above these limits are flagged as suspicious (not dropped)
MINUTES_MAX = 24 * 60          # longer than one day
CALORIES_MAX = 5000
INT32_MAX = 2_147_483_647      # overflow value found in the raw "minutes" column

# Calorie level (label of RQ4): calorie tertiles of the recipes without flag_calories,
# 0 = low, 1 = medium, 2 = high; flagged recipes get no level.
CALORIE_LEVELS = 3

# Near-duplicate threshold: pairs with Jaccard >= this are copies (removed from baskets,
# never split between the train and test sets of RQ4)
DUPLICATE_JACCARD = 0.9

# Cuisine tags used as ground truth, most specific first: a recipe gets the first one it has
# ("chinese" before "asian"). Counts in RAW_recipes.csv (03/11/2026): asian 13,485, italian 7,410,
# southern-united-states 7,114, mexican 6,694, african 2,851, indian 2,708, greek 2,391, french 2,268,
# middle-eastern 2,067, chinese 2,008, caribbean 1,709, german 1,377, thai 1,208, spanish 1,072,
# japanese 851, korean 253.
CUISINE_TAGS = ["chinese", "thai", "japanese", "korean", "indian", "italian", "mexican", "greek",
                "french", "spanish", "german", "middle-eastern", "caribbean", "southern-united-states",
                "african", "asian"]
