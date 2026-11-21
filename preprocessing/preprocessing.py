import ast
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 1024

IRREGULAR_INGREDIENT_PLURALS = {
    "leaves": "leaf",
    "loaves": "loaf",
    "knives": "knife",
    "tomatoes": "tomato",
    "potatoes": "potato",
}
SINGULAR_INGREDIENT_EXCEPTIONS = {
    "asparagus",
    "brownies",
    "cookies",
    "couscous",
    "fries",
    "glass",
    "grass",
    "hummus",
    "molasses",
    "pies",
    "series",
    "smoothies",
    "species",
    "zucchinis",
}


def singularize_ingredient_word(word):
    if word in IRREGULAR_INGREDIENT_PLURALS:
        return IRREGULAR_INGREDIENT_PLURALS[word]
    if word in SINGULAR_INGREDIENT_EXCEPTIONS or len(word) <= 3:
        return word
    if word.endswith(("ches", "shes", "xes", "zes", "oes")):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize_ingredient(ingredient):
    normalized = re.sub(
        r"[a-z]+",
        lambda match: singularize_ingredient_word(match.group()),
        ingredient.lower().strip(),
    )
    return re.sub(r"\s+", " ", normalized)


def normalize_ingredients(ingredients):
    normalized = []
    for ingredient in ingredients:
        item = normalize_ingredient(ingredient)
        if item in {"salt and pepper", "salt and pepper to taste"}:
            normalized.extend(("salt", "pepper"))
        else:
            normalized.append(item)
    return list(dict.fromkeys(normalized))


def write_ingredient_embeddings(recipes, output_path):
    ingredients = list(dict.fromkeys(
        ingredient
        for recipe_ingredients in recipes["ingredients_v2"]
        for ingredient in recipe_ingredients
    ))
    if not ingredients:
        pq.write_table(
            pa.table({
                "ingredient": pa.array([], type=pa.string()),
                "embedding": pa.array([], type=pa.list_(pa.float32())),
            }),
            output_path,
        )
        return

    model = SentenceTransformer(EMBEDDING_MODEL)
    writer = None
    try:
        for start in range(0, len(ingredients), EMBEDDING_BATCH_SIZE):
            batch = ingredients[start:start + EMBEDDING_BATCH_SIZE]
            vectors = model.encode(
                batch,
                batch_size=128,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            table = pa.table({
                "ingredient": batch,
                "embedding": pa.array(vectors.tolist(), type=pa.list_(pa.float32())),
            })
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            print(f"embedded {min(start + len(batch), len(ingredients))} of {len(ingredients)} unique ingredients", flush=True)
    finally:
        if writer is not None:
            writer.close()


# The data can be inside the repo (data/ next to preprocessing/) or next to the repo folder:
# look inside the repo first, then one level up, and use the first place where it is found.
repo_root = Path(__file__).resolve().parents[1]
search_roots = [repo_root, repo_root.parent]
dataset_matches = []
for project_root in search_roots:
    dataset_matches = [
        recipes_path
        for recipes_path in project_root.glob("data*/**/RAW_recipes.csv")
        if (recipes_path.parent / "RAW_interactions.csv").is_file()
    ]
    if dataset_matches:
        break

if len(dataset_matches) != 1:
    raise FileNotFoundError(
        "Expected exactly one dataset containing RAW_recipes.csv and "
        f"RAW_interactions.csv under {project_root} (searched {', '.join(map(str, search_roots))}); "
        f"found {len(dataset_matches)}."
    )
print(f"dataset: {dataset_matches[0].parent}")

dataset_dir = dataset_matches[0].parent
out_dir = dataset_dir / "processed"
out_dir.mkdir(exist_ok=True)

recipes = pd.read_csv(dataset_dir / "RAW_recipes.csv")
inter = pd.read_csv(dataset_dir / "RAW_interactions.csv", parse_dates=["date"])

print(recipes.shape, inter.shape)

# 1. Missing values
recipes = recipes.dropna(subset=["name"])
recipes["description"] = recipes["description"].fillna("")
inter["review"] = inter["review"].fillna("")

# 2. Duplicates
recipes = recipes.drop_duplicates(subset="id")
inter = inter.drop_duplicates(subset=["user_id", "recipe_id"])

# 3. Parse string-lists into real lists
for col in ["tags", "ingredients", "steps", "nutrition"]:
    recipes[col] = recipes[col].apply(ast.literal_eval)

# Keep the source ingredients while providing normalized, unique transactions
# for ingredient-based analyses. Split the common combined seasoning phrase
# so it matches recipes that list salt and pepper separately.
recipes["ingredients_v2"] = recipes["ingredients"].apply(normalize_ingredients)

# 4. Split nutrition into columns
nutr_cols = ["calories", "total_fat", "sugar", "sodium",
             "protein", "sat_fat", "carbs"]
recipes[nutr_cols] = pd.DataFrame(recipes["nutrition"].tolist(), index=recipes.index)
recipes = recipes.drop(columns="nutrition")

# 5. Inspect before filtering outliers and zero ratings
print(recipes[["minutes", "n_steps", "n_ingredients", "calories"]].describe())
print(inter["rating"].value_counts().sort_index())

recipes = recipes[recipes["minutes"].between(1, 1440)]
inter = inter[inter["rating"] > 0]

# 6. Keep only interactions whose recipe survived filtering
inter = inter[inter["recipe_id"].isin(recipes["id"])]

print(recipes.shape, inter.shape)

# 7. Save
recipes.to_parquet(out_dir / "recipes.parquet")
inter.to_parquet(out_dir / "interactions.parquet")
write_ingredient_embeddings(recipes, out_dir / "ingredient_embeddings.parquet")