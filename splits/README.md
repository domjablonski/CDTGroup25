# splits/: shared split for cuisine prediction

`cuisine_split.csv` says, for every labelled recipe, its cuisine and whether it is in `train`, `val` or
`test`. The graph model (graph/) and the XGBoost model (minhash/ or R5's folder) **both use this file**,
so their scores are computed on exactly the same recipes.

| Column | Meaning |
|---|---|
| `id` | recipe id (column `id` of `recipes.parquet`) |
| `cuisine` | label: first of the 16 cuisine tags found in the recipe tags (most specific first) |
| `split` | `train` (64 %), `val` (16 %) or `test` (20 %) |

40,912 recipes, 11 cuisines (those with at least 1,500 labelled recipes): italian, southern-united-states,
mexican, asian, indian, african, greek, french, chinese, middle-eastern, caribbean. Stratified by cuisine;
recipes with exactly the same ingredient set are always in the same part. Seed 42.

## How to use it

```python
import pandas as pd
split = pd.read_csv("splits/cuisine_split.csv")
recipes = pd.read_parquet("data/raw/processed/recipes.parquet", columns=["id", "ingredients_v2"])
data = split.merge(recipes, on="id")
train, val, test = (data[data["split"] == s] for s in ("train", "val", "test"))
```

1. Choose hyperparameters with `train` -> `val` only.
2. Refit the final model on `train` + `val`.
3. Evaluate **once** on `test`: accuracy, **macro-F1** (each cuisine counts the same; the classes are
   imbalanced, italian 7,305 vs caribbean 1,570) and the confusion matrix, with the classes in
   alphabetical order.

Majority class on test: accuracy 0.18, macro-F1 0.03. Logistic regression on all ingredients (one-hot,
class_weight="balanced"): accuracy 0.68, macro-F1 0.62 (graph/results/cuisine_test.csv).

## Regenerate

```bash
python splits/make_cuisine_split.py   # needs data*/**/processed/recipes.parquet, about 30 s
pytest splits -q                      # 6 tests
```
The file is deterministic: regenerating it from the same `recipes.parquet` gives the same split.
