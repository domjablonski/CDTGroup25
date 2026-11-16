# Ingredient pairing and recipe similarity on Food.com

02807 Computational Tools for Data Science – group project (5 members), DTU, autumn 2026.
Deadline: **4 December 2026, 23:59** on DTU Learn (we aim for 3 December).

## How we work (read this first)

| File | What it is |
|---|---|
| [docs/FoodCom_Project_Plan_EN.pdf](docs/FoodCom_Project_Plan_EN.pdf) / [FR](docs/FoodCom_Plan_Projet_FR.pdf) | The plan: calendar, roles, sprint backlog, verification, prompts ready to paste, risks, checklists |
| [docs/CONTRACT.md](docs/CONTRACT.md) | The shared spec: frozen files, schemas, behaviour of every module, tests, results files, ownership |
| [docs/PROJECT_INSTRUCTIONS_FOR_CLAUDE_CODE.txt](docs/PROJECT_INSTRUCTIONS_FOR_CLAUDE_CODE.txt) | Every story step by step, for Claude Code |
| [docs/backlog.md](docs/backlog.md) | One section per story, to paste as GitHub issues |

One story = one fresh Claude Code session = one branch = one pull request. In the repo root:

```
claude
Do story R2-S1. Instructions are in docs/.
```

## Research questions

**Main question:** what structure do the ingredient combinations of 231,637 recipes reveal, and can it be exploited at scale?

| # | Question | Method | Owner |
|---|---|---|---|
| RQ1 | Which ingredients go together beyond chance (rules, lift), and how much memory and time does PCY save over A-Priori? | PCY + A-Priori (own implementation) | R3 |
| RQ2 | How accurately and how fast can MinHash + LSH retrieve similar recipes ("recipes like this one"), compared with exact Jaccard? | MinHash + LSH (own implementation) | R2 |
| RQ3 | Does a PMI-weighted ingredient graph reveal cuisines better than raw co-occurrence? | Louvain / spectral (course) + PMI weighting (outside the course) | R4 |
| RQ4 | Can a recipe's calorie level be predicted from its LSH neighbours, and how close is it to exact kNN and a supervised baseline? | kNN on LSH neighbours (outside the course) | R5 |

Pipeline: R1 normalises ingredients → R3 mines pairs (PAIRS) → R4 weights them by PMI and finds communities; R2 indexes recipes with LSH → neighbours (NEIGHBOURS) → R5 predicts calorie levels; R2's near-duplicates (DUPLICATES) are removed from R3's baskets and kept on one side of R5's train/test split.

## Data

[Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
(Majumder, Li, Ni, McAuley, EMNLP 2019). We use `RAW_recipes.csv` (231,637 recipes) and `RAW_interactions.csv` (1,132,367 reviews).

**Licence: "Data files © Original Authors". Never commit the data.** `data/` is git-ignored; everyone downloads it with the script below.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
```

Download the data (needs a Kaggle API token in `~/.kaggle/kaggle.json`, see https://www.kaggle.com/docs/api):

```bash
python -m foodcom.download
```

No token? Download the two `RAW_*.csv` files manually from the Kaggle page into `data/raw/`.

Clean the data and build the 1% development sample:

```bash
python -m foodcom.cleaning
```

This writes `data/processed/recipes.parquet` (with `ingredients_norm`, `cuisine` and `calorie_level`), `interactions.parquet`, `recipes_sample.parquet` and `interactions_sample.parquet`.

Run the tests:

```bash
pytest
```

## Repository layout

```
data/raw/              RAW_*.csv from Kaggle (git-ignored)
data/processed/        Parquet outputs (git-ignored)
src/foodcom/
  config.py            paths, seed, thresholds, cuisine tags
  download.py          Kaggle download                         (R1)
  cleaning.py          parsing, cleaning, normalisation, sample (R1)
  schemas.py           columns of every exchanged file (frozen)
  lsh.py               MinHash + LSH, similar recipes, neighbours, duplicates (R2)
  itemsets.py          A-Priori, PCY, association rules, pair table           (R3)
  graphs.py            PMI weights, ingredient graph, Louvain, spectral       (R4)
  classify.py          calorie-level kNN from LSH neighbours, baselines       (R5)
notebooks/             one notebook per role + final_notebook.ipynb (appendix)
report/                outline, character budget, contributions table
tests/                 pytest
```

## Roles

| Role | Owner | Reviewer |
|---|---|---|
| R1 Data, ingredients & reproducibility | _name_ | R5 |
| R2 MinHash + LSH (RQ2) | _name_ | R1 |
| R3 PCY + A-Priori (RQ1) | _name_ | R2 |
| R4 PMI graph + Louvain (RQ3) | _name_ | R3 |
| R5 Calorie classifier (RQ4) & report editor | _name_ | R4 |

## Conventions

- Develop on the 1% sample (`*_sample.parquet`), run on the full data only when the code works.
- Course algorithms (MinHash/LSH, A-Priori/PCY) are **our own implementation**. Libraries (datasketch, mlxtend) are only used to check our results.
- Every random step uses `config.SEED`.
- Functions in `src/foodcom/` must follow the signatures and docstrings already written there: they are the contract between roles. Change them only after telling the team.
- Notebooks import from `src/`; no copy-pasted code between notebooks.
- One branch per role, merge into `main` through a pull request reviewed by your reviewer.
- Cite the data source in every figure caption that uses it: "Food.com, Majumder et al. (2019)".

## Existing preprocessing script

The repository already had a first cleaning script, kept as it is:

```bash
python preprocessing/preprocessing.py
```

It looks for `RAW_recipes.csv` and `RAW_interactions.csv` in a `data*` folder under the project root and writes
`recipes.parquet` and `interactions.parquet` to a `processed/` folder next to them (it needs `pyarrow`, now in
`requirements.txt`). Unlike `python -m foodcom.cleaning`, it drops outliers instead of flagging them and does not add
`ingredients_norm`, `cuisine` or `calorie_level`. Which of the two the team keeps is R1's call (story R1-S1b).
