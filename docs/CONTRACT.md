# Food.com project - Shared contract (v2.0, 10 Nov 2026)

02807 Computational Tools for Data Science, DTU, autumn 2026 - group of 5.
Topic (decided 10 Nov): **"Ingredient pairing and recipe similarity on Food.com"**.

This file is the only thing the five roles share. Everyone codes against it, so nobody waits
for anybody. Sections 3.2, 3.3 and 3.5 (frozen files) change only through a pull request
labelled `contract`, approved by all five members; Claude is told to stop rather than change them.

Team (fill in): R1 _name_ (data, ingredients, reproducibility, coordination) - R2 _name_ (MinHash + LSH) -
R3 _name_ (PCY + A-Priori) - R4 _name_ (PMI ingredient graph + Louvain) - R5 _name_ (calorie classifier, report editor).

**Main question.** What structure do the ingredient combinations of 231,637 recipes reveal, and can it
be exploited at scale?

| RQ | Question | Method | Owner |
|---|---|---|---|
| RQ1 | Which ingredients go together beyond chance (rules, lift), and how much memory and time does PCY save over A-Priori? | PCY + A-Priori, our own code | R3 |
| RQ2 | How accurately and how fast can MinHash + LSH retrieve similar recipes ("recipes like this one"), compared with exact Jaccard? | MinHash + LSH, our own code | R2 |
| RQ3 | Does a PMI-weighted ingredient graph reveal cuisines better than raw co-occurrence? | Louvain / spectral (course) + PMI weighting (outside) | R4 |
| RQ4 | Can a recipe's calorie level be predicted from its LSH neighbours, and how close is it to exact kNN and a supervised baseline? | kNN on LSH neighbours (outside) | R5 |

Course topics: similar items / LSH, frequent itemsets (PCY), graph clustering (Louvain). Outside the
course: PMI/NPMI weighting (optional Node2Vec) and k-nearest-neighbour classification.

---

## 3.1 Stack and repository layout

Python 3.11 (3.10+ works), pandas >= 2, pyarrow, NumPy, Numba (optional, for hot loops),
scikit-learn, networkx, matplotlib, scipy, pytest. Reference libraries used **only to check our own
code**: datasketch (MinHash/LSH), mlxtend (A-Priori). Package `foodcom` in `src/` (src layout,
`pip install -e .`). Data: Kaggle "Food.com Recipes and Interactions" (Majumder et al., EMNLP 2019),
licence "Data files (c) Original Authors": **the data never goes into git**.

```
foodcom-project/
  CLAUDE.md, .claude/settings.json, .github/, README.md, pyproject.toml, requirements*.txt  R1
  data/raw/, data/processed/    git-ignored                                                 R1
  docs/CONTRACT.md              this file                                                   FROZEN (all)
  docs/PROJECT_INSTRUCTIONS_FOR_CLAUDE_CODE.txt, docs/backlog.md                            R1
  docs/AI_USAGE.md              everyone appends
  src/foodcom/config.py         paths, seed, constants                                      FROZEN (all)
  src/foodcom/schemas.py        columns of every exchanged file                             FROZEN (all)
  src/foodcom/download.py, cleaning.py                                                      R1
  src/foodcom/lsh.py                                                                        R2
  src/foodcom/itemsets.py                                                                   R3
  src/foodcom/graphs.py                                                                     R4
  src/foodcom/classify.py                                                                   R5
  tests/conftest.py, tests/fixtures/, tests/test_contract.py                                FROZEN (all)
  tests/test_cleaning.py | test_lsh.py | test_itemsets.py | test_graphs.py | test_classify.py  R1 | R2 | R3 | R4 | R5
  scripts/run_all.py                                                                        R1
  scripts/fig_*.py              one figure script per figure                                producer of the figure
  notebooks/0k_<topic>_Rk.ipynb (one per role), notebooks/final_notebook.ipynb              Rk / R1 assembles
  results/*.csv                 every number of the report comes from here                 producer of the file
  report/                       outline, figures/, contributions, the report itself        R5 (each writes a section)
```

Rule: a story only edits the files of its role. Shared files (README, final notebook) are edited by
their owner; others send the text in their PR description or a comment.

## 3.2 Frozen file 1: `src/foodcom/config.py`

Paths `ROOT, DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES`; raw and processed files (`RAW_RECIPES,
RAW_INTERACTIONS, RECIPES, INTERACTIONS, RECIPES_SAMPLE, INTERACTIONS_SAMPLE`); exchanged files
`DUPLICATES (R2 -> R3, R5)`, `NEIGHBOURS (R2 -> R5)`, `PAIRS (R3 -> R4)`, `RULES (R3)`;
`SEED = 42`, `SAMPLE_FRAC = 0.01`; `NUTRITION_COLS` (calories + 6 values in % daily value per serving);
`MINUTES_MAX = 1440`, `CALORIES_MAX = 5000`, `INT32_MAX = 2147483647`; `CALORIE_LEVELS = 3`;
`DUPLICATE_JACCARD = 0.9`; `CUISINE_TAGS` (16 tags, most specific first).

Never hard-code a path, a seed or a column list: import it from `config` / `schemas`.

## 3.3 Frozen file 2: `src/foodcom/schemas.py`

A producer writes **at least** these columns (extra columns allowed); a consumer relies on these only.
`schemas.validate(df, name)` raises if one is missing; every producer calls it before saving.

| File (config name) | Producer | Consumers | Due | Columns |
|---|---|---|---|---|
| `RECIPES` recipes.parquet | R1 | all | v2 Thu 12 Nov | recipe_id, name, minutes, contributor_id, submitted, tags, n_steps, steps, description, ingredients, n_ingredients, **ingredients_norm**, **cuisine**, **calorie_level**, 7 NUTRITION_COLS, flag_minutes, flag_calories, flag_dup_name |
| `INTERACTIONS` interactions.parquet | R1 | (descriptive only) | Thu 12 Nov | user_id, recipe_id, date, rating, review, rated, flag_unknown_recipe, flag_duplicate |
| `DUPLICATES` duplicates.parquet | R2 | R3, R5 | Thu 19 Nov | recipe_id, canonical_id, similarity |
| `NEIGHBOURS` neighbours.parquet | R2 | R5 | Thu 19 Nov | recipe_id, neighbour_id, similarity, rank |
| `PAIRS` pairs.parquet | R3 | R4 | v1 Thu 19, v2 (no duplicates) Sun 22 Nov | item_a, item_b, pair_count, count_a, count_b, n_baskets |
| `RULES` rules.parquet | R3 | - | Sun 22 Nov | antecedent, consequent, support, confidence, lift, group |

Types: ids int64; lists (tags, steps, ingredients, ingredients_norm, antecedent, consequent) are lists
of str (`cleaning.load_recipes()` turns Parquet arrays back into lists); calorie_level Int8 (0, 1, 2 or
NA); similarity float in [0, 1]; rank int (1 = most similar); item_a < item_b.

The function signatures of `src/foodcom/*.py` are part of the contract too: `test_contract.py` checks
every parameter name. The body is the owner's; the signature and the docstring are frozen.

## 3.4 Behaviour

### 3.4.1 Data, ingredients and labels (R1, `cleaning.py`)
- `clean_recipes` (done, run on all 231,637 recipes in about 1 min): parses the list columns, splits
  `nutrition`, renames `id` -> `recipe_id`. The overflow duration becomes NaN. **Flags, never drops**:
  `flag_minutes` (NaN, 0 or > 1,440 min: 3,094 recipes), `flag_calories` (> 5,000 kcal: 1,049),
  `flag_dup_name` (4,410). `cuisine` = first tag of `CUISINE_TAGS` (46,028 recipes).
- `calorie_level` (done): calorie tertiles of the recipes without `flag_calories`, 0 low / 1 medium /
  2 high; flagged recipes get NA. Computed on the full data before the sample is drawn. Measured
  (10 Nov): low <= 217.7 kcal (76,905 recipes), medium 217.8-431.7 (76,827), high >= 431.8 (76,856),
  no level 1,049.
- `ingredients_norm`: sorted unique `normalize_ingredient(x)`. The skeleton rule only lowercases and
  maps 10 synonyms (14,942 -> 14,933 labels). **R1-S1b extends it by Thu 12 Nov** (plurals with an
  exception list, frequent synonyms from the top 500, adjectives such as "fresh", "large", "chopped"
  when they do not change the ingredient). Every rule gets a test; before/after counts go to
  `results/data_summary.csv`. PCY, LSH and the graph all depend on this vocabulary.
- `make_sample`: 1 % of recipes (seed 42) and their reviews; `load_*(sample=True)` reads it.

### 3.4.2 MinHash + LSH (R2, `lsh.py`) - our own implementation
- Tokens hashed with `stable_hash` (CRC32), never Python's `hash()`.
- `shingles`: mode "ingredients" (main) = hashed `ingredients_norm`; mode "steps" (optional) =
  hashed 5-word shingles of the steps text.
- `minhash_signatures`: n_hashes universal hashes h_i(x) = (a_i x + b_i) mod p, p = 2^31 - 1, a_i in
  [1, p), b_i in [0, p) from `np.random.default_rng(seed)`; minimum over the set; empty set -> p.
  Vectorised NumPy (or Numba).
- `lsh_candidate_pairs` / `build_index`: bands x rows = n_hashes; one dict per band (band key ->
  positions). (b, r) is chosen with `s_curve` and with measured precision/recall (below).
- `similar_recipes(index, ingredients, k)`: candidates from the buckets of the query, ranked by exact
  Jaccard. This is the "recipes like this one" search; its latency is measured against brute force.
- `neighbours_table(index, k=20)`: top-20 neighbours of every recipe -> `NEIGHBOURS`.
- `find_duplicates(index, 0.9)`: pairs with Jaccard >= `DUPLICATE_JACCARD`, union-find groups,
  canonical = oldest recipe -> `DUPLICATES` (3,988 recipes share an exact ingredient set with another).
- Evaluation: exact all-pairs Jaccard on 5,000 random recipes (12.5 M pairs, sparse matrices) is the
  ground truth; for each (b, r) and thresholds {0.5, 0.7, 0.8, 0.9}: candidates, precision, recall,
  seconds; datasketch comparison on the 1 % sample; query latency (ms/query) LSH vs brute force.

### 3.4.3 PCY + A-Priori (R3, `itemsets.py`) - our own implementation
- Baskets = `ingredients_norm` of recipes not in `DUPLICATES` (PAIRS v1 uses all recipes, v2 not).
- Frequent: count >= ceil(min_support x N). Measured on the raw labels (3 Nov): support 1 % -> 146
  frequent items, 227 frequent pairs; 0.5 % -> 286 items, 40,755 candidate pairs; **0.1 % -> 1,021
  items, 520,710 candidate pairs** (main run).
- `apriori`: level-wise, up to triples; this is the baseline. `pcy`: pass 1 counts items and hashes
  every pair into `n_buckets` (NumPy int32 array; hash from `stable_hash` of the two sorted items),
  then a bitmap; pass 2 counts only pairs of frequent items in frequent buckets.
  **pcy(...) must return exactly apriori(max_size=2).** Optional extension: multistage PCY.
- Rules: every split X -> Y of frequent itemsets of size >= 2; support, confidence, lift; keep
  confidence >= 0.3. Already visible on the raw data: garlic powder + onion powder lift 13.9,
  cinnamon + nutmeg 8.6, baking powder + baking soda 6.3; olive oil + sugar 0.40 (avoided together).
- `pair_table` -> `PAIRS` (all frequent pairs with item and pair counts) for R4.
- Benchmark: A-Priori vs PCY for min_support {0.005, 0.002, 0.001} x n_buckets {1e5, 1e6, 1e7}:
  seconds, peak memory (`tracemalloc`), pairs counted in pass 2, frequent pairs. Check against mlxtend
  on the 1 % sample.

### 3.4.4 PMI ingredient graph + Louvain (R4, `graphs.py`)
- `pmi_weights`: p(a,b) = pair_count/N, p(a) = count_a/N; lift = p(a,b)/(p(a)p(b)); pmi = log2(lift);
  npmi = pmi / -log2 p(a,b) in [-1, 1]. **This weighting is the outside-course part.**
- `ingredient_graph(pairs, weight, min_weight)`: one graph per weighting in {count, lift, pmi, npmi};
  only positive associations; threshold argued with the degree distribution.
- `dominant_cuisine`: cuisine with the largest share among labelled recipes containing the ingredient,
  kept if share >= 0.4; labels for NMI.
- Louvain: `networkx.community.louvain_communities(weight="weight", resolution=1, seed=SEED)`, mean over
  5 seeds; spectral: scikit-learn `SpectralClustering(affinity="precomputed")`, k from the eigengap.
- `null_graph`: degree-preserving rewiring (10 swaps per edge); every modularity is reported next to
  its null value. `evaluate_partition`: n_communities, modularity, NMI on labelled ingredients.
- The result to show: which weighting gives communities closest to cuisines (NMI), and the top
  ingredients of each community. Optional: Node2Vec embeddings (scope cut 1).

### 3.4.5 Calorie-level classifier from LSH neighbours (R5, `classify.py`)
- Label: `calorie_level` (R1). Task: predict it from the recipe's ingredients only, via its neighbours.
- `train_test_split_recipes`: 80/20 by recipe (seed 42), recipes with a level only; a recipe and its
  canonical copy (`DUPLICATES`) always in the same set (**no leakage through copies**).
- `knn_predict`: vote of the k best neighbours (`NEIGHBOURS`) **that are in the training set**,
  weighted by Jaccard or not; no training neighbour -> NA, reported as coverage.
- Baselines: majority class; exact-Jaccard kNN by brute force on a 5,000-recipe sample (how much does
  LSH lose?); logistic regression on one-hot `ingredients_norm` (scikit-learn, supervised reference).
- `evaluate_classifier`: coverage, accuracy, macro-F1, confusion matrix; sweep k in {1, 3, 5, 10, 20}.
- Expected and honest: ingredient lists carry no quantities, so accuracy will be moderate; the report
  explains which classes are confused and why.

## 3.5 Test conventions
- pytest, `tests/test_<module>.py` owned by the module's role. Tests come **before** the code (from
  this contract), are seen failing, then made green. Never weaken, skip or delete a test.
- Frozen: `tests/conftest.py` (fixtures `mini_recipes`, `mini_interactions`: 8 recipes and 12 reviews,
  with a copy 102/103, the overflow duration, 0 minutes, > 5,000 kcal, a rating 0, a duplicate review,
  an unknown recipe, a self-review), `tests/fixtures/`, `tests/test_contract.py`.
- Toy examples with a known answer are mandatory: identical sets -> identical signatures; MinHash
  estimate within 0.1 of the true Jaccard (200 hashes); 8 hand-made baskets with known counts; PMI of a
  hand-computed pair; two cliques joined by one edge -> 2 Louvain communities; a 6-recipe neighbour table
  with known votes.
- `@pytest.mark.data`: needs `data/processed/` (auto-skipped in CI). `@pytest.mark.slow`: > 30 s.
- Reference checks (datasketch, mlxtend) use `pytest.importorskip` and the 1 % sample.
- Seeds: `config.SEED` everywhere. CI runs `pytest -m "not slow"` on every push and PR.

## 3.6 What we measure (every number in the report comes from a file in `results/`)

| File | Owner | Columns |
|---|---|---|
| data_summary.csv | R1 | metric, value (rows, flags, ingredient labels before/after, cuisines, calorie-level bounds) |
| runtime.csv | R1 | step, seconds, rows (full pipeline on one laptop; specs in the README) |
| itemsets_bench.csv | R3 | algorithm, min_support, n_buckets, seconds, peak_mb, pairs_counted, frequent_pairs |
| rules_top.csv | R3 | group, antecedent, consequent, support, confidence, lift |
| lsh_eval.csv | R2 | bands, rows, threshold, candidates, precision, recall, seconds |
| search_latency.csv | R2 | method, n_recipes, k, ms_per_query |
| graph_eval.csv | R4 | weighting, method, k, n_communities, modularity, null_modularity, nmi |
| communities.csv | R4 | weighting, community, size, top_ingredients, dominant_cuisine |
| classifier_eval.csv | R5 | method, k, weighted, coverage, accuracy, macro_f1 |
| confusion_matrix.csv | R5 | method, true_level, predicted_level, count |

Figures: `report/figures/*.png`, each produced by `scripts/fig_*.py` from a CSV.

## 3.7 Ownership, branches, stories, reviews
- Roles and owned files: 3.1. Stories `R<k>-S<n>` (role k, sprint n): steps in
  `docs/PROJECT_INSTRUCTIONS_FOR_CLAUDE_CODE.txt`, acceptance criteria in `docs/backlog.md`.
- Sprints: S1 Tue 10 -> Tue 17 Nov (correct on toy data and sample), S2 -> Tue 24 Nov (full data,
  evaluation; freeze at the session), S3 -> Sun 29 Nov (writing), S4 -> Thu 3 Dec (polish, hand-in).
- Branch per story: `r<k>/s<n>-<slug>` (e.g. `r2/s1-minhash`). `main` protected: PR, 1 approval,
  CI green, squash merge. Tag `sprint-<n>` at each review, `v1.0` at submission.
- Review circle (the reviewer of a role takes over if its owner is unavailable):
  R1's PRs -> R5, R2's -> R1, R3's -> R2, R4's -> R3, R5's -> R4.
- Contract changes: PR labelled `contract`, edited by a human, approved by all five, version bumped here.
