# graph/ (R4): PMI-weighted ingredient graph and Louvain (RQ3)

**Question.** Does a PMI-weighted ingredient graph reveal cuisines better than raw co-occurrence?

Course topic: graph clustering (Louvain, spectral clustering, modularity). Outside the course: PMI / NPMI
edge weights (lift = p(a,b) / (p(a) p(b)), pmi = log2 lift, npmi = pmi / -log2 p(a,b)).

## Run

From the repo root, after `python preprocessing/preprocessing.py` (needs `ingredients_v2`):

```bash
pytest graph -q                                   # 50 tests on toy examples
python graph/run_graph.py                         # pairs counted here (same definition as PCY)
python graph/run_graph.py --pcy-dir <folder>      # pairs from R3's PCY output files
python graph/sensitivity.py                       # robustness: 9 parameter settings (about 75 s)
```

`graph_R4.ipynb` walks through the whole analysis with comments (for the appendix, about 1 min), and
`report_RQ3.md` is the draft of the RQ3 section of the report (1,650 characters without spaces).

`--pcy-dir` reads `pairs.parquet` when the folder has it (written by `scripts/export_pcy_pairs.py`, it
carries the real number of baskets, also after removing duplicates). Otherwise it reads
`frequent_ingredients_singles.parquet` and `frequent_ingredients_pairs.parquet`
(R3's format) and converts them with `graphs.pairs_from_pcy`. Checked on R3's current script (raw
ingredients, support 1000): the converted PCY output is identical to the exact count (735 pairs). PCY must
run on `ingredients_v2` with support 0.1 % of the recipes to give the pairs used here. Done on 21 Nov with
`pcy/PCY_ingredients_structured.py` (unchanged, column ingredients_v2, support 229 recipes): 5,203 pairs, and
`results/graph_eval.csv` and `results/communities.csv` are byte-for-byte identical to the exact-count run.

## Method

1. Nodes = ingredients that appear in a frequent pair (support >= 0.1 % of recipes). One graph per
   weighting: count, lift, pmi, npmi. For lift / pmi / npmi only positive associations (lift > 1).
2. Threshold per weighting: the largest one that keeps >= 90 % of the ingredients in the largest
   connected component (`results/threshold_sweep.csv`, `figures/degree_distribution.png`).
3. Louvain (5 seeds, mean); spectral clustering with k from the eigengap and with Louvain's k.
4. Two null models, both keeping every degree and the edge weights: edges rewired (is there structure
   beyond the degrees?) and weights shuffled on the same edges (do the weights carry the structure?).
5. Labels: dominant cuisine of an ingredient (>= 40 % of the labelled recipes containing it, >= 10 such
   recipes; cuisine = first of 16 cuisine tags). NMI and AMI (NMI corrected for chance).

## Outputs

| File | Content |
|---|---|
| `results/graph_eval.csv` | one row per weighting and method: threshold, size, modularity, null modularities, NMI, AMI |
| `results/communities.csv` | every community: size, 10 most frequent ingredients, dominant cuisine and its share |
| `results/threshold_sweep.csv` | graph size and connectivity for each candidate threshold |
| `results/sensitivity.csv` | Louvain AMI per weighting when support, connectivity rule or label share change |
| `figures/graph_pmi.png`, `figures/graph_count.png` | 150 most frequent ingredients coloured by Louvain community |
| `figures/degree_distribution.png` | degree distribution of each chosen graph |

## Cuisine prediction from the communities (`run_cuisine.py`)

**Question.** Can the communities of the ingredient graph predict a recipe's cuisine, and how do they compare
with a supervised model on all ingredients (R5's LightGBM)? Same split as R5: `splits/cuisine_split.csv`
(11 cuisines, train / val / test).

1. Graph from all recipes except the test ones (no label used), weighted by count or PMI.
2. Louvain at several resolutions; each recipe -> share of its ingredients in each community.
3. Predictors: community -> cuisine table (with / without the cuisine's prior) and a logistic regression on the
   profile; validation picks the logistic regression.
4. Count and PMI compared on test **at the same number of communities** (10, 30, 100; resolution searched for each
   graph, 5 Louvain seeds), refit on train + val. At high resolution the count graph breaks into almost one
   community per ingredient, so comparing at equal size is the fair comparison.

| Test (macro-F1, mean of 5 seeds) | ~10 communities | ~30 | ~100 |
|---|---|---|---|
| count graph | 0.290 | 0.371 | 0.454 |
| PMI graph | **0.305** | **0.388** | **0.466** |

References on test: majority class 0.03, logistic regression on all ingredients 0.62.

```bash
python graph/run_cuisine.py     # about 7 minutes
```
Outputs: `results/cuisine_val.csv` (validation sweep), `cuisine_val_references.csv`, `cuisine_test.csv`,
`cuisine_per_class.csv`, `cuisine_confusion.csv`, `cuisine_communities.csv` (the ~10 PMI communities and their two
main cuisines) and `figures/cuisine_resolution.png`. Tests: `graph/test_cuisine.py` (15 tests).
