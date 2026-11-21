# graph/ (R4): PMI-weighted ingredient graph and Louvain (RQ3)

**Question.** Does a PMI-weighted ingredient graph reveal cuisines better than raw co-occurrence?

Course topic: graph clustering (Louvain, spectral clustering, modularity). Outside the course: PMI / NPMI
edge weights (lift = p(a,b) / (p(a) p(b)), pmi = log2 lift, npmi = pmi / -log2 p(a,b)).

## Run

From the repo root, after `python preprocessing/preprocessing.py` (needs `ingredients_v2`):

```bash
pytest graph -q                                   # 33 tests on toy examples
python graph/run_graph.py                         # pairs counted here (same definition as PCY)
python graph/run_graph.py --pcy-dir <folder>      # pairs from R3's PCY output files
python graph/sensitivity.py                       # robustness: 9 parameter settings (about 75 s)
```

`graph_R4.ipynb` walks through the whole analysis with comments (for the appendix, about 1 min), and
`report_RQ3.md` is the draft of the RQ3 section of the report (1,650 characters without spaces).

`--pcy-dir` reads `frequent_ingredients_singles.parquet` and `frequent_ingredients_pairs.parquet`
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
