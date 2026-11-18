"""RQ3 robustness (R4): does the ranking of the weightings depend on the parameters we chose?

One parameter changes at a time, the others stay at their main value (in brackets):
  support of the pairs        0.05 %, [0.1 %], 0.2 %
  connectivity rule           largest component keeps 80 %, [90 %], 95 % of the ingredients
  dominant cuisine share      30 %, [40 %], 50 %
For each setting and weighting: Louvain on 5 seeds, mean AMI against the dominant cuisines.

Usage (repo root): python graph/sensitivity.py  ->  graph/results/sensitivity.csv
"""
import time

import numpy as np
import pandas as pd

import graphs
from run_graph import RESULTS, SEEDS, sweep_all

MAIN = {"support": 0.001, "min_lcc_share": 0.9, "min_share": 0.4}
GRID = {"support": [0.0005, 0.001, 0.002], "min_lcc_share": [0.8, 0.9, 0.95], "min_share": [0.3, 0.4, 0.5]}


def run(recipes, support, min_lcc_share, min_share, cache):
    if support not in cache:
        pairs = graphs.count_pairs(recipes["ingredients_v2"], support)
        cache[support] = (graphs.pmi_weights(pairs), sweep_all(pairs))
    weighted, sweep = cache[support]
    labels = graphs.dominant_cuisine(recipes, min_share=min_share)
    rows = []
    for w in graphs.WEIGHTS:
        threshold = graphs.choose_threshold(sweep[sweep["weighting"] == w], min_lcc_share)
        G = graphs.ingredient_graph(weighted, w, min_weight=threshold)
        runs = [graphs.evaluate_partition(G, graphs.louvain(G, seed=s), labels) for s in SEEDS]
        rows.append({"weighting": w, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                     "n_labelled": runs[0]["n_labelled"],
                     "n_communities": np.mean([r["n_communities"] for r in runs]),
                     "modularity": np.mean([r["modularity"] for r in runs]),
                     "ami": np.mean([r["ami"] for r in runs]), "ami_std": np.std([r["ami"] for r in runs])})
    return rows


def main() -> None:
    t0 = time.perf_counter()
    recipes = graphs.load_recipes()
    cache, rows = {}, []
    for param, values in GRID.items():
        for value in values:
            setting = {**MAIN, param: value}
            for r in run(recipes, setting["support"], setting["min_lcc_share"], setting["min_share"], cache):
                rows.append({"varied": param, "value": value, "is_main": setting == MAIN, **r})
            print(f"{param}={value}: done ({time.perf_counter() - t0:.0f} s)")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "sensitivity.csv", index=False, float_format="%.4f")
    table = out.pivot_table(index=["varied", "value"], columns="weighting", values="ami").round(3)[list(graphs.WEIGHTS)]
    print(table.to_string())


if __name__ == "__main__":
    main()
