"""R4-S1 first look at the ingredient graph (PROVISIONAL, owner R4).

R3's PAIRS file (config.PAIRS) is not delivered yet, so this script counts the frequent ingredient
pairs itself with a plain exact count (no PCY, it is not the RQ1 code) and writes them to a separate
file, never to config.PAIRS. Then, for every weighting of graphs.WEIGHTS: graph, Louvain, modularity
next to the null graph, NMI against the dominant cuisine of each ingredient.

Outputs (all provisional, replaced in R4-S2 by results/graph_eval.csv and results/communities.csv):
  data/processed/pairs_provisional_r4.parquet   (git-ignored)
  results/r4_provisional_graph_eval.csv
  results/r4_provisional_communities.csv
  report/figures/r4_provisional_npmi_graph.png

Usage (repo root, venv active):
  python scripts/r4_first_look.py [--raw path/to/RAW_recipes.csv] [--min-support 0.001]
"""
import argparse
import itertools
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from foodcom import cleaning, config, graphs, schemas  # noqa: E402

PAIRS_PROVISIONAL = config.DATA_PROCESSED / "pairs_provisional_r4.parquet"
# Provisional thresholds (R4-S2 argues them with the degree distribution)
MIN_WEIGHT = {"count": 0, "lift": math.sqrt(2), "pmi": 0.5, "npmi": 0.1}

# Reference categorical palette (dataviz skill, light mode), fixed order; "other" communities in gray
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER, INK, MUTED, SURFACE = "#c3c2b7", "#0b0b0b", "#898781", "#fcfcfb"


def load_recipes(raw: Path) -> pd.DataFrame:
    if config.RECIPES.exists():
        return cleaning.load_recipes()
    print(f"cleaning {raw} (about 1 min) ...")
    recipes = cleaning.clean_recipes(pd.read_csv(raw))
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    recipes.to_parquet(config.RECIPES, index=False)
    return recipes


def provisional_pairs(baskets: list[list[str]], min_support: float) -> pd.DataFrame:
    """Exact counts of the pairs of frequent ingredients that are frequent themselves (schemas.PAIRS)."""
    n = len(baskets)
    threshold = math.ceil(min_support * n)
    item_counts = Counter(i for b in baskets for i in b)
    frequent = {i for i, c in item_counts.items() if c >= threshold}
    pair_counts = Counter()
    for b in baskets:
        pair_counts.update(itertools.combinations(sorted(i for i in b if i in frequent), 2))
    rows = [(a, b, c, item_counts[a], item_counts[b]) for (a, b), c in pair_counts.items() if c >= threshold]
    pairs = pd.DataFrame(rows, columns=["item_a", "item_b", "pair_count", "count_a", "count_b"])
    pairs["n_baskets"] = n
    schemas.validate(pairs, "pairs")
    print(f"{n:,} baskets, support {min_support:.2%} (>= {threshold} recipes): "
          f"{len(frequent):,} frequent ingredients, {len(pairs):,} frequent pairs")
    return pairs.sort_values(["item_a", "item_b"], ignore_index=True)


def community_table(G: nx.Graph, partition: dict, labels: dict, weighting: str, top: int = 10) -> pd.DataFrame:
    rows = []
    for c in sorted(set(partition.values())):
        members = [n for n in G if partition[n] == c]
        members.sort(key=lambda n: -G.nodes[n].get("count", 0))
        cuisines = Counter(labels[n] for n in members if n in labels)
        rows.append({"weighting": weighting, "community": c, "size": len(members),
                     "top_ingredients": ", ".join(members[:top]),
                     "dominant_cuisine": cuisines.most_common(1)[0][0] if cuisines else ""})
    return pd.DataFrame(rows)


def draw(G: nx.Graph, partition: dict, path: Path, n_nodes: int = 150) -> None:
    """The n_nodes most frequent ingredients, coloured by community id (fixed order: community c gets
    slot c; ids >= 8 in gray). Layout: each community is pulled around its own centre (visual aid
    only, the communities themselves come from Louvain on the whole graph)."""
    nodes = sorted(G, key=lambda n: -G.nodes[n].get("count", 0))[:n_nodes]
    H = G.subgraph(nodes).copy()
    H.remove_nodes_from([n for n in list(H) if H.degree(n) == 0])
    comms = sorted({partition[n] for n in H})
    colour = {c: (SERIES[c] if c < len(SERIES) else OTHER) for c in comms}

    # layout graph: intra-community edges 5x stronger; start each community on a circle
    L = nx.Graph()
    for u, v, d in H.edges(data=True):
        L.add_edge(u, v, weight=d["weight"] * (5 if partition[u] == partition[v] else 1))
    rng = np.random.default_rng(config.SEED)
    centre = {c: (math.cos(2 * math.pi * i / len(comms)), math.sin(2 * math.pi * i / len(comms)))
              for i, c in enumerate(comms)}
    init = {n: np.array(centre[partition[n]]) + rng.normal(0, 0.15, 2) for n in H}
    pos = nx.spring_layout(L, pos=init, weight="weight", seed=config.SEED, k=0.9 / math.sqrt(len(H)),
                           iterations=60)

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    nx.draw_networkx_edges(H, pos, ax=ax, width=0.4, edge_color=OTHER, alpha=0.45)
    counts = [H.nodes[n]["count"] for n in H]
    scale = 900 / max(counts)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=[30 + scale * c for c in counts],
                           node_color=[colour[partition[n]] for n in H], edgecolors=SURFACE, linewidths=1.0)
    # label the most frequent ingredients of each community, skipping labels that would collide
    span = np.ptp(np.array(list(pos.values())), axis=0)
    placed = []
    for n in sorted(H, key=lambda n: -H.nodes[n]["count"]):
        x, y = pos[n]
        if all(abs(x - px) > 0.09 * span[0] or abs(y - py) > 0.035 * span[1] for px, py in placed):
            ax.annotate(n, (x, y), xytext=(0, 7 + math.sqrt(30 + scale * H.nodes[n]["count"]) / 2),
                        textcoords="offset points", ha="center", fontsize=7, color=INK)
            placed.append((x, y))
        if len(placed) >= 40:
            break
    top = {c: [n for n in sorted(G, key=lambda n: -G.nodes[n].get("count", 0)) if partition[n] == c][:3]
           for c in comms}
    sizes = Counter(partition.values())
    handles = [plt.Line2D([], [], marker="o", ls="", color=colour[c], markersize=7,
                          label=f"{c} · {', '.join(top[c])} ({sizes[c]} ingredients)") for c in comms]
    ax.legend(handles=handles, loc="lower left", fontsize=7, frameon=False, labelcolor="#52514e",
              title="Louvain community · most frequent ingredients", title_fontsize=7)
    ax.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=config.RAW_RECIPES)
    parser.add_argument("--min-support", type=float, default=0.001)
    args = parser.parse_args()

    t0 = time.perf_counter()
    recipes = load_recipes(args.raw)
    pairs = provisional_pairs(recipes["ingredients_norm"].tolist(), args.min_support)
    pairs.to_parquet(PAIRS_PROVISIONAL, index=False)
    item_count = dict(zip(pairs["item_a"], pairs["count_a"])) | dict(zip(pairs["item_b"], pairs["count_b"]))
    labels = graphs.dominant_cuisine(recipes, min_share=0.4)
    print(f"dominant cuisine: {len(labels):,} labelled ingredients ({time.perf_counter() - t0:.0f} s)")

    evals, tables = [], []
    for weighting in graphs.WEIGHTS:
        t = time.perf_counter()
        G = graphs.ingredient_graph(pairs, weight=weighting, min_weight=MIN_WEIGHT[weighting])
        nx.set_node_attributes(G, {n: item_count[n] for n in G}, "count")
        part = graphs.louvain(G, seed=config.SEED)
        res = graphs.evaluate_partition(G, part, labels)
        H = graphs.null_graph(G, seed=config.SEED)
        null = graphs.evaluate_partition(H, graphs.louvain(H, seed=config.SEED))
        evals.append({"weighting": weighting, "method": "louvain", "k": None,
                      "n_communities": res["n_communities"], "modularity": res["modularity"],
                      "null_modularity": null["modularity"], "nmi": res["nmi"],
                      "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                      "min_weight": MIN_WEIGHT[weighting]})
        tables.append(community_table(G, part, labels, weighting))
        print(f"{weighting:>5}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, "
              f"{res['n_communities']} communities, Q {res['modularity']:.3f} (null {null['modularity']:.3f}), "
              f"NMI {res['nmi']:.3f} ({time.perf_counter() - t:.0f} s)")
        if weighting == "npmi":
            draw(G, part, config.FIGURES / "r4_provisional_npmi_graph.png")

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(evals).to_csv(config.RESULTS / "r4_provisional_graph_eval.csv", index=False, float_format="%.4f")
    pd.concat(tables).to_csv(config.RESULTS / "r4_provisional_communities.csv", index=False)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
