"""RQ3 evaluation (R4): which edge weighting makes the ingredient graph reveal cuisines?

For each weighting (count, lift, pmi, npmi):
  1. threshold chosen with the degree structure (largest threshold that keeps >= 90 % of the frequent
     ingredients in the largest connected component), see results/threshold_sweep.csv
  2. Louvain on 5 seeds (mean and std) and spectral clustering with k from the eigengap
  3. modularity next to two null models (rewired edges, shuffled weights)
  4. NMI and AMI against the dominant cuisine of each ingredient

Input: data*/**/processed/recipes.parquet from preprocessing.py, and the frequent pairs of R3's PCY
(--pcy-dir, files frequent_ingredients_singles.parquet + frequent_ingredients_pairs.parquet). Without
--pcy-dir the pairs are counted exactly here (same definition), which is what we use until PCY is run
with the same support on ingredients_v2.

Usage (repo root): python graph/run_graph.py [--pcy-dir data/raw/processed] [--min-support 0.001]
"""
import argparse
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

import graphs  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
SEEDS = [42, 43, 44, 45, 46]
MIN_LCC_SHARE = 0.9

# Reference categorical palette (light mode), fixed order: community c gets slot c, others gray
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER, INK, INK2, SURFACE, GRID = "#c3c2b7", "#0b0b0b", "#52514e", "#fcfcfb", "#e1e0d9"
WEIGHT_COLOUR = dict(zip(graphs.WEIGHTS, SERIES))


def load_pairs(recipes: pd.DataFrame, pcy_dir: Path | None, min_support: float) -> tuple[pd.DataFrame, str]:
    if pcy_dir:
        return graphs.read_pcy_dir(pcy_dir, n_recipes=len(recipes)), f"PCY ({pcy_dir})"
    return graphs.count_pairs(recipes["ingredients_v2"], min_support), f"exact count, support {min_support:.2%}"


def sweep_all(pairs: pd.DataFrame) -> pd.DataFrame:
    """Thresholds = deciles of each weighting's positive values."""
    df = graphs.pmi_weights(pairs)
    out = []
    for w in graphs.WEIGHTS:
        col = "pair_count" if w == "count" else w
        values = df.loc[df["pmi"] > 0, col] if w != "count" else df[col]
        thresholds = np.unique(np.quantile(values, np.linspace(0, 0.9, 19)))
        out.append(graphs.threshold_sweep(df, w, thresholds))
    return pd.concat(out, ignore_index=True)


def evaluate(G: nx.Graph, labels: dict) -> list[dict]:
    rows = []
    lcc = graphs.largest_component(G)
    # Louvain, 5 seeds, on the whole graph
    runs = [graphs.evaluate_partition(G, graphs.louvain(G, seed=s), labels) for s in SEEDS]
    null_rewire = graphs.null_graph(G, mode="rewire")
    null_shuffle = graphs.null_graph(G, mode="shuffle")
    q_rewire = graphs.evaluate_partition(null_rewire, graphs.louvain(null_rewire))["modularity"]
    shuffle_res = graphs.evaluate_partition(null_shuffle, graphs.louvain(null_shuffle), labels)
    rows.append({"method": "louvain", "k": None,
                 "n_communities": np.mean([r["n_communities"] for r in runs]),
                 "modularity": np.mean([r["modularity"] for r in runs]),
                 "modularity_std": np.std([r["modularity"] for r in runs]),
                 "null_rewire_modularity": q_rewire, "null_shuffle_modularity": shuffle_res["modularity"],
                 "nmi": np.mean([r["nmi"] for r in runs]), "ami": np.mean([r["ami"] for r in runs]),
                 "ami_std": np.std([r["ami"] for r in runs]), "null_shuffle_ami": shuffle_res["ami"],
                 "n_labelled": runs[0]["n_labelled"]})
    # Spectral on the largest connected component: k from the eigengap, and k = Louvain's number of
    # communities (same granularity, so the two algorithms can be compared fairly)
    k_louvain = int(round(rows[0]["n_communities"]))
    for method, k in (("spectral_eigengap", graphs.eigengap_k(lcc)), ("spectral_k_louvain", k_louvain)):
        res = graphs.evaluate_partition(lcc, graphs.spectral(lcc, k), labels)
        rows.append({"method": method, "k": k, "n_communities": res["n_communities"],
                     "modularity": res["modularity"], "nmi": res["nmi"], "ami": res["ami"],
                     "n_labelled": res["n_labelled"]})
    return rows


def community_table(G: nx.Graph, partition: dict, labels: dict, counts: dict, weighting: str) -> pd.DataFrame:
    rows = []
    for c in sorted(set(partition.values())):
        members = sorted((n for n in G if partition[n] == c), key=lambda n: -counts[n])
        cuisines = Counter(labels[n] for n in members if n in labels)
        top_cuisine, n_top = cuisines.most_common(1)[0] if cuisines else ("", 0)
        rows.append({"weighting": weighting, "community": c, "size": len(members),
                     "top_ingredients": ", ".join(members[:10]), "dominant_cuisine": top_cuisine,
                     "cuisine_share": n_top / max(sum(cuisines.values()), 1), "n_labelled": sum(cuisines.values())})
    return pd.DataFrame(rows)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(OTHER)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_degrees(graphs_by_weight: dict, path: Path) -> None:
    """Complementary cumulative degree distribution of each chosen graph (log-log)."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for w, G in graphs_by_weight.items():
        deg = np.sort([d for _, d in G.degree()])
        ccdf = 1 - np.arange(len(deg)) / len(deg)
        # lift and pmi keep the same edges (pmi = log2 lift): lift dashed so both stay visible
        ax.loglog(deg, ccdf, color=WEIGHT_COLOUR[w], linewidth=2, linestyle="--" if w == "lift" else "-",
                  label=w + (" (same edges as pmi)" if w == "lift" else ""))
    ax.set_xlabel("degree (number of ingredients linked)", fontsize=8, color=INK2)
    ax.set_ylabel("share of ingredients with degree ≥ x", fontsize=8, color=INK2)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_graph(G: nx.Graph, partition: dict, counts: dict, path: Path, n_nodes: int = 150) -> None:
    """The n_nodes most frequent ingredients coloured by community; communities are pulled around
    their own centre in the layout (visual aid only, communities come from Louvain on the full graph)."""
    nodes = sorted(G, key=lambda n: -counts[n])[:n_nodes]
    H = G.subgraph(nodes).copy()
    H.remove_nodes_from([n for n in list(H) if H.degree(n) == 0])
    comms = sorted({partition[n] for n in H})
    colour = {c: (SERIES[c] if c < len(SERIES) else OTHER) for c in comms}
    L = nx.Graph()
    for u, v, d in H.edges(data=True):
        L.add_edge(u, v, weight=d["weight"] * (5 if partition[u] == partition[v] else 1))
    rng = np.random.default_rng(graphs.SEED)
    centre = {c: (math.cos(2 * math.pi * i / len(comms)), math.sin(2 * math.pi * i / len(comms)))
              for i, c in enumerate(comms)}
    init = {n: np.array(centre[partition[n]]) + rng.normal(0, 0.15, 2) for n in H}
    pos = nx.spring_layout(L, pos=init, weight="weight", seed=graphs.SEED, k=0.9 / math.sqrt(len(H)), iterations=60)

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    nx.draw_networkx_edges(H, pos, ax=ax, width=0.4, edge_color=OTHER, alpha=0.45)
    scale = 900 / max(counts[n] for n in H)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=[30 + scale * counts[n] for n in H],
                           node_color=[colour[partition[n]] for n in H], edgecolors=SURFACE, linewidths=1.0)
    span = np.ptp(np.array(list(pos.values())), axis=0)
    placed = []
    for n in sorted(H, key=lambda n: -counts[n]):
        x, y = pos[n]
        if all(abs(x - px) > 0.09 * span[0] or abs(y - py) > 0.035 * span[1] for px, py in placed):
            ax.annotate(n, (x, y), xytext=(0, 7 + math.sqrt(30 + scale * counts[n]) / 2),
                        textcoords="offset points", ha="center", fontsize=7, color=INK)
            placed.append((x, y))
        if len(placed) >= 45:
            break
    sizes = Counter(partition.values())
    top = {c: [n for n in sorted(G, key=lambda n: -counts[n]) if partition[n] == c][:3] for c in comms}
    handles = [plt.Line2D([], [], marker="o", ls="", color=colour[c], markersize=7,
                          label=f"{c} · {', '.join(top[c])} ({sizes[c]} ingredients)") for c in comms]
    ax.legend(handles=handles, loc="lower left", fontsize=7, frameon=False, labelcolor=INK2,
              title="Louvain community · most frequent ingredients", title_fontsize=7)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pcy-dir", type=Path, default=None)
    parser.add_argument("--min-support", type=float, default=0.001)
    parser.add_argument("--recipes", type=Path, default=None)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    recipes = graphs.load_recipes(args.recipes)
    pairs, source = load_pairs(recipes, args.pcy_dir, args.min_support)
    counts = dict(zip(pairs["item_a"], pairs["count_a"])) | dict(zip(pairs["item_b"], pairs["count_b"]))
    labels = graphs.dominant_cuisine(recipes)
    print(f"{len(recipes):,} recipes, {recipes['cuisine'].notna().sum():,} with a cuisine tag | pairs: {source}: "
          f"{len(counts):,} ingredients, {len(pairs):,} pairs | {len(labels):,} ingredients with a dominant cuisine "
          f"({time.perf_counter() - t0:.0f} s)")

    sweep = sweep_all(pairs)
    sweep.to_csv(RESULTS / "threshold_sweep.csv", index=False, float_format="%.4f")
    weighted = graphs.pmi_weights(pairs)

    rows, chosen, partitions = [], {}, {}
    for w in graphs.WEIGHTS:
        t = time.perf_counter()
        threshold = graphs.choose_threshold(sweep[sweep["weighting"] == w], MIN_LCC_SHARE)
        G = graphs.ingredient_graph(weighted, w, min_weight=threshold)
        chosen[w], partitions[w] = G, graphs.louvain(G)
        lcc_share = sweep.loc[(sweep["weighting"] == w) & (sweep["threshold"] == threshold), "lcc_share"].item()
        for r in evaluate(G, labels):
            rows.append({"weighting": w, "threshold": threshold, "nodes": G.number_of_nodes(),
                         "edges": G.number_of_edges(), "lcc_share": lcc_share, **r})
        lv = rows[-3]
        print(f"{w:>5}: threshold {threshold:.3g}, {G.number_of_nodes()} nodes, {G.number_of_edges():,} edges | "
              f"Louvain {lv['n_communities']:.1f} comm., Q {lv['modularity']:.3f} (rewired {lv['null_rewire_modularity']:.3f}, "
              f"shuffled {lv['null_shuffle_modularity']:.3f}), AMI {lv['ami']:.3f} (shuffled {lv['null_shuffle_ami']:.3f}) | "
              f"spectral k={rows[-2]["k"]} AMI {rows[-2]["ami"]:.3f}, k={rows[-1]["k"]} AMI {rows[-1]["ami"]:.3f} ({time.perf_counter() - t:.0f} s)")

    evals = pd.DataFrame(rows)
    evals.to_csv(RESULTS / "graph_eval.csv", index=False, float_format="%.4f")
    louv = evals[evals["method"] == "louvain"].set_index("weighting")
    best = louv["ami"].idxmax()
    pd.concat([community_table(chosen[w], partitions[w], labels, counts, w) for w in graphs.WEIGHTS]).to_csv(
        RESULTS / "communities.csv", index=False, float_format="%.3f")
    plot_degrees(chosen, FIGURES / "degree_distribution.png")
    plot_graph(chosen[best], partitions[best], counts, FIGURES / f"graph_{best}.png")
    if best != "count":
        plot_graph(chosen["count"], partitions["count"], counts, FIGURES / "graph_count.png")
    print(f"best weighting by AMI (Louvain): {best} | done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
