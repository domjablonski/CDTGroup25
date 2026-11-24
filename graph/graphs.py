"""RQ3: does a PMI-weighted ingredient graph reveal cuisines better than raw co-occurrence? (owner R4)

Course topic: graph clustering (Louvain, spectral clustering, modularity).
Outside the course: the PMI / NPMI weighting of the edges.

Input: the group's recipes.parquet (preprocessing/preprocessing.py, column ingredients_v2) and the
frequent ingredient pairs of R3's PCY (pcy/), or an exact count of the same pairs while PCY is not ready.
Louvain and spectral clustering come from networkx / scikit-learn; the pair table, the weights, the
thresholds, the null models and the evaluation are our own code.
"""
from collections import Counter, defaultdict
from itertools import combinations
from math import ceil
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

SEED = 42
WEIGHTS = ("count", "lift", "pmi", "npmi")
PAIR_COLUMNS = ["item_a", "item_b", "pair_count", "count_a", "count_b", "n_baskets"]
# Cuisine of a recipe = first of these tags found in its tags (most specific first)
CUISINE_TAGS = ["chinese", "thai", "japanese", "korean", "indian", "italian", "mexican", "greek",
                "french", "spanish", "german", "middle-eastern", "caribbean", "southern-united-states",
                "african", "asian"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- data

def find_recipes_file(root: Path = PROJECT_ROOT) -> Path:
    """recipes.parquet written by preprocessing.py (in a processed/ folder under a data* folder)."""
    matches = sorted(root.glob("data*/**/processed/recipes.parquet"))
    if not matches:
        raise FileNotFoundError(f"no data*/**/processed/recipes.parquet under {root}: run preprocessing first")
    return matches[0]


def cuisine_label(tags) -> str | None:
    """First tag of CUISINE_TAGS present in the recipe tags, or None."""
    tag_set = set(tags)
    return next((c for c in CUISINE_TAGS if c in tag_set), None)


def load_recipes(path: Path | None = None) -> pd.DataFrame:
    """id, tags, ingredients_v2 (lists of str) and cuisine."""
    df = pd.read_parquet(path or find_recipes_file(), columns=["id", "tags", "ingredients_v2"])
    for col in ("tags", "ingredients_v2"):  # Parquet gives NumPy arrays
        df[col] = df[col].map(list)
    df["cuisine"] = df["tags"].map(cuisine_label)
    return df


# ----------------------------------------------------------------------------- pairs

def count_pairs(baskets, min_support: float = 0.001) -> pd.DataFrame:
    """Exact counts of the frequent pairs of frequent items (count >= ceil(min_support * N)).
    Stand-in for R3's PCY output: same definition, so the same table."""
    baskets = [set(b) for b in baskets]
    n = len(baskets)
    threshold = ceil(min_support * n)
    item_counts = Counter(i for b in baskets for i in b)
    frequent = {i for i, c in item_counts.items() if c >= threshold}
    pair_counts = Counter()
    for b in baskets:
        pair_counts.update(combinations(sorted(i for i in b if i in frequent), 2))
    rows = [(a, b, c, item_counts[a], item_counts[b], n) for (a, b), c in pair_counts.items() if c >= threshold]
    return pd.DataFrame(rows, columns=PAIR_COLUMNS).sort_values(["item_a", "item_b"], ignore_index=True)


def pairs_from_pcy(singles: pd.DataFrame, pairs: pd.DataFrame, n_baskets: int) -> pd.DataFrame:
    """R3's PCY output (frequent_ingredients_singles: ingredient, support; frequent_ingredients_pairs:
    ingredient1, ingredient2, support) -> pair table with item counts and N."""
    support = dict(zip(singles["ingredient"], singles["support"]))
    a = pairs[["ingredient1", "ingredient2"]].min(axis=1)
    b = pairs[["ingredient1", "ingredient2"]].max(axis=1)
    out = pd.DataFrame({"item_a": a, "item_b": b, "pair_count": pairs["support"].astype(int)})
    out["count_a"] = out["item_a"].map(support).astype(int)
    out["count_b"] = out["item_b"].map(support).astype(int)
    out["n_baskets"] = int(n_baskets)
    return out[PAIR_COLUMNS].sort_values(["item_a", "item_b"], ignore_index=True)


def read_pcy_dir(pcy_dir: Path, n_recipes: int) -> pd.DataFrame:
    """Pair table from a folder of PCY outputs. Uses pairs.parquet when it is there (written by
    scripts/export_pcy_pairs.py, it carries the real number of baskets, e.g. after removing duplicates);
    otherwise converts Andreas's singles + pairs files, assuming one basket per recipe (n_recipes)."""
    full = pcy_dir / "pairs.parquet"
    if full.exists():
        pairs = pd.read_parquet(full)
        missing = [c for c in PAIR_COLUMNS if c not in pairs.columns]
        if missing:
            raise ValueError(f"{full} is missing columns {missing}")
        pairs = pairs[PAIR_COLUMNS].astype({"item_a": object, "item_b": object})
        return pairs.sort_values(["item_a", "item_b"], ignore_index=True)
    singles = pd.read_parquet(pcy_dir / "frequent_ingredients_singles.parquet")
    pairs = pd.read_parquet(pcy_dir / "frequent_ingredients_pairs.parquet")
    return pairs_from_pcy(singles, pairs, n_baskets=n_recipes)


def pmi_weights(pairs: pd.DataFrame) -> pd.DataFrame:
    """pairs + lift, pmi, npmi. p(a,b) = pair_count/N, p(a) = count_a/N; lift = p(a,b)/(p(a)p(b));
    pmi = log2(lift); npmi = pmi / -log2(p(a,b)), in [-1, 1]."""
    out = pairs.copy()
    n = out["n_baskets"].astype(float)
    p_ab = out["pair_count"] / n
    out["lift"] = p_ab / ((out["count_a"] / n) * (out["count_b"] / n))
    out["pmi"] = np.log2(out["lift"])
    h_ab = -np.log2(p_ab)  # 0 only if the pair is in every basket: perfect association, npmi = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        out["npmi"] = np.where(h_ab > 0, out["pmi"] / h_ab, 1.0)
    out["npmi"] = out["npmi"].clip(-1.0, 1.0)  # floating-point overshoot only
    return out


# ----------------------------------------------------------------------------- graph

def ingredient_graph(pairs: pd.DataFrame, weight: str = "npmi", min_weight: float = 0.1) -> nx.Graph:
    """Nodes = ingredients, edge (a, b) if its weight >= min_weight, edge attribute "weight" > 0.
    weight in WEIGHTS ("count" = pair_count). For lift/pmi/npmi only positive associations
    (lift > 1) become edges: pairs that avoid each other are not "close" ingredients."""
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    df = pairs if {"lift", "pmi", "npmi"} <= set(pairs.columns) else pmi_weights(pairs)
    col = "pair_count" if weight == "count" else weight
    keep = (df[col] >= min_weight) & (df[col] > 0)
    if weight != "count":
        keep &= df["pmi"] > 0
    G = nx.Graph()
    G.add_weighted_edges_from(zip(df.loc[keep, "item_a"], df.loc[keep, "item_b"], df.loc[keep, col].astype(float)))
    return G


def largest_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def threshold_sweep(pairs: pd.DataFrame, weight: str, thresholds) -> pd.DataFrame:
    """Graph size for each threshold: nodes, edges, mean degree, share of the frequent ingredients
    (nodes of the unthresholded positive graph) that stay in the largest connected component."""
    df = pmi_weights(pairs)
    n_items = ingredient_graph(df, weight, min_weight=-np.inf).number_of_nodes()
    rows = []
    for t in thresholds:
        G = ingredient_graph(df, weight, min_weight=t)
        lcc = largest_component(G).number_of_nodes()
        rows.append({"weighting": weight, "threshold": float(t), "nodes": G.number_of_nodes(),
                     "edges": G.number_of_edges(),
                     "mean_degree": 2 * G.number_of_edges() / max(G.number_of_nodes(), 1),
                     "lcc_share": lcc / max(n_items, 1)})
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame, min_lcc_share: float = 0.9) -> float:
    """Largest threshold whose largest component still holds >= min_lcc_share of the ingredients:
    we remove as many weak edges as possible without cutting ingredients off the graph."""
    ok = sweep[sweep["lcc_share"] >= min_lcc_share]
    return float(ok["threshold"].max()) if len(ok) else float(sweep["threshold"].min())


def dominant_cuisine(recipes: pd.DataFrame, min_share: float = 0.4, min_recipes: int = 10) -> dict:
    """Ingredient -> cuisine with the highest share among the labelled recipes containing it, if that
    share >= min_share and the ingredient is in >= min_recipes labelled recipes (ties: alphabetical)."""
    labelled = recipes.loc[recipes["cuisine"].notna(), ["ingredients_v2", "cuisine"]]
    counts: dict[str, Counter] = defaultdict(Counter)
    for ingredients, cuisine in zip(labelled["ingredients_v2"], labelled["cuisine"]):
        for ingredient in set(ingredients):
            counts[ingredient][cuisine] += 1
    labels = {}
    for ingredient, by_cuisine in counts.items():
        total = sum(by_cuisine.values())
        cuisine, n = min(by_cuisine.items(), key=lambda kv: (-kv[1], kv[0]))
        if total >= min_recipes and n / total >= min_share:
            labels[ingredient] = cuisine
    return labels


# ----------------------------------------------------------------------------- clustering

def _relabel_by_size(groups) -> dict:
    """Node -> id, 0 = largest community (ties: smallest node name), independent of the algorithm's order."""
    ordered = sorted((sorted(g, key=str) for g in groups), key=lambda g: (-len(g), str(g[0])))
    return {node: cid for cid, g in enumerate(ordered) for node in g}


def louvain(G: nx.Graph, seed: int = SEED, resolution: float = 1.0) -> dict:
    """Node -> community id (networkx louvain_communities on the edge weights)."""
    return _relabel_by_size(nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed))


def eigengap_k(G: nx.Graph, k_max: int = 15) -> int:
    """Number of clusters suggested by the largest gap between consecutive eigenvalues of the
    normalised Laplacian (k >= 2). Use on a connected graph."""
    L = nx.normalized_laplacian_matrix(G, weight="weight").toarray()
    vals = np.sort(np.linalg.eigvalsh(L))[: k_max + 1]
    gaps = np.diff(vals)  # gaps[i] = vals[i+1] - vals[i]; a gap after vals[k-1] suggests k clusters
    return int(np.argmax(gaps[1:]) + 2)


def spectral(G: nx.Graph, k: int, seed: int = SEED) -> dict:
    """Node -> cluster id, spectral clustering on the weighted adjacency matrix (connected graph)."""
    from sklearn.cluster import SpectralClustering

    nodes = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr")
    A.indices, A.indptr = A.indices.astype(np.int32), A.indptr.astype(np.int32)  # sklearn wants int32
    labels = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="cluster_qr",
                                random_state=seed).fit_predict(A)
    groups = defaultdict(list)
    for node, label in zip(nodes, labels):
        groups[label].append(node)
    return _relabel_by_size(groups.values())


def null_graph(G: nx.Graph, seed: int = SEED, mode: str = "rewire") -> nx.Graph:
    """Randomised copy of G that keeps every degree and the multiset of edge weights.
    mode "rewire": edges rewired by double edge swaps (10 per edge), weights shuffled onto them
                   -> is there community structure at all beyond the degrees?
    mode "shuffle": same edges, weights shuffled between them
                   -> do the weights themselves carry the community structure?"""
    if mode not in ("rewire", "shuffle"):
        raise ValueError(f"mode must be 'rewire' or 'shuffle', got {mode!r}")
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from(G.edges())
    m = H.number_of_edges()
    if mode == "rewire" and m >= 2 and H.number_of_nodes() >= 4:
        try:
            nx.double_edge_swap(H, nswap=10 * m, max_tries=1000 * m, seed=seed)
        except nx.NetworkXAlgorithmError:
            pass  # very dense graph: fewer swaps were possible, degrees are still preserved
    weights = np.array([d.get("weight", 1.0) for _, _, d in G.edges(data=True)], dtype=float)
    np.random.default_rng(seed).shuffle(weights)
    for (u, v), w in zip(sorted(H.edges(), key=lambda e: (str(e[0]), str(e[1]))), weights):
        H[u][v]["weight"] = float(w)
    return H


def evaluate_partition(G: nx.Graph, partition: dict, labels: dict | None = None) -> dict:
    """n_communities, weighted modularity, and on the labelled nodes NMI and AMI (AMI corrects NMI for
    chance, which grows with the number of communities). nmi/ami None without labelled nodes."""
    from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

    groups = defaultdict(set)
    for node in G:
        groups[partition[node]].add(node)
    modularity = nx.community.modularity(G, list(groups.values()), weight="weight") if G.number_of_edges() else 0.0
    res = {"n_communities": len(groups), "modularity": float(modularity), "nmi": None, "ami": None, "n_labelled": 0}
    common = [n for n in G if labels and n in labels]
    if common:
        y_true, y_pred = [labels[n] for n in common], [partition[n] for n in common]
        res.update(nmi=float(normalized_mutual_info_score(y_true, y_pred)),
                   ami=float(adjusted_mutual_info_score(y_true, y_pred)), n_labelled=len(common))
    return res
