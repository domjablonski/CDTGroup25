"""RQ3 – cuisine discovery with a PMI-weighted ingredient graph (owner: R4, docs/CONTRACT.md 3.4.4).

Course topic: graph clustering (Louvain, spectral, modularity). Outside the course: the PMI / NPMI
weighting (and optionally Node2Vec embeddings). Louvain and spectral may use networkx / scikit-learn;
the graph construction, the weights, the null model and the evaluation are ours.
Input: config.PAIRS from R3 (schemas.PAIRS).
"""
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

WEIGHTS = ("count", "lift", "pmi", "npmi")


def _relabel_by_size(groups: list) -> dict:
    """Node -> id, communities numbered 0, 1, ... from the largest (ties: smallest node name first),
    so ids do not depend on the order an algorithm returns them in."""
    ordered = sorted((sorted(g, key=str) for g in groups), key=lambda g: (-len(g), str(g[0])))
    return {node: cid for cid, g in enumerate(ordered) for node in g}


def pmi_weights(pairs: pd.DataFrame) -> pd.DataFrame:
    """pairs + columns lift, pmi, npmi computed from pair_count, count_a, count_b, n_baskets:
    p(a,b) = pair_count/N, p(a) = count_a/N; lift = p(a,b)/(p(a)p(b)); pmi = log2(lift);
    npmi = pmi / -log2(p(a,b)) (in [-1, 1])."""
    out = pairs.copy()
    n = out["n_baskets"].astype(float)
    p_ab = out["pair_count"] / n
    p_a, p_b = out["count_a"] / n, out["count_b"] / n
    out["lift"] = p_ab / (p_a * p_b)
    out["pmi"] = np.log2(out["lift"])
    h_ab = -np.log2(p_ab)  # 0 only when the pair is in every basket: perfect association, npmi = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        out["npmi"] = np.where(h_ab > 0, out["pmi"] / h_ab, 1.0)
    out["npmi"] = out["npmi"].clip(-1.0, 1.0)  # remove floating-point overshoot only
    return out


def ingredient_graph(pairs: pd.DataFrame, weight: str = "npmi", min_weight: float = 0.1) -> nx.Graph:
    """Undirected graph: nodes = ingredients, edge (a, b) if its weight >= min_weight;
    weight in WEIGHTS ("count" = pair_count). Edge attribute "weight" (always positive)."""
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    df = pairs if {"lift", "pmi", "npmi"} <= set(pairs.columns) else pmi_weights(pairs)
    col = "pair_count" if weight == "count" else weight
    keep = df[col] >= min_weight
    if weight != "count":
        # CONTRACT 3.4.4: only positive associations (lift > 1, i.e. pmi > 0) become edges
        keep &= df["pmi"] > 0
    keep &= df[col] > 0
    G = nx.Graph()
    G.add_weighted_edges_from(zip(df.loc[keep, "item_a"], df.loc[keep, "item_b"], df.loc[keep, col].astype(float)))
    return G


def dominant_cuisine(recipes: pd.DataFrame, min_share: float = 0.4) -> dict:
    """Ingredient -> cuisine with the highest share among the labelled recipes that contain it,
    only if that share >= min_share (other ingredients get no label)."""
    labelled = recipes.loc[recipes["cuisine"].notna(), ["ingredients_norm", "cuisine"]]
    counts: dict[str, Counter] = defaultdict(Counter)
    for ingredients, cuisine in zip(labelled["ingredients_norm"], labelled["cuisine"]):
        for ingredient in set(ingredients):
            counts[ingredient][cuisine] += 1
    labels = {}
    for ingredient, by_cuisine in counts.items():
        total = sum(by_cuisine.values())
        # highest count; ties broken alphabetically so the result is deterministic
        cuisine, n = min(by_cuisine.items(), key=lambda kv: (-kv[1], kv[0]))
        if n / total >= min_share:
            labels[ingredient] = cuisine
    return labels


def louvain(G: nx.Graph, seed: int = 42) -> dict:
    """Node -> community id (networkx louvain_communities, weight="weight", resolution=1)."""
    communities = nx.community.louvain_communities(G, weight="weight", resolution=1, seed=seed)
    return _relabel_by_size(communities)


def spectral(G: nx.Graph, k: int, seed: int = 42) -> dict:
    """Node -> cluster id, spectral clustering on the weighted adjacency matrix."""
    from sklearn.cluster import SpectralClustering

    nodes = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr")
    A.indices, A.indptr = A.indices.astype(np.int32), A.indptr.astype(np.int32)  # sklearn wants int32
    # cluster_qr: deterministic label assignment (no k-means restarts). On a disconnected graph the
    # caller should pass the largest connected component (CONTRACT 3.4.4, bug table of the plan).
    model = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="cluster_qr",
                               random_state=seed)
    labels = model.fit_predict(A)
    groups = defaultdict(list)
    for node, label in zip(nodes, labels):
        groups[label].append(node)
    return _relabel_by_size(list(groups.values()))


def null_graph(G: nx.Graph, seed: int = 42) -> nx.Graph:
    """Degree-preserving randomised copy of G (double edge swaps, 10 swaps per edge)."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from(G.edges())
    m = H.number_of_edges()
    if m >= 2 and H.number_of_nodes() >= 4:
        n_swaps = 10 * m
        try:
            nx.double_edge_swap(H, nswap=n_swaps, max_tries=100 * n_swaps, seed=seed)
        except nx.NetworkXAlgorithmError:
            pass  # very dense graph: fewer swaps were possible; degrees are still preserved
    # swaps create edges without attributes: give the rewired edges the original weights, shuffled,
    # so the null graph has the same degrees AND the same weight distribution as G
    weights = np.array([d.get("weight", 1.0) for _, _, d in G.edges(data=True)], dtype=float)
    np.random.default_rng(seed).shuffle(weights)
    for (u, v), w in zip(sorted(H.edges(), key=lambda e: (str(e[0]), str(e[1]))), weights):
        H[u][v]["weight"] = float(w)
    return H


def evaluate_partition(G: nx.Graph, partition: dict, labels: dict | None = None) -> dict:
    """{"n_communities", "modularity", "nmi"}; nmi only on nodes that have a label, else None."""
    from sklearn.metrics import normalized_mutual_info_score

    groups = defaultdict(set)
    for node in G:
        groups[partition[node]].add(node)
    modularity = nx.community.modularity(G, list(groups.values()), weight="weight") if G.number_of_edges() else 0.0
    nmi = None
    if labels:
        common = [n for n in G if n in labels]
        if common:
            nmi = float(normalized_mutual_info_score([labels[n] for n in common], [partition[n] for n in common]))
    return {"n_communities": len(groups), "modularity": float(modularity), "nmi": nmi}


def node2vec_embeddings(G: nx.Graph, dimensions: int = 64, seed: int = 42) -> pd.DataFrame:
    """OPTIONAL (scope cut 1): Node2Vec embedding per ingredient, index = ingredient."""
    raise NotImplementedError("R4 (optional)")
