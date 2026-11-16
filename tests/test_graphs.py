"""Tests for graphs.py (owner R4, docs/CONTRACT.md 3.4.4). Toy examples with a known answer."""
import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from foodcom import graphs, schemas


def pairs_df(rows, n_baskets=100):
    """PAIRS table (schemas.PAIRS) from (item_a, item_b, pair_count, count_a, count_b) tuples."""
    df = pd.DataFrame(rows, columns=["item_a", "item_b", "pair_count", "count_a", "count_b"])
    df["n_baskets"] = n_baskets
    schemas.validate(df, "pairs")
    return df


@pytest.fixture
def two_cliques():
    """Two 5-cliques {0..4} and {5..9} joined by the single edge 4-5 (weight 1 everywhere)."""
    G = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    G.add_edge(4, 5)
    nx.set_edge_attributes(G, 1.0, "weight")
    return G


CLIQUE_A, CLIQUE_B = frozenset(range(5)), frozenset(range(5, 10))


def as_sets(partition):
    groups = {}
    for node, c in partition.items():
        groups.setdefault(c, set()).add(node)
    return {frozenset(g) for g in groups.values()}


# --- pmi_weights -----------------------------------------------------------------------------

def test_pmi_weights_hand_computed_pair():
    # N=100, count_a=20, count_b=10, pair=8: p(a,b)=0.08, p(a)p(b)=0.02 -> lift 4, pmi 2
    out = graphs.pmi_weights(pairs_df([("a", "b", 8, 20, 10)]))
    row = out.iloc[0]
    assert row["lift"] == pytest.approx(4.0)
    assert row["pmi"] == pytest.approx(2.0)
    assert row["npmi"] == pytest.approx(2.0 / math.log2(12.5))


def test_pmi_weights_keeps_input_columns_and_rows():
    pairs = pairs_df([("a", "b", 8, 20, 10), ("a", "c", 1, 20, 50)])
    out = graphs.pmi_weights(pairs)
    assert list(out.columns[: len(pairs.columns)]) == list(pairs.columns)
    assert len(out) == 2
    assert {"lift", "pmi", "npmi"} <= set(out.columns)


def test_pmi_weights_independent_and_avoided_pairs():
    # independent: p(a,b) = p(a)p(b) -> pmi 0; avoided: fewer co-occurrences than chance -> pmi < 0
    out = graphs.pmi_weights(pairs_df([("a", "b", 4, 20, 20), ("a", "c", 1, 20, 50)]))
    assert out.loc[0, "pmi"] == pytest.approx(0.0)
    assert out.loc[0, "npmi"] == pytest.approx(0.0)
    assert out.loc[1, "pmi"] < 0 and out.loc[1, "lift"] < 1


def test_npmi_is_bounded_and_one_for_items_always_together():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        a, b = rng.integers(1, 1000, size=2)
        rows.append((f"x{i}", f"y{i}", int(rng.integers(1, min(a, b) + 1)), int(a), int(b)))
    rows.append(("salt", "pepper", 30, 30, 30))  # always together
    out = graphs.pmi_weights(pairs_df(rows, n_baskets=1000))
    assert out["npmi"].between(-1 - 1e-9, 1 + 1e-9).all()
    assert out["npmi"].iloc[-1] == pytest.approx(1.0)


def test_npmi_is_one_when_pair_is_in_every_basket():
    # p(a,b) = 1 -> -log2 p(a,b) = 0: must not divide by zero
    out = graphs.pmi_weights(pairs_df([("a", "b", 100, 100, 100)]))
    assert out.loc[0, "npmi"] == pytest.approx(1.0)


# --- ingredient_graph ------------------------------------------------------------------------

@pytest.fixture
def three_pairs():
    # (a,b) lift 4 / npmi 0.55, (a,c) lift 1.25 / npmi ~0.07, (b,c) lift 0.5 (avoided)
    return pairs_df([("a", "b", 8, 20, 10), ("a", "c", 5, 20, 20), ("b", "c", 1, 10, 20)])


def test_ingredient_graph_npmi_threshold(three_pairs):
    G = graphs.ingredient_graph(three_pairs, weight="npmi", min_weight=0.1)
    assert set(G.edges()) == {("a", "b")}
    assert G["a"]["b"]["weight"] == pytest.approx(2.0 / math.log2(12.5))


def test_ingredient_graph_keeps_only_positive_associations(three_pairs):
    # with a threshold below every value, the avoided pair (b, c) must still be left out
    for weight in ("lift", "pmi", "npmi"):
        G = graphs.ingredient_graph(three_pairs, weight=weight, min_weight=-10)
        assert not G.has_edge("b", "c"), weight
        assert G.has_edge("a", "b") and G.has_edge("a", "c"), weight
        assert all(d["weight"] > 0 for _, _, d in G.edges(data=True)), weight


def test_ingredient_graph_count_weight(three_pairs):
    G = graphs.ingredient_graph(three_pairs, weight="count", min_weight=5)
    assert {frozenset(e) for e in G.edges()} == {frozenset("ab"), frozenset("ac")}
    assert G["a"]["b"]["weight"] == 8


def test_ingredient_graph_accepts_pairs_with_weights_already_computed(three_pairs):
    G1 = graphs.ingredient_graph(three_pairs, weight="pmi", min_weight=0)
    G2 = graphs.ingredient_graph(graphs.pmi_weights(three_pairs), weight="pmi", min_weight=0)
    assert nx.utils.edges_equal(G1.edges(data=True), G2.edges(data=True))


def test_ingredient_graph_rejects_unknown_weight(three_pairs):
    with pytest.raises(ValueError):
        graphs.ingredient_graph(three_pairs, weight="jaccard")


# --- dominant_cuisine ------------------------------------------------------------------------

def test_dominant_cuisine_on_mini_data(mini_recipes):
    labels = graphs.dominant_cuisine(mini_recipes, min_share=0.4)
    assert labels["coconut milk"] == "thai"
    assert labels["feta cheese"] == "greek"
    # "olive oil": one mexican and one greek recipe -> share 0.5, tie broken alphabetically
    assert labels["olive oil"] == "greek"
    # "sugar" only appears in recipes without a cuisine tag -> no label
    assert "sugar" not in labels


def test_dominant_cuisine_min_share(mini_recipes):
    labels = graphs.dominant_cuisine(mini_recipes, min_share=0.6)
    assert "olive oil" not in labels  # share 0.5 < 0.6
    assert labels["coconut milk"] == "thai"


# --- louvain / spectral ----------------------------------------------------------------------

def test_louvain_finds_the_two_cliques(two_cliques):
    part = graphs.louvain(two_cliques, seed=42)
    assert set(part) == set(two_cliques.nodes())
    assert as_sets(part) == {CLIQUE_A, CLIQUE_B}
    assert graphs.evaluate_partition(two_cliques, part)["modularity"] > 0.4


def test_louvain_is_reproducible_with_a_seed():
    G = nx.karate_club_graph()
    assert graphs.louvain(G, seed=7) == graphs.louvain(G, seed=7)


def test_spectral_k2_gives_the_same_split(two_cliques):
    part = graphs.spectral(two_cliques, k=2, seed=42)
    assert set(part) == set(two_cliques.nodes())
    assert as_sets(part) == {CLIQUE_A, CLIQUE_B}


def test_spectral_uses_the_weights():
    # a 4-cycle a-b-c-d: heavy edges a-b and c-d must end up inside the clusters
    G = nx.Graph()
    G.add_weighted_edges_from([("a", "b", 10), ("b", "c", 0.1), ("c", "d", 10), ("d", "a", 0.1)])
    assert as_sets(graphs.spectral(G, k=2, seed=0)) == {frozenset("ab"), frozenset("cd")}


# --- null_graph ------------------------------------------------------------------------------

@pytest.fixture
def random_weighted_graph():
    G = nx.gnm_random_graph(60, 200, seed=1)
    rng = np.random.default_rng(1)
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.uniform(0.1, 1.0))
    return G


def test_null_graph_keeps_every_degree_and_the_weights(random_weighted_graph):
    G = random_weighted_graph
    H = graphs.null_graph(G, seed=42)
    assert dict(H.degree()) == dict(G.degree())
    assert H.number_of_edges() == G.number_of_edges()
    w_g = sorted(d["weight"] for _, _, d in G.edges(data=True))
    w_h = sorted(d["weight"] for _, _, d in H.edges(data=True))
    assert w_h == pytest.approx(w_g)


def test_null_graph_rewires_and_leaves_the_input_untouched(random_weighted_graph):
    G = random_weighted_graph
    before = set(map(frozenset, G.edges()))
    H = graphs.null_graph(G, seed=42)
    assert set(map(frozenset, G.edges())) == before
    shared = len(before & set(map(frozenset, H.edges())))
    assert shared < 0.5 * len(before)  # 10 swaps per edge: most edges have moved


def test_null_graph_is_reproducible(random_weighted_graph):
    H1 = graphs.null_graph(random_weighted_graph, seed=3)
    H2 = graphs.null_graph(random_weighted_graph, seed=3)
    assert nx.utils.edges_equal(H1.edges(data=True), H2.edges(data=True))


def test_null_graph_destroys_the_community_structure():
    G = nx.disjoint_union_all([nx.complete_graph(8) for _ in range(4)])
    for i in range(3):
        G.add_edge(8 * i, 8 * (i + 1))
    nx.set_edge_attributes(G, 1.0, "weight")
    q = graphs.evaluate_partition(G, graphs.louvain(G))["modularity"]
    H = graphs.null_graph(G, seed=0)
    q_null = graphs.evaluate_partition(H, graphs.louvain(H))["modularity"]
    assert q > 0.6 and q_null < q - 0.2


def test_null_graph_on_a_tiny_graph_does_not_fail():
    G = nx.Graph()
    G.add_edge("a", "b", weight=1.0)
    H = graphs.null_graph(G)
    assert set(H.edges()) == {("a", "b")}


# --- evaluate_partition ----------------------------------------------------------------------

def test_evaluate_partition_perfect_labelling(two_cliques):
    part = {n: (0 if n < 5 else 1) for n in two_cliques}
    labels = {n: ("italian" if n < 5 else "thai") for n in two_cliques}
    res = graphs.evaluate_partition(two_cliques, part, labels)
    assert res["n_communities"] == 2
    assert res["nmi"] == pytest.approx(1.0)
    assert res["modularity"] == pytest.approx(nx.community.modularity(two_cliques, [CLIQUE_A, CLIQUE_B]))


def test_evaluate_partition_nmi_only_on_labelled_nodes(two_cliques):
    part = {n: (0 if n < 5 else 1) for n in two_cliques}
    labels = {0: "italian", 1: "italian", 7: "thai", "not-in-graph": "greek"}
    assert graphs.evaluate_partition(two_cliques, part, labels)["nmi"] == pytest.approx(1.0)


def test_evaluate_partition_without_labels(two_cliques):
    part = {n: 0 for n in two_cliques}
    res = graphs.evaluate_partition(two_cliques, part)
    assert res["n_communities"] == 1 and res["nmi"] is None
    assert res["modularity"] == pytest.approx(0.0)
    assert graphs.evaluate_partition(two_cliques, part, {"x": "thai"})["nmi"] is None


def test_evaluate_partition_random_labels_give_low_nmi():
    G = nx.karate_club_graph()
    part = graphs.louvain(G)
    rng = np.random.default_rng(0)
    labels = {n: str(rng.integers(0, 4)) for n in G}
    assert graphs.evaluate_partition(G, part, labels)["nmi"] < 0.3
