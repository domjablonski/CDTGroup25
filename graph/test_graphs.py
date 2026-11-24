"""Tests for graph/graphs.py (R4). Toy examples with a known answer. Run: pytest graph -q"""
import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import graphs


def pairs_df(rows, n_baskets=100):
    """Pair table from (item_a, item_b, pair_count, count_a, count_b) tuples."""
    df = pd.DataFrame(rows, columns=graphs.PAIR_COLUMNS[:-1])
    df["n_baskets"] = n_baskets
    return df


@pytest.fixture
def mini_recipes():
    """8 recipes like the group's recipes.parquet (ingredients_v2 already normalised)."""
    return pd.DataFrame({
        "id": range(101, 109),
        "tags": [["mexican", "easy"], ["italian", "easy"], ["italian"], ["desserts"], ["desserts"],
                 ["thai", "asian", "main-dish"], ["beverages"], ["greek", "healthy"]],
        "ingredients_v2": [["winter squash", "mexican seasoning", "olive oil", "salt"],
                           ["butter", "garlic clove", "bread"], ["butter", "garlic", "bread"],
                           ["butter", "sugar", "egg", "vanilla extract", "flour", "baking soda"],
                           ["butter", "sugar", "egg", "flour", "baking powder"],
                           ["green curry paste", "coconut milk", "chicken", "fish sauce"],
                           ["fruit juice", "soda", "sugar"],
                           ["tomato", "cucumber", "feta cheese", "olive oil", "salt"]],
    }).assign(cuisine=lambda d: d["tags"].map(graphs.cuisine_label))


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


def cliques_chain(n_cliques, size=8):
    G = nx.disjoint_union_all([nx.complete_graph(size) for _ in range(n_cliques)])
    for i in range(n_cliques - 1):
        G.add_edge(size * i, size * (i + 1))
    nx.set_edge_attributes(G, 1.0, "weight")
    return G


# --- data -------------------------------------------------------------------------------------

def test_cuisine_label_takes_the_most_specific_tag():
    assert graphs.cuisine_label(["asian", "thai", "easy"]) == "thai"
    assert graphs.cuisine_label(["desserts"]) is None


def test_load_recipes_turns_arrays_into_lists_and_adds_cuisine(tmp_path, mini_recipes):
    path = tmp_path / "recipes.parquet"
    mini_recipes.drop(columns="cuisine").to_parquet(path)
    df = graphs.load_recipes(path)
    assert isinstance(df["ingredients_v2"].iloc[0], list)
    assert df.loc[df["id"] == 106, "cuisine"].item() == "thai"


# --- pairs ------------------------------------------------------------------------------------

BASKETS = [{"a", "b", "c"}, {"a", "b"}, {"a", "c"}, {"a", "b", "d"}, {"b", "c"}, {"a", "b", "c"},
           {"d"}, {"a", "e"}]


def test_count_pairs_known_counts():
    # N = 8, threshold ceil(0.25 * 8) = 2: items a6 b5 c4 d2 (e1 out); pairs ab4 ac3 bc3 (ad1, bd1 out)
    out = graphs.count_pairs(BASKETS, min_support=0.25)
    got = {(r.item_a, r.item_b): (r.pair_count, r.count_a, r.count_b) for r in out.itertuples()}
    assert got == {("a", "b"): (4, 6, 5), ("a", "c"): (3, 6, 4), ("b", "c"): (3, 5, 4)}
    assert (out["n_baskets"] == 8).all()
    assert list(out.columns) == graphs.PAIR_COLUMNS


def test_pairs_from_pcy_matches_the_exact_count():
    singles = pd.DataFrame({"ingredient": ["a", "b", "c", "d"], "support": [6, 5, 4, 2]})
    pcy_pairs = pd.DataFrame({"ingredient1": ["b", "a", "c"], "ingredient2": ["a", "c", "b"], "support": [4, 3, 3]})
    out = graphs.pairs_from_pcy(singles, pcy_pairs, n_baskets=8)
    pd.testing.assert_frame_equal(out, graphs.count_pairs(BASKETS, min_support=0.25), check_dtype=False)


def test_read_pcy_dir_prefers_the_full_pair_table(tmp_path):
    # pairs.parquet (export_pcy_pairs.py) carries the number of baskets after duplicate removal: 7, not 8
    exact = graphs.count_pairs(BASKETS, min_support=0.25).assign(n_baskets=7)
    exact.astype({"item_a": "string", "item_b": "string"}).to_parquet(tmp_path / "pairs.parquet")
    pd.DataFrame({"ingredient": ["a"], "support": [6]}).to_parquet(tmp_path / "frequent_ingredients_singles.parquet")
    out = graphs.read_pcy_dir(tmp_path, n_recipes=8)
    assert (out["n_baskets"] == 7).all()
    pd.testing.assert_frame_equal(out, exact, check_dtype=False)


def test_read_pcy_dir_falls_back_to_the_singles_and_pairs_files(tmp_path):
    pd.DataFrame({"ingredient": ["a", "b", "c", "d"], "support": [6, 5, 4, 2]}).to_parquet(
        tmp_path / "frequent_ingredients_singles.parquet")
    pd.DataFrame({"ingredient1": ["b", "a", "c"], "ingredient2": ["a", "c", "b"], "support": [4, 3, 3]}).to_parquet(
        tmp_path / "frequent_ingredients_pairs.parquet")
    pd.testing.assert_frame_equal(graphs.read_pcy_dir(tmp_path, n_recipes=8),
                                  graphs.count_pairs(BASKETS, min_support=0.25), check_dtype=False)


# --- pmi_weights ------------------------------------------------------------------------------

def test_pmi_weights_hand_computed_pair():
    # N=100, count_a=20, count_b=10, pair=8: p(a,b)=0.08, p(a)p(b)=0.02 -> lift 4, pmi 2
    row = graphs.pmi_weights(pairs_df([("a", "b", 8, 20, 10)])).iloc[0]
    assert row["lift"] == pytest.approx(4.0)
    assert row["pmi"] == pytest.approx(2.0)
    assert row["npmi"] == pytest.approx(2.0 / math.log2(12.5))


def test_pmi_weights_independent_and_avoided_pairs():
    out = graphs.pmi_weights(pairs_df([("a", "b", 4, 20, 20), ("a", "c", 1, 20, 50)]))
    assert out.loc[0, "pmi"] == pytest.approx(0.0) and out.loc[0, "npmi"] == pytest.approx(0.0)
    assert out.loc[1, "pmi"] < 0 and out.loc[1, "lift"] < 1


def test_npmi_is_bounded_and_one_for_items_always_together():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        a, b = rng.integers(1, 1000, size=2)
        rows.append((f"x{i}", f"y{i}", int(rng.integers(1, min(a, b) + 1)), int(a), int(b)))
    rows.append(("salt", "pepper", 30, 30, 30))
    out = graphs.pmi_weights(pairs_df(rows, n_baskets=1000))
    assert out["npmi"].between(-1, 1).all()
    assert out["npmi"].iloc[-1] == pytest.approx(1.0)


def test_npmi_is_one_when_pair_is_in_every_basket():
    assert graphs.pmi_weights(pairs_df([("a", "b", 100, 100, 100)])).loc[0, "npmi"] == pytest.approx(1.0)


# --- ingredient_graph -------------------------------------------------------------------------

@pytest.fixture
def three_pairs():
    # (a,b) lift 4 / npmi 0.55, (a,c) lift 1.25 / npmi ~0.07, (b,c) lift 0.5 (avoided)
    return pairs_df([("a", "b", 8, 20, 10), ("a", "c", 5, 20, 20), ("b", "c", 1, 10, 20)])


def test_ingredient_graph_npmi_threshold(three_pairs):
    G = graphs.ingredient_graph(three_pairs, weight="npmi", min_weight=0.1)
    assert set(G.edges()) == {("a", "b")}
    assert G["a"]["b"]["weight"] == pytest.approx(2.0 / math.log2(12.5))


def test_ingredient_graph_keeps_only_positive_associations(three_pairs):
    for weight in ("lift", "pmi", "npmi"):
        G = graphs.ingredient_graph(three_pairs, weight=weight, min_weight=-10)
        assert not G.has_edge("b", "c"), weight
        assert G.has_edge("a", "b") and G.has_edge("a", "c"), weight
        assert all(d["weight"] > 0 for _, _, d in G.edges(data=True)), weight


def test_ingredient_graph_count_weight(three_pairs):
    G = graphs.ingredient_graph(three_pairs, weight="count", min_weight=5)
    assert {frozenset(e) for e in G.edges()} == {frozenset("ab"), frozenset("ac")}
    assert G["a"]["b"]["weight"] == 8


def test_ingredient_graph_rejects_unknown_weight(three_pairs):
    with pytest.raises(ValueError):
        graphs.ingredient_graph(three_pairs, weight="jaccard")


# --- thresholds -------------------------------------------------------------------------------

def test_threshold_sweep_and_choice(three_pairs):
    sweep = graphs.threshold_sweep(three_pairs, "npmi", thresholds=[0.0, 0.05, 0.1, 0.6])
    assert sweep["edges"].tolist() == [2, 2, 1, 0]
    assert sweep["lcc_share"].tolist() == pytest.approx([1.0, 1.0, 2 / 3, 0.0])
    assert graphs.choose_threshold(sweep, min_lcc_share=0.9) == 0.05
    assert graphs.choose_threshold(sweep, min_lcc_share=0.5) == 0.1


# --- dominant_cuisine -------------------------------------------------------------------------

def test_dominant_cuisine_on_mini_data(mini_recipes):
    labels = graphs.dominant_cuisine(mini_recipes, min_share=0.4, min_recipes=1)
    assert labels["coconut milk"] == "thai"
    assert labels["feta cheese"] == "greek"
    assert labels["olive oil"] == "greek"  # one mexican + one greek recipe: share 0.5, tie -> alphabetical
    assert "sugar" not in labels           # only in recipes without a cuisine tag


def test_dominant_cuisine_min_share_and_min_recipes(mini_recipes):
    assert "olive oil" not in graphs.dominant_cuisine(mini_recipes, min_share=0.6, min_recipes=1)
    labels = graphs.dominant_cuisine(mini_recipes, min_share=0.4, min_recipes=2)
    assert "olive oil" in labels and "coconut milk" not in labels  # coconut milk: 1 labelled recipe


# --- louvain / spectral / eigengap ------------------------------------------------------------

def test_louvain_finds_the_two_cliques(two_cliques):
    part = graphs.louvain(two_cliques)
    assert as_sets(part) == {CLIQUE_A, CLIQUE_B}
    assert graphs.evaluate_partition(two_cliques, part)["modularity"] > 0.4


def test_louvain_is_reproducible_with_a_seed():
    G = nx.karate_club_graph()
    assert graphs.louvain(G, seed=7) == graphs.louvain(G, seed=7)


def test_spectral_k2_gives_the_same_split(two_cliques):
    assert as_sets(graphs.spectral(two_cliques, k=2)) == {CLIQUE_A, CLIQUE_B}


def test_spectral_uses_the_weights():
    G = nx.Graph()
    G.add_weighted_edges_from([("a", "b", 10), ("b", "c", 0.1), ("c", "d", 10), ("d", "a", 0.1)])
    assert as_sets(graphs.spectral(G, k=2, seed=0)) == {frozenset("ab"), frozenset("cd")}


@pytest.mark.parametrize("n_cliques", [2, 3, 4])
def test_eigengap_finds_the_number_of_cliques(n_cliques):
    assert graphs.eigengap_k(cliques_chain(n_cliques)) == n_cliques


# --- null graphs ------------------------------------------------------------------------------

@pytest.fixture
def random_weighted_graph():
    G = nx.gnm_random_graph(60, 200, seed=1)
    rng = np.random.default_rng(1)
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.uniform(0.1, 1.0))
    return G


@pytest.mark.parametrize("mode", ["rewire", "shuffle"])
def test_null_graph_keeps_degrees_and_weights(random_weighted_graph, mode):
    G = random_weighted_graph
    H = graphs.null_graph(G, mode=mode)
    assert dict(H.degree()) == dict(G.degree())
    w = sorted(d["weight"] for _, _, d in G.edges(data=True))
    assert sorted(d["weight"] for _, _, d in H.edges(data=True)) == pytest.approx(w)


def test_null_graph_rewire_moves_edges_shuffle_keeps_them(random_weighted_graph):
    G = random_weighted_graph
    before = set(map(frozenset, G.edges()))
    rewired = set(map(frozenset, graphs.null_graph(G, mode="rewire").edges()))
    assert len(before & rewired) < 0.5 * len(before)
    assert set(map(frozenset, graphs.null_graph(G, mode="shuffle").edges())) == before
    assert set(map(frozenset, G.edges())) == before  # input untouched


def test_null_graph_is_reproducible(random_weighted_graph):
    H1, H2 = graphs.null_graph(random_weighted_graph, seed=3), graphs.null_graph(random_weighted_graph, seed=3)
    assert nx.utils.edges_equal(H1.edges(data=True), H2.edges(data=True))


def test_rewired_null_destroys_the_community_structure():
    G = cliques_chain(4)
    q = graphs.evaluate_partition(G, graphs.louvain(G))["modularity"]
    H = graphs.null_graph(G, seed=0)
    assert q > 0.6 and graphs.evaluate_partition(H, graphs.louvain(H))["modularity"] < q - 0.2


def test_shuffled_weights_hide_communities_carried_by_the_weights():
    # complete graph of 10 nodes: heavy edges inside {0..4} and {5..9}, light edges between them.
    # The topology alone has no structure; the weights have all of it.
    G = nx.complete_graph(10)
    for u, v in G.edges():
        G[u][v]["weight"] = 10.0 if (u < 5) == (v < 5) else 0.1
    assert as_sets(graphs.louvain(G)) == {CLIQUE_A, CLIQUE_B}
    H = graphs.null_graph(G, mode="shuffle", seed=0)
    q, q_null = (graphs.evaluate_partition(X, graphs.louvain(X))["modularity"] for X in (G, H))
    assert q > 0.4 and q_null < q - 0.2


def test_null_graph_on_a_tiny_graph_does_not_fail():
    G = nx.Graph()
    G.add_edge("a", "b", weight=1.0)
    assert set(graphs.null_graph(G).edges()) == {("a", "b")}


# --- evaluate_partition -----------------------------------------------------------------------

def test_evaluate_partition_perfect_labelling(two_cliques):
    part = {n: (0 if n < 5 else 1) for n in two_cliques}
    labels = {n: ("italian" if n < 5 else "thai") for n in two_cliques}
    res = graphs.evaluate_partition(two_cliques, part, labels)
    assert res["n_communities"] == 2 and res["n_labelled"] == 10
    assert res["nmi"] == pytest.approx(1.0) and res["ami"] == pytest.approx(1.0)
    assert res["modularity"] == pytest.approx(nx.community.modularity(two_cliques, [CLIQUE_A, CLIQUE_B]))


def test_evaluate_partition_only_on_labelled_nodes(two_cliques):
    part = {n: (0 if n < 5 else 1) for n in two_cliques}
    res = graphs.evaluate_partition(two_cliques, part, {0: "italian", 1: "italian", 7: "thai", "x": "greek"})
    assert res["nmi"] == pytest.approx(1.0) and res["n_labelled"] == 3


def test_evaluate_partition_without_labels(two_cliques):
    part = {n: 0 for n in two_cliques}
    res = graphs.evaluate_partition(two_cliques, part)
    assert res["n_communities"] == 1 and res["nmi"] is None and res["ami"] is None
    assert res["modularity"] == pytest.approx(0.0)


def test_ami_is_near_zero_for_random_labels_while_nmi_is_not():
    # many small communities inflate NMI by chance; AMI corrects for it
    rng = np.random.default_rng(0)
    G = nx.empty_graph(300)
    part = {n: int(rng.integers(0, 60)) for n in G}
    labels = {n: str(rng.integers(0, 5)) for n in G}
    res = graphs.evaluate_partition(G, part, labels)
    assert abs(res["ami"]) < 0.05 and res["nmi"] > res["ami"] + 0.05
