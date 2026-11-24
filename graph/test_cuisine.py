"""Tests for graph/cuisine.py: predicting a recipe's cuisine from ingredient communities (R4).
Toy examples with a known answer. Run: pytest graph -q"""
import numpy as np
import pandas as pd
import pytest

import cuisine
import graphs


# --- community profiles -----------------------------------------------------------------------

PARTITION = {"basil": 0, "parmesan": 0, "olive oil": 0, "soy sauce": 1, "ginger": 1, "rice": 1}


def test_profile_is_the_share_of_ingredients_in_each_community():
    X = cuisine.community_profiles([["basil", "parmesan", "ginger", "unknown"]], PARTITION, n_communities=2)
    # 3 known ingredients: 2 in community 0, 1 in community 1; "unknown" (not in the graph) is ignored
    assert X.shape == (1, 2)
    np.testing.assert_allclose(X[0], [2 / 3, 1 / 3])


def test_profile_of_a_recipe_without_known_ingredient_is_zero():
    X = cuisine.community_profiles([["salt", "water"], ["rice"]], PARTITION, n_communities=2)
    np.testing.assert_allclose(X, [[0, 0], [0, 1]])


def test_profile_counts_each_ingredient_once():
    X = cuisine.community_profiles([["basil", "basil", "rice"]], PARTITION, n_communities=2)
    np.testing.assert_allclose(X[0], [0.5, 0.5])


# --- community -> cuisine table and prediction without training -------------------------------

def test_community_cuisine_table_rows_are_distributions():
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    y = np.array(["italian", "italian", "thai", "thai"])
    table = cuisine.community_cuisine_table(X, y, classes=["italian", "thai"], alpha=0.0)
    # community 0: italian 2, thai 0.5 -> 0.8 / 0.2 ; community 1: thai 1.5 -> 0 / 1
    np.testing.assert_allclose(table, [[0.8, 0.2], [0.0, 1.0]])


def test_community_cuisine_table_smoothing_keeps_unseen_communities_uniform():
    X = np.array([[1.0, 0.0, 0.0]])
    table = cuisine.community_cuisine_table(X, np.array(["thai"]), classes=["italian", "thai"], alpha=1.0)
    np.testing.assert_allclose(table[0], [1 / 3, 2 / 3])
    np.testing.assert_allclose(table[2], [0.5, 0.5])  # community never seen in training


def test_predict_without_training_picks_the_cuisine_of_the_communities():
    table = np.array([[0.9, 0.1], [0.2, 0.8]])
    X = np.array([[1.0, 0.0], [0.3, 0.7], [0.0, 0.0]])
    pred = cuisine.predict_from_table(X, table, classes=["italian", "thai"], prior=None, fallback="italian")
    assert list(pred) == ["italian", "thai", "italian"]  # empty profile -> fallback


def test_dividing_by_the_prior_favours_the_rarer_cuisine():
    # community 0 is 60 % italian, but italian is 80 % of all recipes: thai is over-represented
    table = np.array([[0.6, 0.4]])
    X = np.array([[1.0]])
    classes = ["italian", "thai"]
    assert cuisine.predict_from_table(X, table, classes, prior=None)[0] == "italian"
    assert cuisine.predict_from_table(X, table, classes, prior=np.array([0.8, 0.2]))[0] == "thai"


# --- evaluation -------------------------------------------------------------------------------

def test_evaluate_accuracy_and_macro_f1():
    y_true = ["a", "a", "a", "b"]
    y_pred = ["a", "a", "a", "a"]
    res = cuisine.evaluate(y_true, y_pred, classes=["a", "b"])
    assert res["accuracy"] == pytest.approx(0.75)
    assert res["macro_f1"] == pytest.approx((6 / 7 + 0) / 2)  # F1(a) = 2*0.75*1/1.75, F1(b) = 0


# --- end to end on a toy world with two cuisines ----------------------------------------------

def toy_world(seed=0, n=600):
    """Italian recipes draw from an italian pool, thai recipes from a thai pool, plus shared
    staples (salt, water) and 10 % noise ingredients from the other pool."""
    rng = np.random.default_rng(seed)
    pools = {"italian": [f"it{i}" for i in range(12)], "thai": [f"th{i}" for i in range(12)]}
    rows = []
    for i in range(n):
        c = "italian" if i % 2 else "thai"
        other = "thai" if c == "italian" else "italian"
        ings = list(rng.choice(pools[c], 5, replace=False)) + ["salt", "water"]
        if rng.random() < 0.1:
            ings.append(str(rng.choice(pools[other])))
        rows.append({"id": i, "ingredients_v2": ings, "cuisine": c})
    return pd.DataFrame(rows)


def test_end_to_end_communities_recover_the_two_cuisines():
    df = toy_world()
    train, test = df.iloc[:400], df.iloc[400:]
    G = cuisine.build_graph(train["ingredients_v2"], weight="pmi", min_support=0.02)
    partition = graphs.louvain(G, seed=0)
    k = max(partition.values()) + 1
    X_train = cuisine.community_profiles(train["ingredients_v2"], partition, k)
    X_test = cuisine.community_profiles(test["ingredients_v2"], partition, k)
    classes = ["italian", "thai"]
    table = cuisine.community_cuisine_table(X_train, train["cuisine"].to_numpy(), classes)
    pred = cuisine.predict_from_table(X_test, table, classes)
    assert cuisine.evaluate(test["cuisine"], pred, classes)["accuracy"] > 0.95


def test_build_graph_only_keeps_positive_pmi_edges_between_frequent_ingredients():
    df = toy_world()
    G = cuisine.build_graph(df["ingredients_v2"], weight="pmi", min_support=0.02)
    assert "salt" not in G  # in every recipe: pmi 0 with everything, no positive edge
    assert all(d["weight"] > 0 for _, _, d in G.edges(data=True))
    italian_thai = [(u, v) for u, v in G.edges() if {u[:2], v[:2]} == {"it", "th"}]
    assert len(italian_thai) < 0.1 * G.number_of_edges()


# --- resolution search (run_cuisine.py) ---------------------------------------------------------

def chain_of_cliques():
    import networkx as nx

    G = nx.disjoint_union_all([nx.complete_graph(6) for _ in range(8)])
    for i in range(7):
        G.add_edge(6 * i, 6 * (i + 1))
    nx.set_edge_attributes(G, 1.0, "weight")
    return G


@pytest.mark.parametrize("k_target", [2, 3, 5, 8])
def test_resolution_for_k_hits_a_reachable_number_of_communities(k_target):
    # on this graph Louvain can give 1, 2, 3, 5, 8, 14, 16 or 48 communities
    from run_cuisine import resolution_for_k

    assert resolution_for_k(chain_of_cliques(), k_target)[1] == k_target


def test_resolution_for_k_returns_the_closest_when_unreachable():
    from run_cuisine import resolution_for_k

    assert resolution_for_k(chain_of_cliques(), 4)[1] in (3, 5)  # 4 is not reachable
