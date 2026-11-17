
"""
kNN calorie-level classifier on deduped.parquet (RQ4), step by step.

Idea: to predict whether a recipe is low, medium or high in calories, find the
training recipes with the most similar ingredient lists and let them vote.
Similar recipes are found with our own MinHash + LSH, so we never compare all pairs.

Steps
  1. Load deduped.parquet (id, name, ingredients_v2, calories)
  2. Calorie level = tertiles of calories (low / medium / high)
  3. Train / test split (recipes with the same ingredient set stay together)
  4. MinHash signature for every recipe (own implementation)
  5. LSH index of the TRAINING recipes only (32 bands x 4 rows)
  6. For each test recipe: LSH candidates -> exact Jaccard -> top 20 neighbours
  7. kNN vote for several k (plain and similarity-weighted)
  8. Baselines: majority class and logistic regression
  9. Check: exact kNN (no LSH) on 2,000 test recipes
 10. Save the results


"""
import ast
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

# ------------------------------------------------------------------ settings
HERE        = Path(__file__).resolve().parent
DATA_FILE   = HERE / "deduped.parquet"   # output of minhash.py
RESULTS_DIR = HERE / "results"
SEED        = 42
TEST_SIZE   = 0.2        # 20 % of the recipes are used for testing
N_HASH      = 128        # signature length
BANDS, ROWS = 32, 4      # 32 x 4 = 128; finds most pairs with Jaccard >= 0.5
TOP_N       = 20         # neighbours kept per test recipe
K_VALUES    = [1, 3, 5, 10, 15, 20]
MAX_KCAL    = 5000       # recipes above this are treated as errors and left out
LEVELS      = ["low", "medium", "high"]

RESULTS_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)
start = time.time()


def step(msg):
    print(f"\n[{time.time() - start:5.0f} s] {msg}")


# ---------------------------------------------------------------- 1. load data
step("1. Loading recipes")
if DATA_FILE.suffix == ".csv":                       # CSV copy also works
    df = pd.read_csv(DATA_FILE)
    df["ingredients_v2"] = df["ingredients_v2"].map(ast.literal_eval)
else:
    df = pd.read_parquet(DATA_FILE, columns=["id", "name", "ingredients_v2", "calories"])

# parquet gives NumPy arrays -> turn each ingredient list into a Python set
df["ingr"] = df["ingredients_v2"].map(lambda lst: {str(x).strip().lower() for x in lst if str(x).strip()})
df = df[df["ingr"].map(len) > 0]
print(f"   {len(df):,} recipes loaded")

# ---------------------------------------------------- 2. calorie level (target)
step("2. Calorie levels (tertiles)")
df = df[df["calories"] <= MAX_KCAL].reset_index(drop=True)
low_cut, high_cut = df["calories"].quantile([1 / 3, 2 / 3])
df["level"] = np.where(df["calories"] <= low_cut, 0, np.where(df["calories"] <= high_cut, 1, 2))
print(f"   low <= {low_cut:.0f} kcal < medium <= {high_cut:.0f} kcal < high   ({len(df):,} recipes)")
print("   class shares:", df["level"].value_counts(normalize=True).sort_index().round(3).tolist())

# ------------------------------------------------------- 3. train / test split
step("3. Train / test split")
# Recipes with exactly the same ingredient set form one group, and a group is never split:
# otherwise a copy in the training set would simply give away the answer for its twin in the test set.
group = df["ingr"].map(lambda s: "|".join(sorted(s)))     # same text = same ingredient set
splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, test_idx = next(splitter.split(df, groups=group))
print(f"   train {len(train_idx):,}  |  test {len(test_idx):,}")

y = df["level"].to_numpy()
sets = df["ingr"].tolist()

# --------------------------------------------------------------- 4. MinHash
step("4. MinHash signatures (own implementation)")
# Each hash function is h(x) = (a*x + b) mod p. x is the CRC32 code of the ingredient name, which is
# the same on every computer (Python's own hash() is not). A recipe's signature is the minimum of
# h over its ingredients, for each of the 128 functions.
P = (1 << 31) - 1
a = rng.integers(1, P, N_HASH, dtype=np.int64)
b = rng.integers(0, P, N_HASH, dtype=np.int64)

vocab = sorted({ing for s in sets for ing in s})
code = np.array([zlib.crc32(ing.encode()) % P for ing in vocab], dtype=np.int64)
H = (code[:, None] * a + b) % P                      # hash value of every ingredient, every function
pos = {ing: i for i, ing in enumerate(vocab)}

sig = np.empty((len(sets), N_HASH), dtype=np.int64)
for i, s in enumerate(sets):
    sig[i] = H[[pos[ing] for ing in s]].min(axis=0)
print(f"   {len(vocab):,} different ingredients, signatures shape {sig.shape}")

# ------------------------------------------------------------ 5. LSH index
step("5. LSH index of the training recipes")
# Cut each signature into 32 bands of 4 numbers. Two recipes with an identical band land in
# the same bucket and become candidates. Only TRAINING recipes go into the index, so a test
# recipe can only ever be compared with (and voted by) training recipes.
buckets = [defaultdict(list) for _ in range(BANDS)]
for i in train_idx:
    for band in range(BANDS):
        key = sig[i, band * ROWS:(band + 1) * ROWS].tobytes()
        buckets[band][key].append(i)


def jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2)


def lsh_neighbours(i, top_n=TOP_N):
    """Top-n training recipes most similar to recipe i: LSH candidates, then exact Jaccard."""
    cand = set()
    for band in range(BANDS):
        cand.update(buckets[band].get(sig[i, band * ROWS:(band + 1) * ROWS].tobytes(), ()))
    scored = sorted(((jaccard(sets[i], sets[j]), j) for j in cand), reverse=True)
    return scored[:top_n]                            # list of (similarity, training row)


# ------------------------------------------- 6. neighbours of every test recipe
step("6. Finding neighbours of the test recipes")
neigh = {i: lsh_neighbours(i) for i in test_idx}
covered = np.array([len(neigh[i]) > 0 for i in test_idx])
print(f"   {covered.mean():.1%} of test recipes have at least one neighbour")

# ------------------------------------------------------------- 7. kNN vote
step("7. kNN vote")
majority = Counter(y[train_idx]).most_common(1)[0][0]


def knn_vote(nbrs, k, weighted):
    """Each of the k nearest neighbours votes for its level (weight 1, or weight = similarity)."""
    if not nbrs:
        return majority                              # no neighbour: fall back to the majority class
    score = np.zeros(3)
    for sim, j in nbrs[:k]:
        score[y[j]] += sim if weighted else 1.0
    return int(score.argmax())


y_test = y[test_idx]
rows = []
for weighted in (False, True):
    for k in K_VALUES:
        pred = np.array([knn_vote(neigh[i], k, weighted) for i in test_idx])
        acc = accuracy_score(y_test, pred)
        rows.append(dict(model="knn_lsh", k=k, weighted=weighted, accuracy=acc,
                         accuracy_covered=accuracy_score(y_test[covered], pred[covered]),
                         coverage=covered.mean()))
        print(f"   k={k:2d}  weighted={str(weighted):5}  accuracy={acc:.3f}")

# ------------------------------------------------------------- 8. baselines
step("8. Baselines")
acc_major = accuracy_score(y_test, np.full(len(y_test), majority))
rows.append(dict(model="majority_class", accuracy=acc_major))
print(f"   majority class      accuracy={acc_major:.3f}")

# Logistic regression on a 0/1 table: one column per ingredient, 1 if the recipe contains it.
X = sp.csr_matrix((np.ones(sum(map(len, sets)), dtype=np.float32),
                   [pos[ing] for s in sets for ing in s],
                   np.concatenate([[0], np.cumsum([len(s) for s in sets])])),
                  shape=(len(sets), len(vocab)))
logreg = LogisticRegression(max_iter=1000, random_state=SEED).fit(X[train_idx], y[train_idx])
acc_lr = accuracy_score(y_test, logreg.predict(X[test_idx]))
rows.append(dict(model="logistic_regression", accuracy=acc_lr))
print(f"   logistic regression accuracy={acc_lr:.3f}")

# ------------------------------------------- 9. check: exact kNN without LSH
step("9. Check: exact kNN (all training recipes) on 2,000 test recipes")
best = max((r for r in rows if r["model"] == "knn_lsh"), key=lambda r: r["accuracy"])
k_best, w_best = best["k"], best["weighted"]
sample = rng.choice(test_idx, 2000, replace=False)
size = np.asarray(X.sum(axis=1)).ravel()
X_train = X[train_idx]
exact_pred, lsh_pred = [], []
for chunk in np.array_split(sample, 10):
    inter = (X[chunk] @ X_train.T).toarray()                   # shared ingredients
    jac = inter / (size[chunk][:, None] + size[train_idx][None, :] - inter)
    for row, i in zip(jac, chunk):
        top = np.argsort(-row)[:TOP_N]
        exact_pred.append(knn_vote([(row[t], train_idx[t]) for t in top if row[t] > 0], k_best, w_best))
        lsh_pred.append(knn_vote(neigh[i], k_best, w_best))
acc_exact = accuracy_score(y[sample], exact_pred)
acc_lsh_same = accuracy_score(y[sample], lsh_pred)
rows.append(dict(model="knn_exact_2000", k=k_best, weighted=w_best, accuracy=acc_exact))
rows.append(dict(model="knn_lsh_same_2000", k=k_best, weighted=w_best, accuracy=acc_lsh_same))
print(f"   same 2,000 recipes:  exact kNN {acc_exact:.3f}  vs  LSH kNN {acc_lsh_same:.3f}")

# ------------------------------------------------------------- 10. save
step("10. Saving results")
results = pd.DataFrame(rows)
results.to_csv(RESULTS_DIR / "knn_results.csv", index=False)

best_pred = np.array([knn_vote(neigh[i], k_best, w_best) for i in test_idx])
cm = pd.DataFrame(confusion_matrix(y_test, best_pred, labels=[0, 1, 2]),
                  index=["true_" + l for l in LEVELS], columns=["pred_" + l for l in LEVELS])
cm.to_csv(RESULTS_DIR / "knn_confusion_matrix.csv")

print(results.round(3).to_string(index=False))
print(f"\nbest kNN: k={k_best}, weighted={w_best}\n{cm}")
print(f"\nsaved {RESULTS_DIR / 'knn_results.csv'} and {RESULTS_DIR / 'knn_confusion_matrix.csv'}"
      f"  (total {time.time() - start:.0f} s)")
