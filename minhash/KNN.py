#!/usr/bin/env python3
"""
kNN calorie-level classifier on deduped.parquet (RQ4), step by step.

Idea: to predict whether a recipe is low, medium or high in calories, find the
training recipes with the most similar ingredient lists and let them vote.
Similar recipes are found with our own MinHash + LSH, so we never compare all pairs.

Steps
  1. Load deduped.parquet (id, name, ingredients_v2, calories)
  2. Calorie level = tertiles of calories (low / medium / high)
  3. Split: train / test, then train -> fit / validation
     (recipes with the same ingredient set always stay together)
  4. MinHash signature for every recipe (own implementation)
  5. LSH index of the FIT recipes (32 bands x 4 rows)
  6. Choose k and the voting rule on the VALIDATION recipes (k = 1 ... 100)
  7. Add the validation recipes to the index, then evaluate the chosen
     setting ONCE on the TEST recipes
  8. Baselines: majority class and logistic regression
  9. Check: exact kNN (no LSH) on 2,000 test recipes
 10. Save the results
 11. Plots: validation curve + confusion matrix (results/knn_figure.pdf)


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
DATA_FILE   = Path(__file__).resolve().parent / "deduped.parquet"  # output of minhash.py
RESULTS_DIR = Path("results")
SEED        = 42
TEST_SIZE   = 0.2        # 20 % of the recipes are kept aside for the final test
VAL_SIZE    = 0.2        # 20 % of the training recipes are used to choose k
N_HASH      = 128        # signature length
BANDS, ROWS = 32, 4      # 32 x 4 = 128; finds most pairs with Jaccard >= 0.5
K_VALUES    = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]   # tried on the validation set
TOP_N       = max(K_VALUES)                             # neighbours kept per recipe
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

# ------------------------------------------------------------------ 3. split
step("3. Split: train / test, then train -> fit / validation")
# Recipes with exactly the same ingredient set form one group, and a group is never split:
# otherwise a copy on one side would simply give away the answer for its twin on the other side.
# The test set is used only ONCE, at the end. k is chosen on the validation set, which is
# carved out of the training set in the same group-aware way.
group = df["ingr"].map(lambda s: "|".join(sorted(s))).to_numpy()   # same text = same ingredient set
splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, test_idx = next(splitter.split(df, groups=group))
splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
fit_pos, val_pos = next(splitter.split(train_idx, groups=group[train_idx]))
fit_idx, val_idx = train_idx[fit_pos], train_idx[val_pos]
print(f"   fit {len(fit_idx):,}  |  validation {len(val_idx):,}  |  test {len(test_idx):,}")

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
step("5. LSH index of the fit recipes")
# Cut each signature into 32 bands of 4 numbers. Two recipes with an identical band land in
# the same bucket and become candidates. Only the recipes in the index can be found (and vote):
# first the FIT recipes (to choose k), later also the validation recipes (for the final test).
buckets = [defaultdict(list) for _ in range(BANDS)]


def add_to_index(rows):
    for i in rows:
        for band in range(BANDS):
            key = sig[i, band * ROWS:(band + 1) * ROWS].tobytes()
            buckets[band][key].append(i)


add_to_index(fit_idx)


def jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2)


def lsh_neighbours(i, top_n=TOP_N):
    """Top-n indexed recipes most similar to recipe i: LSH candidates, then exact Jaccard."""
    cand = set()
    for band in range(BANDS):
        cand.update(buckets[band].get(sig[i, band * ROWS:(band + 1) * ROWS].tobytes(), ()))
    scored = sorted(((jaccard(sets[i], sets[j]), j) for j in cand), reverse=True)
    return scored[:top_n]                            # list of (similarity, row of an indexed recipe)


def knn_vote(nbrs, k, weighted, fallback):
    """Each of the k nearest neighbours votes for its level (weight 1, or weight = similarity)."""
    if not nbrs:
        return fallback                              # no neighbour: fall back to the majority class
    score = np.zeros(3)
    for sim, j in nbrs[:k]:
        score[y[j]] += sim if weighted else 1.0
    return int(score.argmax())


# --------------------------------------------- 6. choose k on the validation set
step("6. Choosing k on the validation set (voters = fit recipes only)")
majority_fit = Counter(y[fit_idx]).most_common(1)[0][0]
neigh_val = {i: lsh_neighbours(i) for i in val_idx}
y_val = y[val_idx]
rows = []
for weighted in (False, True):
    for k in K_VALUES:
        pred = np.array([knn_vote(neigh_val[i], k, weighted, majority_fit) for i in val_idx])
        acc = accuracy_score(y_val, pred)
        rows.append(dict(split="validation", model="knn_lsh", k=k, weighted=weighted, accuracy=acc))
        print(f"   k={k:3d}  weighted={str(weighted):5}  validation accuracy={acc:.3f}")
val_rows = [r for r in rows if r["model"] == "knn_lsh"]
best = max(val_rows, key=lambda r: (round(r["accuracy"], 3), -r["k"]))   # ties -> the smaller k
k_best, w_best = best["k"], best["weighted"]
print(f"   -> chosen on validation: k={k_best}, weighted={w_best} (validation accuracy {best['accuracy']:.3f})")
del neigh_val

# ----------------------------------------------- 7. final evaluation on the test set
step("7. Final test: voters = all training recipes (fit + validation), chosen k only")
add_to_index(val_idx)                               # the index now holds every training recipe
majority = Counter(y[train_idx]).most_common(1)[0][0]
neigh = {i: lsh_neighbours(i, top_n=k_best) for i in test_idx}
covered = np.array([len(neigh[i]) > 0 for i in test_idx])
y_test = y[test_idx]
best_pred = np.array([knn_vote(neigh[i], k_best, w_best, majority) for i in test_idx])
acc_test = accuracy_score(y_test, best_pred)
rows.append(dict(split="test", model="knn_lsh", k=k_best, weighted=w_best, accuracy=acc_test,
                 accuracy_covered=accuracy_score(y_test[covered], best_pred[covered]),
                 coverage=covered.mean()))
print(f"   {covered.mean():.1%} of test recipes have at least one neighbour")
print(f"   TEST accuracy (k={k_best}, weighted={w_best}) = {acc_test:.3f}")

# ------------------------------------------------------------- 8. baselines
step("8. Baselines")
acc_major = accuracy_score(y_test, np.full(len(y_test), majority))
rows.append(dict(split="test", model="majority_class", accuracy=acc_major))
print(f"   majority class      accuracy={acc_major:.3f}")

# Logistic regression on a 0/1 table: one column per ingredient, 1 if the recipe contains it.
X = sp.csr_matrix((np.ones(sum(map(len, sets)), dtype=np.float32),
                   [pos[ing] for s in sets for ing in s],
                   np.concatenate([[0], np.cumsum([len(s) for s in sets])])),
                  shape=(len(sets), len(vocab)))
logreg = LogisticRegression(max_iter=1000, random_state=SEED).fit(X[train_idx], y[train_idx])
acc_lr = accuracy_score(y_test, logreg.predict(X[test_idx]))
rows.append(dict(split="test", model="logistic_regression", accuracy=acc_lr))
print(f"   logistic regression accuracy={acc_lr:.3f}")

# ------------------------------------------- 9. check: exact kNN without LSH
step("9. Check: exact kNN (all training recipes) on 2,000 test recipes")
sample = rng.choice(test_idx, 2000, replace=False)
size = np.asarray(X.sum(axis=1)).ravel()
X_train = X[train_idx]
exact_pred, lsh_pred = [], []
for chunk in np.array_split(sample, 10):
    inter = (X[chunk] @ X_train.T).toarray()                   # shared ingredients
    jac = inter / (size[chunk][:, None] + size[train_idx][None, :] - inter)
    for row, i in zip(jac, chunk):
        top = np.argsort(-row)[:k_best]
        exact_pred.append(knn_vote([(row[t], train_idx[t]) for t in top if row[t] > 0], k_best, w_best, majority))
        lsh_pred.append(knn_vote(neigh[i], k_best, w_best, majority))
acc_exact = accuracy_score(y[sample], exact_pred)
acc_lsh_same = accuracy_score(y[sample], lsh_pred)
rows.append(dict(split="test", model="knn_exact_2000", k=k_best, weighted=w_best, accuracy=acc_exact))
rows.append(dict(split="test", model="knn_lsh_same_2000", k=k_best, weighted=w_best, accuracy=acc_lsh_same))
print(f"   same 2,000 recipes:  exact kNN {acc_exact:.3f}  vs  LSH kNN {acc_lsh_same:.3f}")

# ------------------------------------------------------------- 10. save
step("10. Saving results")
results = pd.DataFrame(rows)
results.to_csv(RESULTS_DIR / "knn_results.csv", index=False)

cm = pd.DataFrame(confusion_matrix(y_test, best_pred, labels=[0, 1, 2]),
                  index=["true_" + l for l in LEVELS], columns=["pred_" + l for l in LEVELS])
cm.to_csv(RESULTS_DIR / "knn_confusion_matrix.csv")

print(results.round(3).to_string(index=False))
print(f"\nchosen on validation: k={k_best}, weighted={w_best}\n{cm}")
print(f"\nsaved {RESULTS_DIR / 'knn_results.csv'} and {RESULTS_DIR / 'knn_confusion_matrix.csv'}"
      f"  (total {time.time() - start:.0f} s)")


# ------------------------------------------------------------- 11. plots
step("11. Plots for the report")
import matplotlib
matplotlib.use("Agg")                     # save to files, no window needed
import matplotlib.pyplot as plt


def plot_knn(results_csv, cm_csv, out_file):
    """(a) validation accuracy against k, (b) test confusion matrix (row %, count)."""
    res = pd.read_csv(results_csv)
    cm_counts = pd.read_csv(cm_csv, index_col=0).to_numpy()
    val = res[res["split"] == "validation"]
    chosen = res[(res["split"] == "test") & (res["model"] == "knn_lsh")].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.2), gridspec_kw=dict(width_ratios=[1.5, 1]))

    # (a) one line per voting rule; the chosen setting is circled
    for weighted, color, marker, name in ((False, "#eb6834", "o", "unweighted"),
                                          (True, "#2a78d6", "s", "weighted")):
        d = val[val["weighted"] == weighted].sort_values("k")
        ax1.plot(d["k"], d["accuracy"] * 100, color=color, marker=marker, ms=4, lw=1.6, mec="white")
        ax1.text(d["k"].iloc[-1] * 1.18, d["accuracy"].iloc[-1] * 100, name, va="center", fontsize=7)
    best = val[(val["k"] == chosen["k"]) & (val["weighted"] == chosen["weighted"])]["accuracy"].iloc[0] * 100
    ax1.plot(chosen["k"], best, "o", ms=10, mfc="none", mec="black")
    ax1.annotate(f"chosen: k = {int(chosen['k'])}", xy=(chosen["k"], best), xytext=(chosen["k"], best - 1.6),
                 ha="center", fontsize=7, arrowprops=dict(arrowstyle="-", color="grey", lw=0.7, shrinkB=6))
    ticks = sorted(val["k"].unique().astype(int))
    ax1.set_xscale("log"); ax1.minorticks_off()
    ax1.set_xticks([t for t in ticks if t in (1, 3, 5, 10, 20, 50, 100)])
    ax1.set_xticklabels([t for t in ticks if t in (1, 3, 5, 10, 20, 50, 100)])
    ax1.set_xlim(0.8, max(ticks) * 2.1)
    ax1.set_xlabel("k (neighbours voting)", fontsize=8); ax1.set_ylabel("validation accuracy (%)", fontsize=8)
    ax1.grid(axis="y", color="#e6e5e1", lw=0.6); ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("(a) validation accuracy vs k", fontsize=8, loc="left")

    # (b) confusion matrix: colour = share of the true level (row %), text = % and count
    share = cm_counts / cm_counts.sum(axis=1, keepdims=True)
    ax2.imshow(share, cmap="Blues", vmin=0, vmax=0.8)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{share[i, j]:.0%}\n{cm_counts[i, j]:,}", ha="center", va="center", fontsize=7,
                     color="white" if share[i, j] > 0.5 else "black")
    ax2.set_xticks(range(3), LEVELS, fontsize=7); ax2.set_yticks(range(3), LEVELS, fontsize=7)
    ax2.set_xlabel("predicted level", fontsize=8); ax2.set_ylabel("true level", fontsize=8)
    ax2.tick_params(length=0); [s.set_visible(False) for s in ax2.spines.values()]
    ax2.set_title("(b) test confusion matrix", fontsize=8, loc="left")

    ax1.tick_params(labelsize=7)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out_file)                                    # .pdf for Overleaf (sharp at any size)
    fig.savefig(Path(out_file).with_suffix(".png"), dpi=250)  # .png for quick viewing
    plt.close(fig)


plot_knn(RESULTS_DIR / "knn_results.csv", RESULTS_DIR / "knn_confusion_matrix.csv", RESULTS_DIR / "knn_figure.pdf")
print(f"   saved {RESULTS_DIR / 'knn_figure.pdf'} and .png")