# lgbm/ (R5): cuisine prediction with LightGBM

**Question.** How well can a recipe's cuisine be predicted from its ingredients alone, and how does a
supervised model compare with the communities of the PMI graph (R4)?

Outside the course: LightGBM (gradient boosted decision trees). Trees are added one after the other, and
each new tree is fitted to the errors (gradient of the loss) left by the previous ones. LightGBM makes this
fast by splitting on histograms of the features and by always growing the leaf that reduces the loss most.

## Run

From the repo root, after `python preprocessing/preprocessing.py`:

```bash
python lgbm/lgbm_cuisine.py tune    # train -> val: choose the parameters
python lgbm/lgbm_cuisine.py test    # refit on train + val, evaluate once on test
python lgbm/lgbm_cuisine.py plot    # redraw the figure only
```

## Setup (shared with R4, see splits/README.md)

- Split: `splits/cuisine_split.csv`, read and never recomputed (40,912 recipes, 11 cuisines).
- Features: 0/1 ingredients (`ingredients_v2`). No tags, name or description, since they contain the label.
- `class_weight="balanced"`, parameters chosen on val by macro F1, one evaluation on test.
- Logistic regression gets the same features, the same class weights and its own C chosen on val.

## Outputs (`results/`)

| File | Content |
|---|---|
| `lgbm_val.csv` | every setting tried on val |
| `lgbm_params.json` | chosen parameters |
| `lgbm_test.csv` | accuracy and macro F1 on test: LightGBM, logistic regression, majority class |
| `lgbm_bootstrap.csv` | paired bootstrap: macro F1 of LightGBM minus logistic regression |
| `lgbm_per_class.csv` | F1 per cuisine |
| `lgbm_confusion_matrix.csv` | LightGBM, rows = true cuisine, alphabetical order |
| `lgbm_test_predictions.csv` | prediction for every test recipe, for comparison with the graph |
| `lgbm_top_ingredients.csv` | 30 ingredients with the highest gain |
| `lgbm_figure.pdf / .png` | confusion matrix and F1 per cuisine |
