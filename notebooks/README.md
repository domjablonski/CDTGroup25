# Notebooks

One working notebook per role, plus the final notebook submitted as the report appendix.

| Notebook | Owner | Content |
|---|---|---|
| `01_data_R1.ipynb` | R1 | loading, cleaning checks, ingredient normalisation before/after, calorie levels, descriptive figures |
| `02_lsh_R2.ipynb` | R2 | RQ2: shingles, MinHash, S-curve and (b, r) tuning vs exact Jaccard, "recipes like this one", latency |
| `03_itemsets_R3.ipynb` | R3 | RQ1: A-Priori vs PCY (time, memory), association rules overall and per cuisine, check against mlxtend |
| `04_graphs_R4.ipynb` | R4 | RQ3: PMI/NPMI weights, ingredient graphs, Louvain vs spectral, modularity vs null, NMI vs cuisines |
| `05_classify_R5.ipynb` | R5 | RQ4: no-leakage split, kNN on LSH neighbours, baselines, accuracy / macro-F1, confusion matrix |
| `final_notebook.ipynb` | R1 (assembly) | clean, documented version of the five notebooks, in the order of the report |

Rules:
- Import code from `src/foodcom/`; keep notebooks for calls, figures and comments.
- Start every notebook with `SAMPLE = True` and switch to `False` only for final runs.
- Before merging into `final_notebook.ipynb`: "Restart & Run All" without errors.
