# Report outline – "Ingredient pairing and recipe similarity on Food.com"

Limit for a group of 5: **12,500 characters excluding spaces** (≈ 4.5 pages), names of all members on the title page.
Count with: `python scripts/count_chars.py report/report.txt`.

| Section | Budget | Writer | Content |
|---|---|---|---|
| Title, names, abstract + outline | 900 | R5 | Main question, methods (course / outside the course), main results in numbers |
| 1. Introduction | 1,300 | R5 | Motivation (ingredient pairing, "recipes like this one"), main question, RQ1–RQ4, which methods are from the course and which are not |
| 2. Data and preprocessing | 1,500 | R1 | Source and licence, size, cleaning flags, ingredient normalisation (before/after), calorie levels, sample |
| 3. RQ1 – Ingredient pairing (PCY vs A-Priori) | 2,000 | R3 | Support threshold, PCY vs A-Priori (time, memory, pairs counted), top rules overall and per cuisine |
| 4. RQ2 – Recipe similarity (MinHash + LSH) | 2,000 | R2 | Shingles, choice of (b, r) with the S-curve, precision/recall vs exact Jaccard, search latency, example queries |
| 5. RQ3 – Cuisine discovery (PMI graph + Louvain) | 1,800 | R4 | PMI/NPMI weighting (explained briefly: outside the course), Louvain vs spectral, modularity vs null, NMI vs cuisines |
| 6. RQ4 – Calorie level from LSH neighbours | 1,700 | R5 | kNN (explained briefly: outside the course), no-leakage split, baselines, accuracy/macro-F1, confusions |
| 7. Discussion and conclusion | 1,300 | R5 + all | One answer per RQ, limitations, what we would do next |
| **Total** | **12,500** | | |

Rules from the brief to keep in mind:
- do not re-explain algorithms taught in class; explain the outside-course methods briefly;
- argue every choice (parameters, thresholds, weights);
- evaluate and reflect: if something works badly, say why.

Appendix (not counted): contributions table (`contributions.md`) and `notebooks/final_notebook.ipynb`.
