# Backlog – one section = one GitHub issue (template .github/ISSUE_TEMPLATE/story.md)

Topic: "Ingredient pairing and recipe similarity on Food.com". Full steps for Claude:
docs/PROJECT_INSTRUCTIONS_FOR_CLAUDE_CODE.txt (PARTS 4-8). Estimates in hours.
Milestones: S1 Tue 17 Nov · S2 Tue 24 Nov (freeze) · S3 Sun 29 Nov · S4 Thu 3 Dec.
Hand-offs: H1 Thu 12 Nov (recipes v2, R1 → all) · H2 Thu 19 Nov (DUPLICATES + NEIGHBOURS R2 → R3, R5; PAIRS v1 R3 → R4) · PAIRS v2 Sun 22 Nov.

## S1 · Algorithms correct (Tue 10 → Tue 17 Nov)
### R1-S1a · Skeleton into the group repo (R1, 2 h) — Wed 11
- [ ] skeleton merged by PR; `pytest` 53 passed in CI; board, milestones and issues created
### R1-S1b · Ingredient normalisation v2 (R1, 5 h) — hand-off H1 Thu 12
- [ ] one test per rule (plurals with exceptions, adjectives, synonyms); 30 examples in the PR
- [ ] `results/data_summary.csv` (labels before/after, calorie-level bounds); group told to rerun cleaning
### R2-S1 · Shingles, MinHash, banding, search (R2, 7 h)
- [ ] identical sets → identical signatures; |estimate − Jaccard| < 0.1 on 200 pairs (200 hashes)
- [ ] exact band candidates on hand-made signatures; `similar_recipes` finds 103 for 102 (mini data)
- [ ] candidate overlap with datasketch on the sample; provisional neighbours on the sample Sun 15
### R3-S1 · A-Priori, PCY, rules, pair table (R3, 7 h)
- [ ] known counts on 8 hand-made baskets; support boundary; pcy == apriori(max_size=2) with n_buckets = 7
- [ ] same frequent itemsets as mlxtend on the sample; provisional pairs on the sample Sun 15
### R4-S1 · PMI weights and graph functions (R4, 6 h)
- [ ] PMI/NPMI of a hand-computed pair; two cliques → 2 Louvain communities, modularity > 0.4
- [ ] spectral k=2 same split; null graph keeps degrees; NMI = 1 for perfect labels; first graph picture
### R5-S1 · Classifier functions + report document (R5, 6 h)
- [ ] 6-recipe neighbour table with known votes; only training neighbours vote; NA → coverage
- [ ] copies never split between train and test; report document with budgets shared

## S2 · Full data + evaluation (→ Tue 24 Nov, FREEZE at the session)
### R1-S2 · Full pipeline + integration (R1, 5 h)
- [ ] `scripts/run_all.py` → `runtime.csv`; integration on the full data Fri 20; 3 descriptive figures
### R2-S2 · Tuning, search, hand-off (R2, 8 h) — H2 Thu 19
- [ ] `lsh_eval.csv`: 4 (b, r) × 4 thresholds vs exact Jaccard on 5,000 recipes; (b, r) argued with the S-curve
- [ ] `neighbours.parquet` (top 20) and `duplicates.parquet` on main Thu 19
- [ ] `search_latency.csv` (LSH vs brute force, 3 sizes); 3 example queries kept
### R3-S2 · Benchmark, PAIRS, rules (R3, 8 h) — H2 Thu 19, v2 Sun 22
- [ ] `itemsets_bench.csv`: 3 supports × 3 bucket counts, seconds, peak MB, pairs counted
- [ ] `pairs.parquet` v1 Thu 19, v2 without duplicates Sun 22; `rules_top.csv` overall and per cuisine
### R4-S2 · Cuisine discovery (R4, 8 h)
- [ ] `graph_eval.csv`: 4 weightings × Louvain/spectral, modularity vs null, NMI vs cuisine
- [ ] `communities.csv` for the best weighting; 5 lines of interpretation
### R5-S2 · Evaluation on the full data (R5, 6 h)
- [ ] `classifier_eval.csv`: k sweep, weighted or not, majority, logistic, exact kNN on 5,000 recipes
- [ ] `confusion_matrix.csv` of the best model; 5 lines on which levels are confused and why

## S3 · Writing (Wed 25 → Sun 29 Nov; sections Fri 27)
### R1-S3 Data (1,500, 4 h) · R2-S3 RQ2 (2,000, 4 h) · R3-S3 RQ1 (2,000, 4 h) · R4-S3 RQ3 (1,800, 4 h)
- [ ] section within budget (`count_chars.py`), every number from `results/`, figures captioned
- [ ] R1: final notebook assembled, Run All on the sample and the full data
### R5-S3 · Report assembly (R5, 7 h)
- [ ] Introduction + abstract, RQ4, conclusion; full draft ≤ 12,500 characters Sun 29

## S4 · Polish and hand-in (Mon 30 Nov → Thu 3 Dec; review Tue 1 Dec)
### R1-S4 · Fresh-clone check + hand-in package (R1, 3 h)
- [ ] clean clone → install → run_all → notebook Run All; tag v1.0
### R2-S4 / R3-S4 / R4-S4 · Cross-review (2 h each)
- [ ] PROMPT E on the next module of the circle; comments on the next section
### R5-S4 · Final review + submission (R5, 3 h)
- [ ] PROMPT R list empty or justified; names + contributions; submitted on DTU Learn Thu 3 Dec
