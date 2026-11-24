# AI usage log
We use Claude Code (Anthropic) as a coding assistant. Every entry: what the AI produced, what we
checked and how, what we changed by hand. The report states how AI was used (check DTU's rules
on AI in group projects and the course page).

| Date | Story | Member | What Claude did | How we verified | Human changes |
|------|-------|--------|-----------------|-----------------|---------------|
| 2026-11-03 | skeleton | R1 | Repo skeleton, cleaning.py, contract, tests, CI, docs | Ran pytest (51 passed); ran clean_recipes on the full RAW_recipes.csv | Team names to fill in |
| 2026-11-10 | topic change | R1 | Contract v2.0, stubs (lsh, itemsets, graphs, classify), calorie_level, tests, docs, plan PDFs for "Ingredient pairing and recipe similarity" | pytest (53 passed); cleaning run on the full RAW_recipes.csv (calorie tertiles 217.7 / 431.7 kcal) | Team names to fill in |
| 2026-11-16 | R4-S1 | R4 | test_graphs.py (25 toy tests), graphs.py (pmi_weights, ingredient_graph, dominant_cuisine, louvain, spectral, null_graph, evaluate_partition), scripts/r4_first_look.py (provisional pairs + first graph) | Tests seen failing then green (78 passed); hand-computed PMI (lift 4, pmi 2); provisional run on the full data, communities checked by eye | - |
