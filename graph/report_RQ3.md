### RQ3: Do ingredient communities reveal, and predict, a recipe's cuisine?

**Graph.** Nodes are the 572 ingredients of the 5,203 pairs found in at least 0.1 % of the recipes (the PCY
pairs). Edges are weighted by raw count or by PMI, our outside-course method: pmi = log2 p(a,b)/(p(a)p(b))
measures how much more often two ingredients co-occur than by chance; only positive associations are kept.
Each threshold keeps 90 % of the ingredients connected, and Louvain finds the communities.

**Do communities match cuisines?** Labelling an ingredient with its dominant cuisine, raw counts give 3 to 4
communities and an AMI of 0.23, with a modularity below a degree-preserving null model: salt or butter link to
everything, so the graph encodes popularity. PMI doubles the AMI (0.47) and yields baking, Italian, Mexican,
Asian, bread and citrus communities; shuffling the PMI weights on the same edges drops it to 0.19, so the
weights carry the cuisines. PMI beats counts in all 9 robustness settings.

**Can they predict it?** On the shared split (40,912 recipes, 11 cuisines, test 20 %), the graph is built
without labels and without test recipes; a recipe becomes the share of its ingredients in each community, and a
logistic regression predicts the cuisine. Since finer resolutions give more communities and better scores, we
compare weightings at equal size. PMI beats counts at every size (test macro-F1 0.305, 0.388, 0.466 versus 0.290,
0.371, 0.454 for about 10, 30, 100 communities, 5 seeds). With 100 features instead of 7,371 ingredients, it keeps
75 % of the macro-F1 of a logistic regression on all ingredients (0.623). The trade-off is readability: with 10
communities the Italian, Mexican and Asian ones are clear, but the score is low. Mexican (F1 0.72) and Italian
(0.70) are easy; Caribbean (0.28) and African (0.30) are not: African recipes are mostly confused with Middle
Eastern and Indian ones, which use the same spices, and Caribbean ones with Southern US and Mexican ones.

<!-- Sources: graph/results/graph_eval.csv, communities.csv, sensitivity.csv (part 1); splits/README.md,
graph/results/cuisine_test.csv, cuisine_per_class.csv, cuisine_communities.csv, cuisine_confusion.csv (part 2). Majority class macro-F1
0.03. Comparison with R5's LightGBM to add at the end. -->
