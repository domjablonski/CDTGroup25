### RQ3: Does a PMI-weighted ingredient graph reveal cuisines?

We build one graph per edge weighting from the 5,203 ingredient pairs that occur in at least 0.1 % of the
recipes (572 ingredients, the same pairs as PCY). Edges are weighted by raw count, lift, PMI or NPMI. PMI
is our outside-course method: pmi = log2 p(a,b)/(p(a)p(b)), and NPMI divides it by −log2 p(a,b) to remove
the bias towards rare pairs. For lift, PMI and NPMI only positive associations are kept. Each threshold is
the largest that keeps 90 % of the ingredients in the largest connected component, so all weightings are
compared at the same connectivity. We cluster with Louvain (5 seeds) and spectral clustering, and compare
modularity with two null models that keep degrees and weights: rewired edges and shuffled weights. An
ingredient is labelled with a cuisine if at least 40 % of the tagged recipes containing it share it (179 of
506 ingredients); we measure agreement with AMI, which unlike NMI does not grow with the number of communities.

Raw counts give 3 to 4 communities, an AMI of 0.23 and a modularity (0.23) below the rewired null model
(0.33): common ingredients such as salt or butter link to everything, so the graph mostly encodes popularity.
PMI doubles the AMI (0.47; lift 0.40, NPMI 0.44) and yields 7 interpretable communities: baking, American
savoury dishes, Italian/Mediterranean, Mexican, Asian, bread and citrus (Figure X). Shuffling the PMI weights
on the same edges drops the AMI to 0.19, so the weights rather than the topology carry the cuisines. Spectral
clustering with the eigengap always finds k = 2 (sweet versus savoury); with Louvain's k it comes close
(AMI 0.40). Over 9 settings of support, connectivity rule and label share, PMI-type weights always beat raw
counts (AMI 0.27–0.52 versus 0.15–0.29), but the order among lift, PMI and NPMI changes. Limits: only 20 % of
the recipes carry a cuisine tag, and the normalisation still separates garlic and garlic clove.

<!-- Sources: results/graph_eval.csv (AMI, modularity, null models, spectral), results/communities.csv
(community names), results/sensitivity.csv (9 settings), run_graph.py output (5,203 pairs, 572 ingredients,
179 labelled of 506). Figure X = figures/graph_pmi.png. Numbers to re-check after the PCY run of R3. -->
