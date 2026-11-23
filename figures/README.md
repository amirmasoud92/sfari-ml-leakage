# Figure panels

Each data-driven panel has its own folder, `figure_<n>/panel_<letter>/` (`figure_S<n>` for Additional file 1).
The folder contains the plotted values as CSV and a `plot.py` that draws that panel alone, using the colours and
helpers in `style.py`. The column names are described in each script's docstring. The CSVs hold the values exactly
as plotted, with no participant identifiers.

| Figure | Panels with data and code | Panels without code |
|---|---|---|
| 1 | none | all cards (overview schematic; its small plots repeat the results of Figs. 2d, 3b, 4 and 5a) |
| 2 | a identification under controls · b single 1-Hz bins · c variance shares by frequency · d single-electrode identification maps · e alpha-band variance-share maps | — |
| 3 | b dose-response · c model families · d convolutional networks · e positive controls | a (schematic of the evaluation schemes) |
| 4 | `panel/` covariate and EEG classification | — |
| 5 | a reliability vs effect size · b error-corrected AUC · c reliability by group · d–f individual values of three markers | — |
| 6 | a extreme deviations by group · b deviations vs clean minutes · c self-identifiability by group · d self-identifiability vs clean minutes | — |
| S1 | a paradigm-pair identification matrix · b identification by feature family | — |
| S2 | a same vs other visit · b margin over sensor covariance · c self-similarity change within a visit · d variance with a visit term | — |
| S3 | a size-matched control · b cross-visit leak · c number of children · d EEGNet with permuted labels | — |

Where a panel shows a summary of individual runs (Fig. 3b, Fig. S3a, b), both the runs (`runs.csv`) and the plotted
summary (`summary.csv`) are given.
