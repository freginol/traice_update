# TrAIce

This code encapsulates the main TrAIce backend pipeline, for Investment Advisor (IA) analysis.

- `traice_pipeline.py`: the original one-file codebase. Very cumbersome to work with. This is kept as legacy but running it is not recommended.
- `traice_batch.py`: the main driver for the rewrite. See "How to Run" below.
- `check_pickled.py`: a helper utility for checking that two sets of pickled DataFrames match.

#### How to Run

1) Input files are required: `LEX_TRADE.csv`, `LEX_ACCT_TRADE_BRIDGE.csv`, `LEX_ACCT.csv`, `LEX_TRD_IND.csv`, `LEX_COMPLAINTS.csv`, and `LEX_GROSS_IA_INCOME.csv`. Check with Lex management or contact Radek Guzik.

2) Set the variables at the top of `traice_batch.py`:

- INPUT_DIR: the location of the input files
- PICKLED_DIR: where to write the intermediate pickled results (see "Batch Setup")
- COMPLETION_DIR: where to write the completion files (see "Batch Setup")

3) Run `traice_batch.py`.

#### Batch Setup

The `trace_batch.py` process takes a long time to run (about 2 hours on a 2018 Lenovo laptop). To allow quicker re-runs, it has been divided into ten steps. For details on each step, consult `traice_batch.py` or the underlying modules in the `traice` package:

1) Load
2) Merge
3) Breakout
4) Aggregate
5) Fit
6) HitList
7) HitListExp
8) KRIDetails
9) WellBeing
10) BranchBin

Every time a step is run, its results are pickled to PICKLED_DIR, and a "completion" file for the step is written to the COMPLETION_DIR. If a completion file exists for a given step, it will not be re-run if `traice_batch.py` is triggered again. This allows the `traice_batch.py` to be re-run multiple times in case of failures or to perform analysis, without repeating work needlessly.

To "release" a step and allow it to run again, simply delete the corresponding completion file.

A note about the fit.py process: The running of the explainer in line 100 of fit.py takes over 1.5 hours. It is recommended to run it once and pickle the results. In the current code the results have been pickled as '/df_ia_agg_scored_explainer.pkl'.
