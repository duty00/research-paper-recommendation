# README figure data

`figure_data.json` contains the measurements used in the README plots:

- Aggregate training-label and train/test venue counts from the original competition files.
- Nine Public leaderboard scores recorded by the team.
- Seven observed fold scores per seed for V5 and both V6 candidates: 42 scores in total.

The CV source report and raw input hashes are recorded under `provenance`. The Public scores come from the submission results supplied by the team. The snapshot contains no paper IDs, paper text, or row-level labels.

Regenerate the figures from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-figures.txt
.\.venv\Scripts\python.exe tools/make_readme_figures.py
```

`figures/manifest.json` records plotting versions, input/output hashes, and the CV means. The dots in the CV plot are measured fold scores. The boxplots summarize two repeated splits of the same dataset; they should not be interpreted as confidence intervals from independent datasets.
