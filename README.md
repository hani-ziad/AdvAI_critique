# AdvAI_critique: does sprint grading hold up on real Jira data?

My paper critique for CSBP711 Advanced AI, Fall 2026.
Hani AbuSharkh, HCT Al Ain.

## The paper

Y. J. Pérez Castillo, S. D. Orantes Jiménez and P. O. Letelier Torres, "Sprint Management in Agile Approach: Progress and Velocity Evaluation Applying Machine Learning," *Information*, vol. 15, no. 11, art. 726, 2024. https://doi.org/10.3390/info15110726 (open access, CC BY 4.0).

The authors take two charts from the Worki tool (the burndown and "completed vs not completed work") and turn each sprint into 11 numbers, R1 to R11: how much work got done, stalled periods, days without a real drop in remaining work, sudden drops and sudden jumps. An expert grades each sprint as Excellent, Good, Fair or Poor. They compare seven classifiers (KNN, SVM, MLP, RF, NB, LR, AdaBoost) with cross-validation on 551 student sprints, and SVM comes out on top at 0.88 accuracy (Table 3).

I chose it because it is the closest published work to my course project, which predicts sprint outcomes on TAWOS (8 open-source Jira projects) from the sprint state rebuilt out of the Jira change log. Both are about watching a sprint and flagging the ones in trouble.

## What I test

Their data and code are private, so I rebuilt the closest version I could on TAWOS, with my Assignment 1 setup (the same 1,691 sprints with committed work, the same per-project 70/30 time split, seed 42).

1. `src/build_burndown.py` rebuilds a day-by-day burndown for every sprint from the change log.
2. `src/features.py` holds my versions of R1 to R11 and the 4-level grade. The paper gives no thresholds, so I wrote my rules down before running any model. TAWOS has no expert grades, so my grade is a fixed rule that reads the same chart, like their expert did, with class sizes taken from the paper's Table 4.
3. `src/reality_check.py` runs three experiments:
   - **E1**, the paper's setup: end-of-sprint features, the 7 classifiers and random stratified 10-fold CV. I add two baselines the paper leaves out: always guessing the most common grade, and logistic regression on R1 alone.
   - **E2**, fairer splits: my time split, and leaving one project out.
   - **E3**, early warning: only the first 25% or 50% of the sprint is visible, which is how the paper says the tool should be used.

Everything runs twice: once with my main grade and once with a "pace only" grade as a check.

## Results

Accuracy; the full table is in `results/reality_check_results.csv`.

| Setting | Main grade: best of 7 | Main grade: R1 only | Pace-only: best of 7 | Pace-only: R1 only | Most common grade |
|---|---|---|---|---|---|
| Paper, Table 3 (551 student sprints, their grades) | .88 (SVM) | not reported | | | not reported (.30 from Table 4) |
| E1 random 10-fold CV, end of sprint | .774 (RF) | .774 | .661 (RF) | .624 | .297 / .288 |
| E2 time split, end of sprint | .748 (RF) | .751 | .646 (MLP) | .552 | .284 / .254 |
| E2 leave one project out | .764 (LR) | .770 | .639 (MLP) | .607 | .154 / .134 |
| E3 time split, first 50% only | .585 (MLP, RF, LR tie) | .597 | .671 (LR) | .669 | .284 / .254 |
| E3 time split, first 25% only | .507 (NB) | .474 | .587 (MLP) | .550 | .284 / .254 |

What I take from it:

- **Almost everything comes from R1.** With the main grade, logistic regression on "% of work completed" alone matches all seven models (.774 against .774). With the pace-only grade the other ten features add .04 under random CV and .09 under the time split. Either way, the models mostly know how much was finished.
- **Random folds didn't inflate the score much here** (.774 against .748 on the time split and .764 across projects). My grade uses the same rule for every project. Theirs came from one expert grading about three sprints per student team, which I can't test.
- **As an early warning it gets much weaker.** At 25% of the sprint, accuracy falls to between .51 and .59, against .28 for guessing. Poor sprints are still caught (recall around .9), but with lots of false alarms: precision for Poor drops from about .89 to .54 with the main grade.
- **The result depends on how the grade is defined.** Changing the weights moves accuracy by more than .10, and the paper never says how its expert decided.
- **The paper's "Recall" column equals its "Accuracy" column in every row of Table 3.** Weighted-average recall always does that, so the table hides how each grade does. Its own Table 4 shows Fair at recall .62 and F1 .68, while the other grades are all above .88.

## How to run

```
pip install -r requirements.txt
python src/reality_check.py          # about 4 minutes, uses data/burndown_daily.csv
```

To rebuild `data/burndown_daily.csv` from the raw TAWOS tables, you need my main project repo (https://github.com/hani-ziad/SwEng_202610) on disk:

```
set SWENG_ROOT=C:\Project\course-submission\SwEng_202610     (Windows; default is ../course-submission/SwEng_202610)
python src/build_burndown.py         # about 15 seconds
```

My run used Python 3.13, scikit-learn 1.9.1, pandas 3.0, numpy 2.5 and seed 42 (see `results/summary.json`).

## Files

- `src/build_burndown.py` rebuilds the daily burndown (1,691 sprints, 24,973 sprint-days).
- `src/features.py` defines R1 to R11 and the grade.
- `src/reality_check.py` runs E1 to E3 for both grades.
- `data/burndown_daily.csv` is the daily burndown, derived from TAWOS (Apache 2.0; Tawosi et al., MSR 2022).
- `results/` has all scores, the bootstrap intervals (`reality_check_intervals.csv`), the per-sprint table and the run log.
- `report/` has the report draft and my notes on the paper.

## A note on honesty

The paper's numbers are quoted from its Tables 3 and 4. I didn't run their code or see their data, since neither is public. My R1 to R11 and grade are approximations built on issue counts rather than Worki hours, and my grade is a rule, not an expert. Every number in my tables comes from my own runs.

## AI-use statement

I used Claude (Anthropic) as a coding and writing assistant. It helped write and run the scripts in `src/` and drafted the README and report. I chose the paper, checked its tables against the published article, set the experiment plan, and reviewed and edited all code and text. All results come from running the code in this repository.
