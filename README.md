# AdvAI_critique: does sprint pace classification hold up on real Jira data?

CSBP711 Advanced AI, Fall 2026, Paper Critique assignment.
Author: Hani AbuSharkh (HCT, Al Ain).

## The paper

Y. J. Pérez Castillo, S. D. Orantes Jiménez and P. O. Letelier Torres, "Sprint Management in Agile Approach: Progress and Velocity Evaluation Applying Machine Learning," *Information*, vol. 15, no. 11, art. 726, 2024. https://doi.org/10.3390/info15110726 (open access, CC BY 4.0).

**What it does.** It turns two sprint charts from the Worki tool (burndown and "completed vs not completed work") into 11 numbers per sprint (R1 to R11: % of work completed, stalled periods, days without reduction, excessive drops, increases). An expert labels each sprint Excellent, Good, Fair or Poor. Seven classifiers (KNN, SVM, MLP, RF, NB, LR, AdaBoost) are compared with cross-validation on 551 student sprints. SVM is best at 0.88 accuracy (Table 3).

**Why we chose it.** It is the closest published work to our course project, which predicts sprint outcomes on TAWOS (8 open-source Jira projects) from the sprint state rebuilt from the Jira change log. Both papers are about watching a sprint and flagging the ones in trouble.

## What we test

The paper's data and code are private, so we rebuild a faithful approximation on TAWOS with our Assignment 1 setup (same 1,691 nonempty sprints, same per-project 70/30 time split, seed 42):

1. A day-by-day burndown for every sprint, rebuilt from the change log (`src/build_burndown.py`).
2. Our versions of R1 to R11 and of the 4-level TAG (`src/features.py`). The paper gives no thresholds, so we wrote our rules down before running any model. TAWOS has no expert labels, so our TAG is a fixed rule that reads the same chart, which is what the paper's expert did. The class sizes follow the paper's Table 4.
3. Three experiments (`src/reality_check.py`):
   - **E1** the paper's protocol: end-of-sprint features, 7 classifiers, random stratified 10-fold CV, plus two baselines the paper does not report (majority class, and logistic regression on R1 alone).
   - **E2** honest splits: time split and leave-one-project-out.
   - **E3** early warning: features from the first 25% or 50% of the sprint only, which is the use the paper motivates.
   Everything runs twice: with our main TAG and with a "pace only" TAG as a sensitivity check.

## Results (accuracy; full table in `results/reality_check_results.csv`)

| Setting | Main TAG: best of 7 | Main TAG: R1 only | Pace-only TAG: best of 7 | Pace-only TAG: R1 only | Majority class |
|---|---|---|---|---|---|
| Paper, Table 3 (551 student sprints, their labels) | .88 (SVM) | not reported | | | not reported (.30 from Table 4) |
| E1 random 10-fold CV, end of sprint | .774 (RF) | .774 | .661 (RF) | .624 | .297 / .288 |
| E2 time split, end of sprint | .748 (RF) | .751 | .646 (MLP) | .552 | .284 / .254 |
| E2 leave one project out | .764 (LR) | .770 | .639 (MLP) | .607 | .154 / .134 |
| E3 time split, first 50% only | .585 (MLP, RF, LR tie) | .597 | .671 (LR) | .669 | .284 / .254 |
| E3 time split, first 25% only | .507 (NB) | .474 | .587 (MLP) | .550 | .284 / .254 |

What we take from it:

- **Most of the skill is R1.** With our main label, logistic regression on "% of work completed" alone matches all seven models (.774 vs .774). With the pace-only label the other ten features add .04 under random CV and .09 under the time split. Either way, most of what the models know is how much was finished.
- **Random CV did not inflate much here** (.774 vs .748 time split and .760 across projects). Our label is the same rule for every project; the paper's label came from one expert on about three sprints per student team, which we cannot test.
- **As an early warning it is much weaker.** At 25% of the sprint, accuracy falls to .51 to .59, against .28 for the majority class. Poor sprints are still caught (recall about .9) but with many false alarms (precision drops from about .89 to .54 for the main label).
- **The result depends on how the label is defined.** Changing the label weights moves accuracy by more than .10. The paper does not say how its expert decided.
- **The paper's "Recall" column is the same as its "Accuracy" column in every row** of Table 3. That is what weighted-average recall always gives, so the table hides the per-class picture. From its own Table 4, Fair has recall .62 and F1 .68, while the other classes are above .88.

## How to run

```
pip install -r requirements.txt
python src/reality_check.py          # about 4 minutes, uses data/burndown_daily.csv
```

To rebuild `data/burndown_daily.csv` from the raw TAWOS tables you need our main project repo (https://github.com/hani-ziad/SwEng_202610) on disk:

```
set SWENG_ROOT=C:\Project\course-submission\SwEng_202610     (Windows; default is ../course-submission/SwEng_202610)
python src/build_burndown.py         # about 15 seconds
```

Our run: Python 3.13, scikit-learn 1.9.1, pandas 3.0, numpy 2.5, seed 42 (see `results/summary.json`).

## Files

- `src/build_burndown.py` rebuilds the daily burndown (1,691 sprints, 24,973 sprint-days).
- `src/features.py` defines R1 to R11 and the TAG.
- `src/reality_check.py` runs E1 to E3 for both labels.
- `data/burndown_daily.csv` is the derived daily burndown (from TAWOS, Apache 2.0, Tawosi et al., MSR 2022).
- `results/` holds all scores, bootstrap intervals (`reality_check_intervals.csv`), the per-sprint table and the run log.
- `report/` holds the report draft and our notes on the paper.

## Honesty notes

- The paper's numbers are quoted from its Tables 3 and 4. We did not run their code or see their data; neither is public.
- Our R1 to R11 and TAG are approximations on issue counts, not Worki hours, and our TAG is a rule, not an expert. Every number in our tables is from our own run.

## AI-use statement

We used Claude (Anthropic) as a coding and writing assistant. It helped write the scripts in `src/`, ran them, and drafted the README and report text. We chose the paper, checked the paper's tables against the published article, set the experiment plan, and reviewed and edited all code and text. All results come from running the code in this repository.
