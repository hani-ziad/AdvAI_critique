# Paper Critique: Can a Burndown Chart Grade a Sprint? A Reality Check on Open-Source Jira Data

Hani AbuSharkh, CSBP711 Advanced AI, Fall 2026. Code: https://github.com/hani-ziad/AdvAI_critique

## 1. Summary and positioning

**Paper.** Pérez Castillo, Orantes Jiménez and Letelier Torres, "Sprint Management in Agile Approach: Progress and Velocity Evaluation Applying Machine Learning," *Information* 15(11):726, 2024.

**Problem.** A manager running many projects at once cannot read every sprint's charts. The authors want a model that flags the sprints whose progress and pace are poor.

**Method.** From the Worki tool they pull two charts per sprint, the burndown and the "completed vs not completed work" chart, and turn them into 11 numbers (R1 to R11): % of work completed, how often and how long progress stalled, days without a real drop in remaining effort, and sudden drops or rises. An expert labels each of 551 student sprints as Excellent, Good, Fair or Poor. Seven classifiers are compared with cross-validation.

**Claim.** SVM, LR and RF classify sprint pace with accuracy around .87 to .88 (Table 3), and ML "demonstrates a clear advantage over traditional evaluation methods".

**Why it matters to us.** Our course project predicts sprint outcomes on TAWOS, eight open-source Jira projects, from the sprint state rebuilt from the Jira change log. Its Phase 3 watches the sprint at 25%, 50% and 75% of its length, which is the same idea as reading a burndown. This paper is the closest published attempt to grade sprints from their charts, so it is the natural one to test.

## 2. Critical analysis

**Strengths.** The problem is real and the features are easy to explain to a Scrum Master. The paper says which charts each feature comes from and gives the full confusion matrix (Table 4), which lets a reader check its numbers.

**Weakness 1: the label and the features come from the same charts.** The expert who labelled the sprints helped choose the features "based on how progress is visualized in the charts", and then labelled the sprints by looking at those charts. A high score therefore shows that a model can copy the expert's reading of a chart. It does not show that it can foresee anything. Nothing in the paper measures how much the expert's labels depend on R1 alone.

**Weakness 2: the timing does not match the use.** The paper sells the tool for spotting sprints "that may require immediate intervention", but R1 is defined as the work completed "by the end of the Sprint". At that point there is nothing left to save. No result is reported for a model that sees only part of the sprint.

**Weakness 3: no baseline.** Table 3 compares seven classifiers with each other and with nothing else. There is no majority class (from Table 4 it would be .303), no one-feature rule, and no "traditional method", even though RQ3 claims ML beats traditional methods. SVM, LR and RF differ by .01 with no intervals, yet SVM is named the candidate for deployment.

**Weakness 4: the metrics hide the weak class.** In every row of Table 3 "Recall" equals "Accuracy". That is what a weighted-average recall always gives, so the column adds nothing. From Table 4, Fair has recall .62 and F1 .68 while the other classes are above .88. Specificity is promised in Section 4.6 but never shown. The text also misreads Table 4 once (it calls 5 Good sprints predicted as Fair "Fair misclassified as Good").

**Weakness 5: evaluation and data.** The number of folds and whether folds keep a project's sprints together are not stated. With about three sprints per team, random folds put sprints of the same team on both sides. The data are student projects of two to four weeks, labelled by one expert with no agreement measure and no labelling guide. Data and code are private and the API cannot be named, so the study cannot be repeated. The paper went from submission to acceptance in about five weeks.

**Verdict.** The evidence supports a narrow claim: on these student sprints, a model can reproduce one expert's end-of-sprint grade. It does not support the wider claims about early intervention or an advantage over traditional methods.

## 3. Reality check

**Setup.** We could not get the paper's data or code, so we built a faithful approximation on TAWOS with our Assignment 1 setup: the same 1,691 nonempty sprints, the same per-project 70/30 time split, and seed 42. For each sprint we rebuilt a day-by-day burndown from the change log (24,973 sprint-days), then computed our versions of R1 to R11 on issue counts, using written rules fixed before any model was run. TAWOS has no expert labels, so our TAG is a fixed rule that reads the same chart the way the expert did: half "pace" (how close the done share stayed to the time elapsed) and half final completion, cut into four classes at the paper's class sizes. As a check we also used a "pace only" label. We ran the paper's seven classifiers with library defaults.

**Table 1. Accuracy. Paper numbers are quoted from its Table 3; all other numbers are from our runs.**

| Setting | Best of 7 models | LR on R1 only | Majority class |
|---|---|---|---|
| Paper: 551 student sprints, expert TAG, CV | .88 (SVM) | not reported | .30 (from Table 4) |
| Ours, main TAG: random 10-fold CV (paper protocol) | .774 (RF) | .774 | .297 |
| Ours, main TAG: time split, end of sprint | .748 (RF) [.708, .787] | .751 | .284 |
| Ours, main TAG: leave one project out | .764 (LR) | .770 | .154 |
| Ours, main TAG: time split, first 50% of sprint | .585 [.540, .624] | .597 | .284 |
| Ours, main TAG: time split, first 25% of sprint | .507 [.464, .552] | .474 | .284 |
| Ours, pace-only TAG: random 10-fold CV | .661 (RF) | .624 | .288 |
| Ours, pace-only TAG: time split, first 25% | .587 (MLP) | .550 | .254 |
| Our course pipeline (Phases 1-3, different task, AUC): spillover at sprint start / at 50% | .885 / .957 | | |
| Our course pipeline (Phases 1-3, different task, AUC): late closure at sprint start | .751 | | |

Brackets are 95% sprint-level bootstrap intervals.

**Gaps and what explains them.**

1. *Our ceiling is lower (.77 vs .88).* The data differ (open-source teams, sprints of about two weeks, issue counts instead of hours) and the labels differ (a rule instead of an expert). A milder or simpler expert would give higher scores. We cannot close this gap without their data.
2. *One feature does the work.* With our main label, R1 alone matches all seven models (.774 vs .774). With the pace-only label the other ten features add .04 under random CV and .09 under the time split. On end-of-sprint data the models mostly read back how much was finished. This is the property behind Weakness 1.
3. *Random CV did not inflate our score much* (.774 vs .748 time split, .764 across projects). Our label is the same rule for every project, so there is no team-specific labelling to leak. The paper's case may differ, since one expert labelled about three sprints per student team, but we could not test that.
4. *Early warning is much weaker.* With only the first 25% of the sprint, accuracy drops to .51 (main label), still well above .28 for the majority class. Poor sprints are still caught (recall about .89) but with many false alarms: Poor precision falls from .89 at the end of the sprint to .54. This is the use the paper sells, and it is not what Table 3 measures.
5. *The label definition moves the result by more than .10* (.774 vs .661 under the same protocol). Since the paper does not say how its expert decided, its .88 is tied to one person's grading.

**Link to Assignment 1.** Our own pipeline asks forward-looking questions (will the sprint spill over or close late) and gets AUC .885 for spillover at the start of the sprint and .957 at its midpoint. That supports the paper's broad idea that watching the work helps, but only when the target is an outcome that has not happened yet, not a grade of a finished chart.

## 4. Next steps

Ask the authors for the 551 labelled sprints and rerun E1 to E3 with project-grouped folds. Have two people label a sample of TAWOS sprints with the paper's four classes to measure agreement. Replace the end-of-sprint grade with a forward target (Poor at the end, predicted at 25% or 50%) and compare with our Phase 3 early-warning model, which is the part of our course project this critique feeds.

## 5. AI-use statement

We used Claude (Anthropic) as a coding and writing assistant: it helped write and run the scripts in the repository and drafted parts of this report. We chose the paper, checked every quoted number against the published article, fixed the experiment plan and the feature rules before running, and reviewed all code and text. All results in Table 1 apart from the paper's row come from our own runs.
