# Can a Burndown Chart Grade a Sprint? A Reality Check on Open-Source Jira Data

Hani AbuSharkh, CSBP711 Advanced AI, Fall 2026. Code: https://github.com/hani-ziad/AdvAI_critique

## 1. Summary and positioning

The paper is Pérez Castillo, Orantes Jiménez and Letelier Torres, "Sprint Management in Agile Approach: Progress and Velocity Evaluation Applying Machine Learning," *Information* 15(11):726, 2024.

It starts from a real problem. A manager who runs several projects at once can't read every sprint's charts, so the authors want a model that points at the sprints whose progress looks bad. They take two charts per sprint from the Worki tool, the burndown and the "completed vs not completed work" chart, and reduce them to 11 numbers (R1 to R11). These cover how much work got done, how often and for how long progress stalled, days when remaining effort didn't really drop, and sudden drops or jumps. An expert then grades each of 551 student sprints as Excellent, Good, Fair or Poor, and seven classifiers are compared with cross-validation.

The headline claim is that SVM, LR and RF grade sprints with about .87 to .88 accuracy (Table 3), and that machine learning "demonstrates a clear advantage over traditional evaluation methods".

I picked this paper because it is the closest published work to my course project. That project predicts sprint outcomes on TAWOS (eight open-source Jira projects) from the sprint state, which I rebuild from the Jira change log. Its third phase watches the sprint at 25%, 50% and 75% of the way through, which is the same idea as reading a burndown chart.

## 2. Critical analysis

There is a lot to like. The problem matters, the features are easy to explain to a Scrum Master, and the paper prints the full confusion matrix (Table 4), so a reader can check its numbers. That is how I found most of the issues below.

**The grade and the features come from the same charts.** The expert helped choose the features "based on how progress is visualized in the charts", then graded each sprint by looking at those same charts. So a high score shows that a model can copy one person's reading of a chart. It doesn't show that the model can see anything coming. The paper never checks how much of the grade is just R1, the share of work completed.

**The timing doesn't match the promise.** The paper sells the tool for spotting sprints "that may require immediate intervention". But R1 is the work completed "by the end of the Sprint", and by then there is nothing left to rescue. No result is given for a model that only sees part of a sprint.

**There is no baseline.** Table 3 compares seven classifiers with each other and with nothing else. Always guessing the most common grade would score .303 (from Table 4), and there is no simple one-feature rule and no "traditional method", even though the third research question claims ML beats them. SVM, LR and RF are within .01 of each other with no intervals, yet SVM is picked for deployment.

**The metrics hide the weak class.** In every row of Table 3, "Recall" equals "Accuracy". That is what weighted-average recall always gives, so the column tells us nothing new. Table 4 shows the real picture: Fair has recall .62 and F1 .68, while every other grade is above .88. Specificity is promised in Section 4.6 but never reported, and the text misreads Table 4 once (it calls five Good sprints predicted as Fair "Fair misclassified as Good").

**The evaluation and data are thin.** We aren't told how many folds were used or whether a team's sprints were kept together. With about three sprints per team, random folds will put the same team on both sides of the split. The sprints come from student projects, graded by one expert with no second opinion and no grading guide. The data and code are private and the API can't even be named, so nobody can repeat the study. The paper also went from submission to acceptance in about five weeks.

Overall, the evidence supports a narrow claim: on these student sprints, a model can reproduce one expert's end-of-sprint grade. It doesn't support the bigger claims about early intervention or beating traditional methods.

## 3. Reality check

I couldn't get the paper's data or code, so I built the closest version I could on TAWOS, using my Assignment 1 setup: the same 1,691 sprints with committed work, the same per-project 70/30 time split and seed 42. For each sprint I rebuilt a day-by-day burndown from the change log (24,973 sprint-days) and computed my own versions of R1 to R11 on issue counts. I wrote the rules down before running any model. TAWOS has no expert grades, so my grade is a fixed rule that reads the chart the way the expert did: half pace (how closely the done share kept up with the time used) and half final completion, cut into four grades at the paper's class sizes. As a check I also tried a "pace only" grade. The seven classifiers use library defaults.

**Table 1. Accuracy. The paper's numbers are quoted from its Table 3; every other number comes from my own runs.**

| Setting | Best of 7 models | LR on R1 only | Most common grade |
|---|---|---|---|
| Paper: 551 student sprints, expert grade, CV | .88 (SVM) | not reported | .30 (from Table 4) |
| Mine, main grade: random 10-fold CV (paper's setup) | .774 (RF) | .774 | .297 |
| Mine, main grade: time split, end of sprint | .748 (RF) [.708, .787] | .751 | .284 |
| Mine, main grade: leave one project out | .764 (LR) | .770 | .154 |
| Mine, main grade: time split, first 50% of sprint | .585 [.540, .624] | .597 | .284 |
| Mine, main grade: time split, first 25% of sprint | .507 [.464, .552] | .474 | .284 |
| Mine, pace-only grade: random 10-fold CV | .661 (RF) | .624 | .288 |
| Mine, pace-only grade: time split, first 25% | .587 (MLP) | .550 | .254 |
| My course pipeline (different task, AUC): spillover at sprint start / at 50% | .885 / .957 | | |
| My course pipeline (different task, AUC): late closure at sprint start | .751 | | |

Brackets are 95% bootstrap intervals over sprints.

**What explains the gaps.**

1. *My best score is lower (.77 against .88).* The data are different (open-source teams, sprints of about two weeks, issue counts instead of hours) and so is the grade (a rule instead of a person). A more lenient grader would push scores up. I can't close this gap without their data.
2. *One feature does nearly all the work.* With the main grade, "% of work completed" on its own matches all seven models (.774 against .774). With the pace-only grade, the other ten features add .04 under random CV and .09 under the time split. On end-of-sprint data the models are mostly reading back how much got finished, which is exactly the problem in the first weakness above.
3. *Random folds didn't inflate my scores much* (.774 against .748 on the time split and .764 across projects). My grade is the same rule for every project, so there is no team-specific grading to leak. The paper may be different, since one expert graded about three sprints per student team, but I couldn't test that.
4. *As an early warning it is much weaker.* Seeing only the first 25% of the sprint, accuracy drops to .51, though that is still well above .28 for guessing. Poor sprints are still caught (recall about .89), but with a lot of false alarms: precision for Poor falls from .89 at the end of the sprint to .54. This is the use the paper promises, and Table 3 doesn't measure it.
5. *How you define the grade moves the result by more than .10* (.774 against .661 under the same setup). The paper doesn't say how its expert decided, so its .88 depends on one person's grading.

**How this connects to Assignment 1.** My own pipeline asks forward-looking questions: will the sprint spill over, or close late? It gets AUC .885 for spillover at the start of a sprint and .957 at the midpoint. That backs the paper's general idea that watching the work helps, but only when you predict something that hasn't happened yet, not when you grade a chart that is already finished.

## 4. Next steps

I would ask the authors for the 551 graded sprints and rerun everything with folds that keep each team together. I'd also have two people grade a sample of TAWOS sprints with the paper's four grades to see how often they agree. Most of all, I'd turn the end-of-sprint grade into a real early warning (predict at 25% or 50% whether the sprint will end Poor) and compare it with my Phase 3 model. That is what my final project does.

## 5. AI-use statement

I used Claude (Anthropic) as a coding and writing assistant. It helped write and run the scripts in the repository and drafted parts of this report. I chose the paper, checked every quoted number against the published article, fixed the experiment plan and feature rules before running anything, and reviewed all code and text. Apart from the paper's row, every result in Table 1 comes from my own runs.
