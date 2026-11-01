# Notes on the paper (checked against the published HTML on 1 Nov 2026)

Pérez Castillo, Orantes Jiménez, Letelier Torres. *Information* 15(11):726, 2024. Received 25 Sep 2024, accepted 3 Nov 2024, published 12 Nov 2024 (about 5 weeks from submission to acceptance). Special issue "Technology, Learning and Teaching of Electronics with Information Applications".

## Data
- Worki tool (TUNE-UP Process, Universitat Politècnica de València), software engineering course projects built by students.
- About 200 projects, about 3 sprints each; 551 sprints after extraction.
- Data and code private ("available on request"; API access "cannot be mentioned in the document").
- Burndown in hours of "estimable" activities; completed vs not completed work chart in hours of user stories.

## Features (Table 1), defined in words only, no thresholds given
R1 % work completed by the end of the sprint; R2 any period with no new completed work; R3 number of such periods; R4 % of days stalled; R5 days without progress; R6 no significant reduction in remaining effort; R7 count of such days; R8 excessive reduction; R9 count; R10 significant increase; R11 count.

## Label
TAG = Excellent 4, Good 3, Fair 2, Poor 1, "labeled with the aid of an expert". One expert, no agreement measure, no labelling guide. Features were also chosen "with the help of an expert ... based on how progress is visualized in the charts".
Table 2 example rows: R1 = 100 with TAG Fair (2) twice, so the label is not R1 alone.

## Models and evaluation
KNN, SVM, MLP, RF, NB, LR, AdaBoost. "Cross-validation was employed"; number of folds, grouping, scaling, hyperparameters not given. Specificity is promised in Section 4.6 but not reported.

## Table 3 (Recall, F1, Accuracy, Precision)
| Model | Recall | F1 | Accuracy | Precision |
|---|---|---|---|---|
| KNN | .86 | .86 | .86 | .87 |
| SVM | .88 | .87 | .88 | .88 |
| MLP | .86 | .86 | .86 | .87 |
| RF | .87 | .86 | .87 | .87 |
| NB | .69 | .66 | .69 | .78 |
| LR | .87 | .87 | .87 | .88 |
| AB | .77 | .75 | .77 | .75 |
Recall equals Accuracy in every row, which is what weighted-average recall always gives.

## Table 4, SVM confusion matrix (rows = true, columns = predicted: Poor, Fair, Good, Excellent)
| | Poor | Fair | Good | Excellent |
|---|---|---|---|---|
| Poor | 129 | 15 | 2 | 0 |
| Fair | 16 | 62 | 21 | 1 |
| Good | 0 | 5 | 158 | 4 |
| Excellent | 0 | 0 | 4 | 134 |
Sums to 551, so it pools all cross-validation folds. Class sizes 146 / 100 / 167 / 138, so always guessing Good scores .303.
Per class (my arithmetic from Table 4): recall .884 / .620 / .946 / .971; precision .890 / .756 / .854 / .964; F1 .887 / .681 / .898 / .968; macro F1 .858; accuracy .877.
Small error in the text: it says "five instances labeled as Fair were misclassified as Good", but the 5 sit in the Good row, Fair column (Good predicted as Fair).

## Claims vs evidence
- RQ3: "ML models ... demonstrate a clear advantage over traditional evaluation methods". No traditional method or baseline is run.
- Motivation: detect sprints "that may require immediate intervention". All features are measured at the end of the sprint.
- Discussion: SVM "is considered an excellent candidate" for deployment. SVM, LR and RF are within .01, with no intervals or tests.
