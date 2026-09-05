AI4Health Competition 2024 — Hospital Readmission Risk Prediction

A competition project (Care4You team) predicting early hospital readmission risk from patient discharge data — ~28,500 patient records and 275 raw features — culminating in a CatBoost model tuned for Matthews Correlation Coefficient (MCC) with a custom decision threshold.

Problem

Given patient demographic, clinical, and discharge information, predict whether a patient will be readmitted early (binary label). The dataset is realistically messy: missing values, encoded categorical fields, physiologically implausible outliers, and — critically — significant class imbalance, since early readmissions are the minority outcome.

Approach
1. Data cleaning
Imputed missing values in key categorical fields (AdmissionDx, Education, Current_Work_Status) using mode imputation
Dropped a leftover index column and checked for a sentinel missing-value code (-8) across all columns
Removed exact duplicate rows
Removed physiologically implausible records (e.g., discharge height > 220cm, discharge weight > 200kg) — outliers likely reflecting data entry errors rather than real patients
2. Categorical encoding
Manually mapped ordinal/categorical fields to integers: admission diagnosis category (A–Z), age group bands, discharge specialty, gender, and admission type (emergency vs. elective)
Applied identical mappings to both the train and test sets to keep encodings consistent
3. Feature scaling and class imbalance
Standardized features with StandardScaler
Addressed class imbalance with RandomOverSampler, balancing the training set from a skewed split to an even 26,680 / 26,680 (majority/minority class)
4. Modeling
CatBoostClassifier, tuned with custom class weights to adjust the model's treatment of the two classes
Evaluated using Matthews Correlation Coefficient (MCC) rather than plain accuracy — a more appropriate metric than accuracy for imbalanced binary classification, since MCC accounts for all four confusion-matrix quadrants
Selected a custom decision threshold (0.21), well below the default 0.5, chosen empirically based on submission results rather than the model's default cutoff
5. Submission
Generated predictions and class probabilities for the held-out test set (7,336 patients)
Wrote results to a competition submission file in the required format (Label,Probability_0,Probability_1)
Iteration history

This wasn't a single-shot solution — it was refined across roughly a week of competition submissions (Care4You_1 through Care4You_8), trying different model families and evaluation strategies before converging on the final approach:

Submission	Model	Key change
Care4You_1	MLPClassifier (neural network)	First working baseline, simple dropna cleaning
—	CatBoostClassifier	Switched to gradient boosting, added L2 regularization, adopted MCC as the eval metric instead of accuracy
Care4You_5	LightGBM	Tried a second gradient boosting library for comparison
Care4You_6	LightGBM + GridSearchCV	Systematic hyperparameter search (boosting type, leaves, learning rate, estimators) scored directly against MCC
Care4You_7	LightGBM + L1/L2 regularization	Added cross-validated ROC AUC tracking and introduced custom decision threshold tuning (0.4) instead of the default 0.5
Care4You_8 (final)	CatBoostClassifier, tuned params + class weights	Returned to CatBoost with refined hyperparameters, custom class weights, and a further-tuned threshold of 0.21
Results
Final test set: 7,336 patients, 273 features after cleaning
At the tuned threshold of 0.21, the model flagged 2,136 of 7,336 patients (~29.1%) as high readmission risk
Tech stack

Python · pandas · scikit-learn · imbalanced-learn (RandomOverSampler) · CatBoost

Notes

This was a competition setting with a fixed evaluation metric (MCC) and submission format, so several choices here — the specific class weights, the 0.21 threshold, the oversampling ratio — were tuned empirically against the competition's scoring rather than derived from first principles. The project demonstrates an end-to-end workflow for imbalanced tabular classification, from data cleaning and feature preparation to model comparison and decision-threshold tuning.
