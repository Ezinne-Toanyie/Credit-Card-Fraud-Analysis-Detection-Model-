# Credit Card Fraud Detection Using Isolation Forest

An unsupervised anomaly detection project that flags potentially fraudulent credit card transactions using an Isolation Forest, without relying on fraud labels during training.

## Overview

Fraud is rare, costly and constantly evolving, and confirmed fraud labels are scarce. Isolation Forest learns what "normal" transactions look like and flags those that deviate, which suits highly imbalanced, real-world transaction data. This project builds, trains and honestly evaluates that approach.

## Dataset

| Property | Detail |
|---|---|
| Transactions | 1,296,675 |
| Fraud rate | 0.58% (7,506 fraudulent) |
| Modelling features | 12 |
| Target | `is_fraud` (used for evaluation only) |

Fields cover the transaction (`amt`, `hour`), the cardholder (`age`, `gender`, `job`, `city`, `state`, `city_pop`), locations (cardholder and merchant lat/long) and merchant `category`.

**Source:** [Credit Card Transactions Fraud Detection Dataset (Kaggle, by Kartik Shenoy)](https://www.kaggle.com/datasets/kartik2112/fraud-detection). This is a simulated dataset covering January 2019 to December 2020.

## Approach

- **No resampling.** No SMOTE, oversampling or undersampling. The natural 0.58% fraud rate was preserved so the model can learn what "normal" looks like.
- **Feature engineering.** Added `distance_km`, the Haversine distance between cardholder and merchant, since fraud often occurs far from a cardholder's usual activity.
- **Encoding and scaling.** Label encoding for `city`, `state`, `job` and `gender`; one-hot encoding for merchant `category`; `StandardScaler` on all features.
- **Model.** `IsolationForest` with `n_estimators=100`, `contamination=0.5789%` (set from the real fraud rate) and `random_state=42`.
- **Split.** Stratified 80/20 train/test split to keep the fraud ratio in both sets.
- **Output mapping.** Raw predictions (1 = inlier, -1 = outlier) remapped to 0 (legitimate) and 1 (fraud).

## Results

Evaluated on 259,335 held-out transactions:

| | Predicted normal | Predicted fraud |
|---|---|---|
| **Actual normal** | 256,454 | 1,380 |
| **Actual fraud** | 1,420 | 81 |

| Fraud class | Score |
|---|---|
| Precision | 0.06 |
| Recall | 0.05 |
| F1-score | 0.05 |

Overall accuracy is 99%, but that is misleading: about 99.4% of transactions are legitimate, and the macro average is only 0.52. The model recognises normal behaviour very well but detected only 81 of 1,501 fraud cases, with a high false-alarm rate. Not every fraud case looks statistically unusual in this feature space, which is a known limitation of unsupervised anomaly detection.

## Limitations and Next Steps

- Low recall (5%) means most fraud goes undetected; low precision (6%) means many false alarms.
- Tune the decision threshold and contamination rate to trade off precision against recall.
- Use the Isolation Forest anomaly score as a feature in a supervised model (hybrid approach).
- Benchmark against Local Outlier Factor and autoencoders.
- Add behavioural features beyond distance, such as transaction velocity.

## Tech Stack

Python, pandas, scikit-learn
