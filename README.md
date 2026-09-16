# Gateway Fleet Visit Prioritization

A data-driven predictive maintenance system for prioritising gateway site visits from historical telemetry data.

The system analyses gateway telemetry, identifies patterns associated with operational faults, and produces a ranked prediction of gateways that may require attention.

## Project Overview

Gateway maintenance teams have a limited number of site visits available each week. The objective of this project is to use historical telemetry data to identify gateways with a higher likelihood of operational faults and support maintenance prioritisation.

The project includes:

- Historical telemetry analysis
- Feature engineering from gateway visit history
- Machine learning based fault prediction
- Decision-threshold tuning
- Cross-validation
- Feature importance analysis
- Gateway-level prediction
- Streamlit-based prediction interface

## Machine Learning Approach

A Random Forest classification model is used to estimate the probability of a gateway experiencing an operational fault.

The modelling pipeline includes:

1. Historical telemetry loading
2. Data preprocessing
3. Feature selection
4. Model training
5. 5-fold cross-validation
6. Decision-threshold evaluation
7. Final model training
8. Gateway fault-risk prediction

The final model uses 52 usable features derived from the available telemetry data.

## Model Validation

The final model was evaluated using 5-fold cross-validation.

Final validation results:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8033 |
| Accuracy | 0.7667 |
| Precision | 0.6513 |
| Recall | 0.8534 |
| F1-Score | 0.7388 |
| Decision Threshold | 0.45 |

The selected threshold was 0.45 based on the evaluated threshold comparison.

## Important Features

The most influential features in the final model include:

- `offline_duration_sec_r7d`
- `offline_duration_sec_r7d_sum`
- `read_rate`
- `offline_duration_sec_r28d`
- `offline_duration_sec_r28d_sum`
- `read_rate_4w`
- `rsrp_bad_share_r28d`
- `online_duration_mins_r7d`
- `disconnection_cnt_r7d_sum`

These features indicate that recent connectivity behaviour, offline duration, read activity and disconnection patterns provide useful signals for fault prediction.

## Prediction System

The trained model generates a fault probability for each gateway.

Predictions are classified using the selected decision threshold:

- Probability >= 0.45 → Fault Likely
- Probability < 0.45 → No Fault

Prediction results are stored in:

```text
reports/prediction_results.csv