# Decisions

## 1. Part 2 area: Data Science

We chose Data Science because the challenge deliberately leaves the operational definition of "needs a visit" open. The goal was to turn that ambiguity into a measurable, defensible decision rule rather than treating the supplied baseline as the final answer.

## 2. Baseline first

We kept the supplied 3-sigma method as our reference point. This provided a simple, reproducible benchmark before evaluating additional evidence from historical telemetry.

## 3. Cost-aware decision making

The challenge charges €380 for an unnecessary visit and €600 per week for leaving a faulty gateway unvisited. We therefore considered operational consequences alongside classification metrics when evaluating the modelling approach.

## 4. Deterministic ranking

Ties are resolved using `gateway_id` after score. This makes repeated runs produce the same ordering.

## 5. Data quality

We measured telemetry coverage before deciding how observations should contribute to the model. Missing telemetry was not silently interpreted as a healthy reading.

## 6. Final modelling approach

Historical gateway telemetry was used to construct additional features describing recent connectivity, activity and fault-related behaviour. A Random Forest classifier was evaluated using 5-fold cross-validation and different decision thresholds.

The final model uses 52 usable features and a decision threshold of 0.45. The final validation produced a ROC-AUC of 0.8033, with accuracy of 0.7667, precision of 0.6513, recall of 0.8534 and F1-score of 0.7388.

## What we cannot claim yet

The final scoring window has a hidden ground truth. Historical field visits and engineer review are useful evidence for investigation, but they are not treated as the official answer key.