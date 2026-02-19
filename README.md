# Predictive Maintenance: Machine Failure Classification

Using sensor data from manufacturing equipment, this project builds and evaluates multiple classification models to predict machine failures before they occur — enabling proactive maintenance and reducing downtime.

## Problem Statement

Unplanned manufacturing downtime costs an estimated $50B annually, with individual incidents running $10,000–$250,000 per hour. Beyond the financial impact, unexpected failures introduce safety risks and productivity losses. The goal of this project is to predict machine failures early so maintenance teams can act before a breakdown occurs.

## Dataset & Set Up

[AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) via Kaggle.

Download the dataset and place `ai4i2020.csv` in the same directory as the notebook before running.

**10,000 rows** of sensor readings with the following key features:

- `rotational_speed_[rpm]` — machine rotation speed
- `torque_[nm]` — torque applied
- `tool_wear_[min]` — tool wear over time
- `type` — machine quality variant (L, M, H)
- Failure flags: TWF, HDF, PWF, OSF, RNF

## Approach

### Target Variable
Rather than predicting binary machine failure, the model predicts `primary_failure` — a multi-class target (TWF, OSF, PWF, HDF, RNF, Unknown, No Failure) created by prioritizing failure types when multiple occur simultaneously.

### Feature Engineering
Three features were engineered from raw sensor data:
- `overstrain_metric` = tool wear × torque (detects OSF conditions)
- `temp_diff` = process temperature − air temperature (detects HDF conditions)
- `power_estimate` = torque × rotational speed (detects PWF conditions)

### Model Evaluation
Five models were compared using weighted recall and stratified 5-fold cross-validation:

| Model                 | CV Mean Recall |
| --------------------- | -------------- |
| KNN                   | 0.9749         |
| Logistic Regression   | 0.9785         |
| Decision Tree         | 0.9849         |
| Random Forest         | 0.9907         |
| Bagged Decision Trees | 0.9905         |

**Random Forest** was selected as the final model based on highest performance and interpretability.

## Results

The final Random Forest model was evaluated on its ability to detect actual failures (excluding "No Failure"):

- **74% failure recall** — correctly identifies approximately 3 out of 4 actual machine failures
- **Only 2 false alarms** out of 9,643 non-failure cases
- Top predictive features: `overstrain_metric` and `power_estimate`

| | Predicted Failure | Predicted No Failure |
|---|---|---|
| **Actual Failure** | 266 | 91 |
| **Actual No Failure** | 2 | 9,641 |

## Tools & Libraries

- Python, Jupyter Notebook
- pandas, scikit-learn, matplotlib
- Models: KNN, Logistic Regression, Decision Tree, Random Forest, Bagged Decision Trees

## Next Steps

- Address class imbalance — the dataset is heavily skewed toward "No Failure"
- Extend the model to predict multiple simultaneous failures per machine
- Integrate predictions into maintenance dashboards with early warning alerts
- Recalibrate model as new sensor data becomes available
