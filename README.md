# ai-construction-risk-optimizer

🏗 AI Construction Risk Optimizer
📌 Overview

AI-driven system for predicting financial risk and profitability of small-scale residential construction projects.

The project simulates a small development company building 5–10 houses per year (7–12M RUB per house) and uses machine learning to:

Predict project margin

Detect budget overrun risk

Explain model decisions

Simulate business scenarios (What-If analysis)

This project represents a prototype of a future AI product for small construction businesses.

🎯 Business Problem

Small developers often face:

Budget overruns

Delays due to suppliers or logistics

Low transparency in financial forecasting

High exposure to market volatility

This system provides predictive analytics and risk modeling to optimize decisions before construction starts.

📊 Dataset

Synthetic dataset of 6000 projects including:

Land cost and district class

House area and design complexity

Crew experience and workload

Supplier reliability and delivery distance

Seasonality and market indices

Mortgage rate and demand index

Targets:

actual_margin

budget_overrun

final_profit

delay_days

🤖 Models
1️⃣ Margin Prediction (Regression)

Model: XGBoost Regressor
Metric: MAE / RMSE

2️⃣ Budget Overrun Risk (Classification)

Model: XGBoost Classifier
Metric: ROC-AUC

🔎 Explainability

SHAP values used for:

Global feature importance

Local project-level explanations

Risk transparency for decision-makers

🔁 What-If Simulation

Allows changing:

Crew experience

Delivery distance

Material class

Location class

Returns:

Predicted margin

Probability of budget overrun

🚀 Tech Stack

Python

Pandas / NumPy

XGBoost

SHAP

Plotly

Streamlit

💡 Business Impact

Potential benefits:

Reduction in budget overruns

Improved profitability forecasting

Smarter land acquisition decisions

Operational risk control

🔮 Future Improvements

Real-world dataset integration

Time-series price forecasting

Reinforcement learning for project portfolio optimization

SaaS dashboard for construction firms
