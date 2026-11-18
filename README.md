# Algeria Gas Flaring Analysis Using Golang

## Project Overview

This project investigates gas flaring activities in Algeria using Golang in Google Colab.

**Module:** Next Generation Data Science

**Dataset:** NGDS_CO2_Algeria.csv

## Project Methodology

- Data loading and inspection
- Data cleaning and preprocessing
- Exploratory data analysis
- Geographical analysis
- Feature engineering
- Machine learning regression
- Model evaluation
- Explainable Artificial Intelligence

## Source Code

| File | Purpose |
|---|---|
| environment_check.go | Environment verification |
| data_inspection.go | Initial dataset inspection |
| data_cleaning.go | Data cleaning and preprocessing |
| eda_analysis.go | Exploratory data analysis |
| geographical_analysis.go | Geographical analysis |
| feature_engineering.go | Feature engineering and data splitting |
| regression_models.go | Regression model development |
| model_evaluation.go | Model performance evaluation |
| explainable_ai.go | Explainable AI analysis |

## Processed Datasets

- cleaned_algeria.csv – 2,592 observations
- train_data.csv – 2,073 observations
- test_data.csv – 519 observations

The processed datasets are submitted separately through Blackboard.

## Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Multiple Linear Regression | 38.5287 | 50.6062 | 0.0185 |
| Decision Tree | 34.7118 | 48.1177 | 0.1127 |
| Random Forest | 34.2643 | 46.3606 | 0.1763 |

Random Forest achieved the best comparative performance. However, its overall predictive capability remains limited.

## Explainable AI

- Permutation Feature Importance
- Partial Dependence Plots
- Approximate Shapley Values

Latitude and Longitude were identified as the most influential predictors.

## Technologies Used

- Golang
- Google Colab
- Gota
- Gonum
- Gonum Plot
- Go-ECharts
- Orb

## Execution Instructions

Install the required Go dependencies using:

`go mod download`

Execute each Go source file individually using the following format:

`go run data_inspection.go`

Run the analytical programs in the methodology order. The required dataset, datasets directory and generated intermediate files must be available before executing the relevant programs.

## AI Usage Declaration

ChatGPT was used to support code generation, analytical methodology, implementation and documentation. All numerical findings were obtained from actual program execution using the assigned dataset.
