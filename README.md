# Salary Prediction MLOps Project

## Project Overview

This project aims to develop a predictive model that estimates salaries using years of experience as the primary predictor. It includes the implementation of multiple regression models, model tracking and management with MLflow, and model monitoring with WhyLogs.

## Project Structure

```bash
salary_prediction_mlop/
│
├── pipeline/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_deployment.py
│   ├── model_monitoring.py
│
├── resources/
│   │   └── capstone_project_dataset.csv
│   ├── config.ini
│
├── main.py
├── requirements.txt
└── README.md
