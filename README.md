Salary Prediction MLOps Project

Project Overview
This project aims to develop a predictive model to estimate salaries based on years of experience using multiple regression techniques. The project includes training various models, tracking experiments with MLflow, deploying the best-performing model, and monitoring it in production using WhyLabs. The goal is to provide accurate salary predictions while ensuring model reliability and transparency throughout its lifecycle.

Project Structure
bash
Copy code
salary_prediction_mlops_project/
│
├── pipeline/                      # Folder for pipeline-related code
│   ├── __init__.py                # Empty file to mark the directory as a Python package
│   ├── data_preprocessing.py      # Module for loading and preprocessing data
│   ├── model_training.py          # Module for training models and tracking experiments with MLflow
│   ├── model_deployment.py        # Module for deploying the best model
│   └── model_monitoring.py        # Module for monitoring the model using WhyLabs
│
├── resources/                     # Folder for project resources
│   └── salary_data.csv            # CSV file containing the salary data
│
├── main.py                        # Main script to integrate all modules and execute the workflow
├── requirements.txt               # List of required libraries for the project
└── README.md                      # Project documentation and instructions
Installation
To set up the project environment, follow these steps:

Clone the Repository

bash
Copy code
git clone https://github.com/your-username/salary-prediction-mlops.git
cd salary-prediction-mlops-project
Create and Activate a Virtual Environment

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install Dependencies

bash
Copy code
pip install -r requirements.txt
Dependencies
pandas: For data manipulation and analysis.
scikit-learn: For machine learning models and metrics.
mlflow: For managing the machine learning lifecycle.
whylabs: For model monitoring.
numpy: For numerical operations.
matplotlib (optional): For visualizations.
Usage
Data Preparation
Ensure that the dataset (salary_data.csv) is placed in the resources/ directory.

Running the Workflow
Preprocessing Data

The data_preprocessing.py module is used to load and preprocess the data. It splits the data into training and test sets.

Training Models

The model_training.py module trains three types of regression models:

Linear Regression
Random Forest Regressor
Decision Tree Regressor
It logs experiment details using MLflow, including parameters and metrics.

Deploying the Best Model

The model_deployment.py module handles the deployment of the best-performing model. It saves the model to disk for future use.

Monitoring the Model

The model_monitoring.py module integrates with WhyLabs to monitor the model's performance in production, tracking metrics and detecting anomalies.

To run the entire workflow, execute the following command:

bash
Copy code
python main.py
Configuration
MLflow: Ensure that MLflow is properly configured to log experiments and models. You can set the tracking URI and other configurations as needed.
WhyLabs: Configure the WhyLabs client with your API key in the model_monitoring.py module.
Example Output
After running the workflow, you should see:

Training results for each model, including MSE values.
The best model saved to the specified path.
Performance metrics and monitoring data sent to WhyLabs.
Troubleshooting
Error Loading Data: Ensure that salary_data.csv is correctly placed in the resources/ directory and is properly formatted.
MLflow Issues: Verify that MLflow is correctly set up and that you have permissions to log experiments.
WhyLabs Configuration: Double-check your WhyLabs API key and client configuration.