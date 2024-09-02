import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple regression models and log results using MLflow.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target set.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test target set.

    Returns:
        dict: A dictionary containing model names and their respective MSE values.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(),
        'Decision Tree Regressor': DecisionTreeRegressor()
    }

    mse_scores = {}

    for model_name, model in models.items():
        try:
            # Set the MLflow experiment and tracking URI
            mlflow.set_experiment("Salary Prediction")
            mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

            # Start a new MLflow run
            with mlflow.start_run(run_name=model_name):
                # Train the model
                model.fit(X_train, y_train)

                # Evaluate the model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_scores[model_name] = mse

                # Save the model locally
                model_name = '{}.pkl'.format(model_name.replace(" ", "_"))
                model_filepath = os.path.join('model_deployment', model_name)
                joblib.dump(model, model_filepath)

                # Log parameters, metrics, and model to MLflow
                mlflow.log_param('model_name', model_name)
                mlflow.log_metric('mse', mse)
                mlflow.sklearn.log_model(model, model_name)

        except Exception as e:
            print("Error training or evaluating {}: {}".format(model_name, e))

    return mse_scores
