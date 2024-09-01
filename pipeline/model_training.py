import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


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

    results = {}

    for name, model in models.items():
        try:
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)

                # Log parameters, metrics, and model
                mlflow.log_params({'model': name})
                mlflow.log_metrics({'mse': mse})
                mlflow.sklearn.log_model(model, name)

                results[name] = mse
        except Exception as e:
            print(f"Error training or evaluating {name}: {e}")
            results[name] = None

    return results