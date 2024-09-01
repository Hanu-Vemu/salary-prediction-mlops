import mlflow.sklearn


def load_best_model(model_name):
    """
    Load the best-performing model from MLflow.
    
    Parameters:
        model_name (str): The name of the model to be loaded.
    
    Returns:
        model: The loaded model.
    """
    try:
        return mlflow.sklearn.load_model(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise


def save_model_to_disk(model, path):
    """
    Save the model to disk using MLflow.
    
    Parameters:
        model: The model to be saved.
        path (str): Path where the model should be saved.
    """
    try:
        mlflow.pyfunc.save_model(path=path, python_model=model)
    except Exception as e:
        print(f"Error saving model to disk at {path}: {e}")
        raise