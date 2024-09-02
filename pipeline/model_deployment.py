import mlflow.sklearn
import joblib
import os

def load_best_model(model_name):
    """
    Load the best-performing model from the saved model files on disk.
    
    Parameters:
        model_name (str): The name of the model to be loaded.
    
    Returns:
        model: The loaded model object.
    """
    try:
        # Replace spaces with underscores to match saved file naming
        model_path = f'model_deployment/{model_name.replace(" ", "_")}.pkl'
        print(f'model_path: {model_path}')
        model = joblib.load(model_path)
        print(f"Model {model_name} loaded successfully from {model_path}")
        return model
    except FileNotFoundError as e:
        print(f"Error loading model {model_name}: {e}")
        raise
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise

def save_model_to_disk(model, directory):
    """
    Save the trained model to disk using joblib.
    
    Parameters:
        model: The trained model to be saved.
        directory (str): The directory where the model will be saved.
    
    Returns:
        str: The path to the saved model file.
    """
    try:
        # Create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the model as 'best_model.pkl' in the specified directory
        model_filename = os.path.join(directory, 'best_model.pkl')
        joblib.dump(model, model_filename)
        print(f"Model saved successfully at {model_filename}")
        return model_filename
    except Exception as e:
        print(f"Error saving model to disk at {directory}: {e}")
        raise
