from pipeline.data_preprocessing import load_and_preprocess_data
from pipeline.model_training import train_and_evaluate_models
from pipeline.model_deployment import load_best_model, save_model_to_disk
from pipeline.model_monitoring import configure_whylabs, monitor_model
from dotenv import load_dotenv
import os

# WHYLABS_API_KEY = os.getenv('WHYLABS_API_KEY')
# WHYLABS_DEFAULT_ORG_ID = os.getenv('WHYLABS_DEFAULT_ORG_ID')
# WHYLABS_DEFAULT_DATASET_ID = os.getenv('WHYLABS_DEFAULT_DATASET_ID')

def predict():
    """
    Predict function to execute the salary prediction workflow.
    """
    try:
        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv('WHYLABS_API_KEY')

        # Load and preprocess data
        data_path = 'resources/capstone_project_dataset.csv'
        X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

        # Train models and evaluate
        results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
        
        # Print results
        for model_name, mse in results.items():
            print('{} - MSE: {}'.format(model_name, mse))

        # Select the best model based on the lowest MSE
        best_model_name = min(results, key=results.get)
        print("The best model is: {}".format(best_model_name))

        # Load the best model
        try:
            best_model = load_best_model(best_model_name)
            
            # Save the model to disk
            save_model_to_disk(best_model, 'model_deployment')

            # Monitor model performance
            y_pred = best_model.predict(X_test)
            monitor_model(y_pred, y_test)
            
        except Exception as e:
            print("Error in model loading or monitoring: {}".format(e))

    except Exception as e:
        print("An error occurred during the main execution: {}".format(e))


if __name__ == '__main__':
    predict()
