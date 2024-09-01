from pipeline.data_preprocessing import load_and_preprocess_data
from pipeline.model_training import train_and_evaluate_models
from pipeline.model_deployment import load_best_model, save_model_to_disk
from pipeline.model_monitoring import configure_whylabs, monitor_model


def predict():
    """
    Predict function to execute the salary prediction workflow.
    """
    try:
        # Load and preprocess data
        data_path = 'resources/salary_data.csv'
        X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

        # Train models and evaluate
        results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
        
        # Print results
        for model_name, mse in results.items():
            print(f'{model_name} - MSE: {mse}')

        # Select the best model based on the lowest MSE
        best_model_name = min(results, key=results.get)
        print(f"The best model is: {best_model_name}")

        # Load the best model
        best_model = load_best_model(best_model_name)

        # Save the model to disk
        save_model_to_disk(best_model, 'model deployment')

        # Monitor model performance
        whylabs_client = configure_whylabs('YOUR_API_KEY')
        y_pred = best_model.predict(X_test)
        monitor_model(whylabs_client, y_pred, y_test)

    except Exception as e:
        print(f"An error occurred during the main execution: {e}")


if __name__=='__main__':
    predict()