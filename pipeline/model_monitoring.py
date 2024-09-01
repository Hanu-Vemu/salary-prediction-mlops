import whylabs


def configure_whylabs(api_key):
    """
    Configure the WhyLabs client for model monitoring.
    
    Parameters:
        api_key (str): WhyLabs API key.
    
    Returns:
        whylabs.Client: Configured WhyLabs client.
    """
    try:
        return whylabs.Client(api_key=api_key)
    except Exception as e:
        print(f"Error configuring WhyLabs client: {e}")
        raise


def monitor_model(client, predictions, actuals):
    """
    Monitor the model's performance using WhyLabs.
    
    Parameters:
        client (whylabs.Client): Configured WhyLabs client.
        predictions (array-like): Model predictions.
        actuals (array-like): Actual values.
    """
    try:
        client.log_metrics(
            {
                'predictions': predictions.tolist(),
                'actuals': actuals.tolist()
            }
        )
    except Exception as e:
        print(f"Error monitoring model: {e}")
        raise