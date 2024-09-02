import whylogs as why
from whylogs.extras.image_metric import log_image
from whylogs.api.writer.whylabs import WhyLabsWriter
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone


def configure_whylabs():
    """
    Configure WhyLabs using environment variables.

    Returns:
        WhyLabsWriter: The configured WhyLabs writer.
    """
    try:
        why.init(force_local=True, reinit=True)
        api_key = os.getenv('WHYLABS_API_KEY')
        org_id = os.getenv('WHYLABS_DEFAULT_ORG_ID')
        dataset_id = os.getenv('WHYLABS_DEFAULT_DATASET_ID')

        if not all([api_key, org_id, dataset_id]):
            raise ValueError("Missing environment variables for WhyLabs configuration.")

        # Initialize WhyLabsWriter with the provided API key
        writer = WhyLabsWriter(api_key=api_key, org_id=org_id, dataset_id=dataset_id)
        print("WhyLabs configured successfully.")
        return writer
    except Exception as e:
        print("Error configuring WhyLabs: {}".format(e))
        raise


def monitor_model(y_pred: np.ndarray, y_test: np.ndarray):
    """
    Monitor model performance using WhyLabs.

    Parameters:
        y_pred (np.ndarray): Model predictions.
        y_test (np.ndarray): Actual target values.

    Returns:
        None
    """
    try:
        # Load WhyLabs writer
        writer = configure_whylabs()

        # Ensure y_pred and y_test are numpy arrays
        if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
            y_test = y_test.values

        # Ensure y_pred and y_test are 1-dimensional arrays
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
        if len(y_test.shape) > 1:
            y_test = y_test.flatten()

        # Create a DataFrame for WhyLogs
        df = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test})

        # Create a WhyLogs profile
        profile = why.log(df, dataset_timestamp=datetime.now(tz=timezone.utc)).profile()

        # Log metrics to WhyLabs
        writer.write(profile)

        print("Model monitoring completed successfully.")
    except Exception as e:
        print("Error in model loading or monitoring: {}".format(e))
        raise
