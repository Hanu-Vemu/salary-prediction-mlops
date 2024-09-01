import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the salary data.
    
    Parameters:
        file_path (str): Path to the CSV file containing salary data.
    
    Returns:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train (pd.Series): Training target set.
        y_test (pd.Series): Test target set.
    """
    try:
        # Load dataset
        data = pd.read_csv(file_path)

        # Prepare features and target
        X = data[['YearsExperience']] 
        y = data[['Salary']]

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    
    except Exception as Error:
        print(f"Error loading or preprocessing data: {Error}")
        raise