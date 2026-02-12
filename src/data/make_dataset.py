import pandas as pd
import os

def load_data(file_path):
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    return pd.read_csv(file_path)

def validate_data(df):
    """Basic data validation."""
    required_columns = ['Churn?']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return True

def clean_data(df):
    """Performs basic data cleaning."""
    # Remove trailing dots from the target column if they exist (based on initial head check)
    if df['Churn?'].dtype == 'object':
        df['Churn?'] = df['Churn?'].str.rstrip('.').map({'True': 1, 'False': 0})
    elif df['Churn?'].dtype == 'bool':
        df['Churn?'] = df['Churn?'].astype(int)
    
    # Basic cleaning: handle missing values if any
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    raw_data_path = "data/raw/customer_churn.csv"
    processed_data_path = "data/processed/cleaned_churn.csv"
    
    print(f"Loading data from {raw_data_path}...")
    df = load_data(raw_data_path)
    
    print("Validating data...")
    validate_data(df)
    
    print("Cleaning data...")
    df_cleaned = clean_data(df)
    
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df_cleaned.to_csv(processed_data_path, index=False)
    print(f"Cleaned data saved to {processed_data_path}")
