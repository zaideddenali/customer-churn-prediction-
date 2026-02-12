import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def build_features(df):
    """Encodes categorical variables and scales numerical features."""
    df = df.copy()
    
    # Drop columns that are not useful for prediction
    cols_to_drop = ['Phone', 'State'] # Phone is ID-like, State might be too granular for basic model
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Encode categorical variables
    binary_cols = ["Int'l Plan", "VMail Plan"]
    for col in binary_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            
    # Target variable is already handled in cleaning
    
    # Split features and target
    X = df.drop(columns=['Churn?'])
    y = df['Churn?']
    
    return X, y

def scale_features(X_train, X_test):
    """Scales numerical features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    processed_data_path = "data/processed/cleaned_churn.csv"
    if not os.path.exists(processed_data_path):
        print("Processed data not found. Run make_dataset.py first.")
    else:
        df = pd.read_csv(processed_data_path)
        X, y = build_features(df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
        
        # Save split data
        pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("data/processed/X_train.csv", index=False)
        pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("data/processed/X_test.csv", index=False)
        y_train.to_csv("data/processed/y_train.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)
        
        print("Feature engineering complete. Split data saved to data/processed/")
