import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

def train_and_save_model():
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "dataset" / "processed" / "Processed CAR DETAILS.csv"
    
    df = pd.read_csv(data_path)
    category_cols = ['Brand', 'Model', 'Variant', 'Fuel', 'Seller_Type', 'Transmission', 'Owner']
    
    # Label Encoding for categorical columns
    label_encoders = {}
    for col in category_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        # Save encoders back to directory
        with open(base_dir / f"models/{col}_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

    # Separate features and target
    X = df.drop(columns=['Selling_Price'])
    y = df['Selling_Price']

    # Standard scale numerical columns
    scaler = StandardScaler()
    numerical_cols = ['Year', 'Km_Driven']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save trained model to both required paths
    paths = [
        base_dir / "models" / "Random Forest.pkl",
        base_dir / "streamlit-app" / "Random Forest.pkl"
    ]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(model, f)

    print("Model and Encoders successfully trained and saved with scikit-learn 1.5.2!")

if __name__ == "__main__":
    train_and_save_model()
