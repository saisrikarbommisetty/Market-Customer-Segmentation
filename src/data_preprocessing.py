import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    scaler = StandardScaler()

    features = [
        "Gender",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    scaled_data = scaler.fit_transform(df[features])

    return df, scaled_data