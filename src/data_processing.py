import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataProcessor:
    """
    Handles loading and preprocessing of the dataset.
    Includes cleaning, encoding categorical variables, and scaling features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_data(self, path):
        return pd.read_csv(path)

    def preprocess(self, df):
        # Remove duplicate rows
        df = df.drop_duplicates()

        # Encode binary categorical variables
        categorical_cols = ["Stage_fear", "Drained_after_socializing"]
        binary_map = {"Yes": 1, "No": 0}

        for col in categorical_cols:
            df[col] = df[col].map(binary_map)

        # Target encoding (Extrovert=1, Introvert=0)
        df["Personality"] = self.label_encoder.fit_transform(df["Personality"])

        X = df.drop("Personality", axis=1)
        y = df["Personality"]

        # Data scaling
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y
