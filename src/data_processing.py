import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataProcessor:
    """
    Handles loading and preprocessing of the dataset,
    including cleaning, encoding, and feature scaling.
    """

    def __init__(self):
        """
        Initializes preprocessing tools:
        - StandardScaler for feature scaling
        - LabelEncoder for target encoding
        """
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_data(self, path):
        """
        Loads dataset from a CSV file.

        Args:
            path (str): File path to dataset

        Returns:
            DataFrame: Loaded dataset
        """
        return pd.read_csv(path)

    def preprocess(self, df):
        """
        Cleans and prepares the dataset for model training.

        Steps:
        - Removes duplicate rows
        - Encodes categorical variables
        - Splits into features and target
        - Scales numerical features
        """

        # Remove duplicate rows to avoid bias
        df = df.drop_duplicates()

        # Convert Yes/No categorical variables to binary (1/0)
        categorical_cols = ["Stage_fear", "Drained_after_socializing"]
        binary_map = {"Yes": 1, "No": 0}

        for col in categorical_cols:
            df[col] = df[col].map(binary_map)

        # Encode target variable (Introvert=0, Extrovert=1)
        df["Personality"] = self.label_encoder.fit_transform(df["Personality"])

        # Split features and target
        X = df.drop("Personality", axis=1)
        y = df["Personality"]

        # Scale features for better model performance
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def transform_input(self, input_data):
        """
        Applies the same scaling transformation to new input data.

        Args:
            input_data (list): Raw user input

        Returns:
            array: Scaled input data
        """
        return self.scaler.transform([input_data])
