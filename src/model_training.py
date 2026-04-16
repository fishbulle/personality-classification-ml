from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import joblib


class ModelTrainer:
    """
    Handles training, evaluation, saving, loading, and prediction
    for the machine learning model.
    """

    def __init__(self):
        """
        Initializes the model.
        Logistic Regression is used for binary classification.
        """
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        """
        Trains the model and evaluates its performance.

        Steps:
        - Splits data into training and test sets
        - Fits the model on training data
        - Generates predictions on test data
        - Evaluates performance using multiple metrics
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        print("\n--- Model Evaluation ---")
        print("Accuracy:", accuracy_score(y_test, preds))
        print("F1 Score:", f1_score(y_test, preds))
        print("ROC AUC:", roc_auc_score(y_test, probs))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
        print("\nClassification Report:\n", classification_report(y_test, preds))

    def save_model(self, path="model.pkl"):
        """
        Saves the trained model to a file.

        Args:
            path (str): File path where the model will be saved
        """
        joblib.dump(self.model, path)

    def load_model(self, path="model.pkl"):
        """
        Loads a previously saved model from a file.

        Args:
            path (str): File path to the saved model
        """
        self.model = joblib.load(path)

    def predict(self, X):
        """
        Makes predictions using the trained model.

        Args:
            X (array-like): Input data (must be preprocessed)

        Returns:
            array: Predicted class labels
        """
        return self.model.predict(X)
