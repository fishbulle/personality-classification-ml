from data_processing import DataProcessor
from model_training import ModelTrainer


def main():
    """
    Trains and saves the personality classification model.

    Workflow:
    - Load dataset
    - Preprocess data
    - Train model
    - Save the model
    """

    # Initialize components
    dp = DataProcessor()
    mt = ModelTrainer()

    print("Loading data...")
    df = dp.load_data("data/personality_dataset.csv")

    print("Processing data...")
    X, y = dp.preprocess(df)

    print("Training model...")
    mt.train(X, y)

    print("Saving model...")
    mt.save_model()

    print("Training complete and model saved.")


if __name__ == "__main__":
    main()
