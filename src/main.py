from data_processing import DataProcessor
from model_training import ModelTrainer
from app import App


def main():
    """
    Main entry point of the application.

    Workflow:
    - Load dataset
    - Preprocess data
    - Train model
    - Save and reload model
    - Start the user interface
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

    print("Loading model...")
    mt.load_model()

    print("\nStarting application...")
    app = App(mt, dp)
    app.run()


if __name__ == "__main__":
    main()
