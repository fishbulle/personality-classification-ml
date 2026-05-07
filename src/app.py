import streamlit as st
from pathlib import Path

from data_processing import DataProcessor
from model_training import ModelTrainer

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "personality_dataset.csv"


def load_model_and_processor():
    """
    Loads the trained machine learning model and initializes
    the data processor.

    Returns:
        tuple:
            DataProcessor instance for preprocessing user input
            ModelTrainer instance with loaded trained model
    """

    dp = DataProcessor()
    mt = ModelTrainer()

    df = dp.load_data(DATA_PATH)
    dp.preprocess(df)

    mt.load_model()

    return dp, mt


def main():
    """
    Main entry point for the Streamlit web application.

    Workflow:
    - Configure Streamlit page settings
    - Load trained model and processor
    - Collect user behavioral data
    - Preprocess input data
    - Generate personality prediction
    - Display prediction result
    """

    # Configure Streamlit page
    st.set_page_config(
        page_title="Personality Predictor",
        page_icon="🧠",
        layout="centered",
    )

    # Page title and instructions
    st.title("🧠 Personality Predictor")

    st.write(
        "Answer based on relative levels and not exact units. "
        "Higher number means higher frequency."
    )

    # Load model and preprocessing tools
    dp, mt = load_model_and_processor()

    st.subheader("Enter your information")

    # User input fields
    time_alone = st.slider("Time spent alone", 0.0, 11.0, 5.0)
    stage_fear = st.radio(
        "Do you feel stage fear?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    social_events = st.slider("Social event attendance", 0.0, 10.0, 5.0)
    going_outside = st.slider("Going outside", 0.0, 7.0, 3.0)
    drained = st.radio(
        "Do you feel drained after socializing?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    friends = st.slider("Friends circle size", 0.0, 15.0, 5.0)
    posts = st.slider("Post frequency", 0.0, 10.0, 5.0)

    # Run prediction when button is clicked
    if st.button("Predict"):
        # Collect user input into a list
        data = [
            time_alone,
            stage_fear,
            social_events,
            going_outside,
            drained,
            friends,
            posts,
        ]

        # Apply preprocessing to match training data format
        processed = dp.transform_input(data)

        # Generate prediction using trained model
        prediction = mt.predict(processed)

        # Display prediction result
        st.subheader("Result")

        if prediction[0] == 1:
            st.success("🧑 You seem like an extrovert")
        else:
            st.info("🧘 You seem like an introvert")


if __name__ == "__main__":
    main()
