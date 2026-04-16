class App:
    """
    Handles user interaction through a terminal-based interface.
    Collects input data and uses the trained model for predictions.
    """

    def __init__(self, model, processor):
        """
        Initializes the application.

        Args:
            model: Trained machine learning model
            processor: DataProcessor instance for preprocessing input
        """
        self.model = model
        self.processor = processor

    def get_user_input(self):
        """
        Prompts the user to input behavioral data.

        The user provides values on a relative scale rather than exact units,
        to match the format of the training dataset.

        Returns:
            list: User input as numerical values
        """
        print(
            "\nAnswer based on relative levels and not exact units. Higher number means higher frequency."
        )

        # Collect user input for each feature
        time_alone = float(input("Time spent alone (0-11): "))
        stage_fear = int(input("Do you feel stage fear? (1 = Yes, 0 = No) "))
        social_events = float(input("Social event attendance (0-10): "))
        going_outside = float(input("Going outside (0-7): "))
        drained = int(
            input("Do you feel drained after socializing? (1 = Yes, 0 = No) ")
        )
        friends = float(input("What size is your friends circle? (0-15) "))
        posts = float(input("Post frequency (0-10): "))

        return [
            time_alone,
            stage_fear,
            social_events,
            going_outside,
            drained,
            friends,
            posts,
        ]

    def run(self):
        """
        Runs the application loop.

        Continuously collects user input, performs prediction,
        and displays the result until the user chooses to exit.
        """
        while True:
            # Get input from user
            data = self.get_user_input()

            # Apply preprocessing (scaling)
            processed = self.processor.transform_input(data)

            # Make prediction
            prediction = self.model.predict(processed)

            # Display result
            print("\n--- Result ---")
            if prediction[0] == 1:
                print("🧑 You seem like an extrovert")
            else:
                print("🧘 You seem like an introvert")

            # Ask user if they want to try again
            again = input("\nTry again? (y/n): ")
            if again.lower() != "y":
                break
