# Personality Classification ML

## 📌 Purpose

This project is a machine learning application that classifies whether a person is an **introvert or extrovert** based on behavioral data.

The model is trained on a dataset containing features such as time spent alone, social activity, and social media usage.

The application includes an interactive web interface built with Streamlit for real-time predictions.

---

## ⚙️ Features

* Data preprocessing (cleaning, encoding, scaling)
* Exploratory Data Analysis (EDA) in Jupyter Notebook
* Model training using Logistic Regression
* Model evaluation (Accuracy, F1 Score, ROC AUC, Confusion Matrix)
* Interactive Streamlit web application for predictions
* Model persistence using Joblib

---

## 📂 Project Structure

```text
project/
│
├── data/
│   └── personality_dataset.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── app.py
│   ├── data_processing.py
│   ├── main.py
│   └── model_training.py
│
├── model.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/fishbulle/personality-classification-ml.git
cd personality-classification-ml
```

---

### 2. Create and activate a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Train the model

Run the training script:

```bash
python src/main.py
```

This will:

* Load and preprocess the dataset
* Train the Logistic Regression model
* Save the trained model as `model.pkl`

---

### 5. Start the Streamlit application

From the project root:

```bash
streamlit run src/app.py
```

The app will open automatically in your browser.

---

## 🧠 How It Works

1. The dataset is loaded and preprocessed
2. Features are cleaned, encoded, and scaled
3. A Logistic Regression model is trained
4. The trained model is saved using Joblib
5. Users enter behavioral data in the Streamlit interface
6. The model predicts whether the user is more introverted or extroverted

---

## 📊 Input Features

The prediction is based on behavioral indicators such as:

* Time spent alone
* Stage fear
* Social event attendance
* Going outside frequency
* Feeling drained after socializing
* Friend circle size
* Social media posting frequency

---

## 📦 Dependencies

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* streamlit
* joblib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Notes

* The dataset uses relative scales rather than exact real-world units
* User input is therefore based on relative values (e.g. 0–10 scale)
* The trained model file (`model.pkl`) is generated during runtime and is not included in the repository

---

## 👤 Author

Angelina Malmros