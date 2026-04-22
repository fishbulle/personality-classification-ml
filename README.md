# Personality Classification ML

## 📌 Purpose

This project is a machine learning application that classifies whether a person is an **introvert or extrovert** based on behavioral data.

The model is trained on a dataset containing features such as time spent alone, social activity, and social media usage.

---

## ⚙️ Features

* Data preprocessing (cleaning, encoding, scaling)
* Exploratory Data Analysis (EDA) in Jupyter Notebook
* Model training using Logistic Regression
* Model evaluation (Accuracy, F1 Score, ROC AUC, Confusion Matrix)
* Terminal-based application for real-time predictions

---

## 📂 Project Structure

```
project/
│
├── data/
│   └── personality_dataset.csv
│
├── src/
│   ├── data_processing.py
│   ├── model_training.py
│   ├── app.py
│   └── main.py
│
├── notebooks/
│   └── analysis.ipynb
│
└── requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash id="q0b6fx"
git clone https://github.com/fishbulle/personality-classification-ml.git
cd personality-classification-ml
```

### 2. Create and activate a virtual environment

#### Windows:

```bash id="k8rv5u"
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux:

```bash id="pq5lyh"
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash id="2h1s4z"
pip install -r requirements.txt
```

### 4. Run the application

```bash id="pynm3y"
cd src
python main.py
```

---

## 🧠 How It Works

1. The dataset is loaded and preprocessed (cleaning, encoding, scaling)
2. The model is trained on the processed data
3. The trained model is saved and reused
4. The user provides input through the terminal
5. The model predicts whether the user is introvert or extrovert

---

## 📦 Dependencies

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* joblib

---

## ⚠️ Notes

* The dataset uses relative scales rather than exact real-world units.
* User input is therefore based on relative values (e.g., 0–10 scale).
* The trained model file (`model.pkl`) is generated during runtime and is not included in the repository.

---

## 📊 Dataset

The dataset used contains behavioral indicators such as:

* Time spent alone
* Social event attendance
* Going outside frequency
* Social media activity

---

## 👤 Author

Angelina Malmros
