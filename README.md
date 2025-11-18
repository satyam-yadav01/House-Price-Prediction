#  Real Estate Price Prediction (Machine Learning + Streamlit)

This project predicts house prices using a **Random Forest Machine Learning model** trained on the `Housing.csv` dataset. It also includes a **Streamlit web application** that allows users to input property details and receive instant price predictions.

---

## 🚀 Features

* Full ML pipeline using **Random Forest Regressor**
* Preprocessing with scaling + one-hot encoding
* Model evaluation (MAE, MSE, RMSE, R²)
* Interactive Streamlit web app for predictions
* Clean and modular project structure

---

## 📂 Project Structure

```
House Price Prediction/
│-- index.py              # Train ML model and save model.pkl
│-- app.py                # Streamlit app for predictions
│-- model.pkl             # Saved trained model
│-- Housing.csv           # Dataset
│-- requirements.txt      # Dependencies
│-- README.md             # Documentation
```

---

## ⚙️ Installation

### 1. Clone the repo

```
git clone <your-github-repo-url>
cd House-Price-Prediction
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## 📘 Usage

### ▶️ Train the Model

```
python train_model.py
```

This generates `model.pkl`.

### 🌐 Run the Streamlit App

```
streamlit run app.py
```

Now open the link shown in the terminal (usually `localhost:8501`).

---

## 📊 Model Details

* **Algorithm:** Random Forest Regressor
* **Metrics:**

  * MAE (Mean Absolute Error)
  * MSE (Mean Squared Error)
  * RMSE (Root Mean Squared Error)
  * R² Score

The Random Forest model provides strong performance for tabular datasets with mixed numeric and categorical features.

---

## 🧪 Dataset

The dataset includes columns:

```
price, area, bedrooms, bathrooms, stories,
mainroad, guestroom, basement, hotwaterheating,
airconditioning, parking, prefarea, furnishingstatus
```

---

## 📦 Requirements

See `requirements.txt` for all dependencies.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

This project is open-source and free to use.

---

## ⭐ Acknowledgements

* Streamlit for the web app framework
* Scikit-learn for the ML pipeline
* Kaggle/UCI dataset (Housing.csv)

---


