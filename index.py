import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('Housing.csv')
print("Data Loaded Successfully!")
print(df.head())


print("Data Info:")
print(df.info())

print("Missing Values:")
print(df.isnull().sum())


# Price distribution
plt.figure(figsize=(6,4))
sns.histplot(df['price'], kde=True)
plt.title("Price Distribution")
plt.show()


# Correlation heatmap 
numeric_df = df[['price','area','bedrooms','bathrooms','stories','parking']]
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# --------- PREPROCESSING ---------
numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating','airconditioning', 'prefarea', 'furnishingstatus']


X = df[numeric_features + categorical_features]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Preprocessing
preprocessor = ColumnTransformer([("num", StandardScaler(), numeric_features),("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)])


# ----------- MODEL PIPELINE ----------
pipeline = Pipeline([("preprocessor", preprocessor),("model", RandomForestRegressor(n_estimators=300, random_state=42))])

print("Training Random Forest model...")
pipeline.fit(X_train, y_train)


# ---------- EVALUATION ------------
preds = pipeline.predict(X_test)


mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)


print(" Model Evaluation ")
print(f"MAE : {mae:,.2f}")
print(f"MSE : {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R² : {r2:.4f}")


# ---------- SAMPLE PREDICTION -----------
sample = X_test.iloc[0]
sample_df = sample.to_frame().T

pred_price = pipeline.predict(sample_df)[0]

print("Sample Input:")
print(sample)
print(f"Predicted Price: {pred_price:,.2f}")