# models/bmi_model.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("../datasets/health_dataset.csv")  # Age, Gender, Height, Weight, Index
X = data[["Age", "Gender", "Height", "Weight"]].values
y = data["Index"].values  # 0-5 (Extremely Weak to Extreme Obesity)

# Preprocess
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])  # Encode Gender
X = X.astype(float)

# Reshape for LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Model
model = Sequential()
model.add(LSTM(50, input_shape=(1, 4), activation="relu"))
model.add(Dense(6, activation="softmax"))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=10, batch_size=32)

# Save model
model.save("bmi_model.h5")

# Prediction function
def predict_bmi(age, gender, height, weight):
    model = load_model("bmi_model.h5")
    gender_encoded = le.transform([gender])[0]
    input_data = np.array([[age, gender_encoded, height, weight]]).astype(float)
    input_data = input_data.reshape((1, 1, 4))
    pred = model.predict(input_data)
    return np.argmax(pred)

# Input Values
age = int(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ")
height = float(input("Enter Height (in cm): "))
weight = float(input("Enter Weight (in kg): "))

# Prediction
result = predict_bmi(age, gender, height, weight)
print(f"Predicted BMI Category: {result}")
