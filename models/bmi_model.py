# models/bmi_model.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load dataset and train model (this part could be run once separately to save the model)
data = pd.read_csv("../datasets/health_dataset.csv")  # Age, Gender, Height, Weight, Index
X = data[["Age", "Gender", "Height", "Weight"]].values
y = data["Index"].values  # 0-5 (Extremely Weak to Extreme Obesity)

# Preprocess
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])  # Encode Gender
X = X.astype(float)

# Reshape for LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Model (train only if not already saved)
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
    pred = model.predict(input_data, verbose=0)  # Silent prediction
    return np.argmax(pred)

# Flask endpoint for BMI prediction
@app.route("/bmi", methods=["POST"])
def get_bmi():
    try:
        # Get data from POST request
        data = request.get_json()
        age = int(data["age"])
        gender = data["gender"]
        height = float(data["height"])
        weight = float(data["weight"])

        # Get prediction from model
        result = predict_bmi(age, gender, height, weight)

        # Return result as JSON
        return jsonify({
            "bmi_category": int(result),  # Convert to int for JSON compatibility
            "recommendation": f"Category {result} predicted"  # Placeholder recommendation
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "bmi_category": None,
            "recommendation": "Error processing request"
        }), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)