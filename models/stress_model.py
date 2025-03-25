# models/stress_model.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load dataset (used for training, can be skipped if model is pre-trained)
data_path = "E:/Gayatri_project/datasets/mental_health_dataset.csv"
if not pd.io.common.file_exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")
data = pd.read_csv(data_path)  # Age, Gender, Occupation, etc.

# Select features and target
features = ['Age', 'Gender', 'Occupation', 'Marital_Status', 'Sleep_Duration', 
            'Sleep_Quality', 'Wake_Up_Time', 'Bed_Time', 'Physical_Activity', 'Screen_Time']
target_column = "Stress_Detection"
X = data[features].values
y = data[target_column].values  # Low, Medium, High

# Preprocess data
le = LabelEncoder()
for i in [1, 2, 3, 5]:  # Encode categorical: Gender, Occupation, Marital_Status, Sleep_Quality
    X[:, i] = le.fit_transform(X[:, i])
X = X.astype(float)
X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM
y = le.fit_transform(y)  # Encode target: Low=0, Medium=1, High=2

# Split data (for training, run once)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train model (run once, then save)
model = Sequential()
model.add(LSTM(50, input_shape=(1, 10), activation="relu"))  # 10 features, 1 timestep
model.add(Dense(3, activation="softmax"))  # 3 output classes
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model (commented out after first run)
# model.save("stress_model.h5")

# Load pre-trained model
model = load_model("stress_model.h5")

# Prediction function
def predict_stress(age, gender, occupation, marital_status, sleep_duration, 
                   sleep_quality, wake_up_time, bed_time, physical_activity, screen_time):
    """Predict stress level from input features."""
    input_data = np.array([
        age, 
        le.transform([gender])[0], 
        le.transform([occupation])[0], 
        le.transform([marital_status])[0], 
        sleep_duration, 
        le.transform([sleep_quality])[0], 
        wake_up_time, 
        bed_time, 
        physical_activity, 
        screen_time
    ]).astype(float)
    input_data = input_data.reshape((1, 1, 10))  # Reshape to match model input
    pred = model.predict(input_data, verbose=0)  # Silent prediction
    return ["Low", "Medium", "High"][np.argmax(pred)]

# Flask endpoint for stress prediction
@app.route("/stress", methods=["POST"])
def get_stress():
    try:
        # Get data from POST request
        data = request.get_json()
        age = float(data["age"])
        gender = data["gender"]
        occupation = data["occupation"]
        marital_status = data["maritalStatus"]
        sleep_duration = float(data["sleepDuration"])
        sleep_quality = data["sleepQuality"]
        wake_up_time = float(data["wakeupTime"])
        bed_time = float(data["bedTime"])
        physical_activity = float(data["physicalActivity"])
        screen_time = float(data["screenTime"])

        # Get prediction from model
        result = predict_stress(
            age, gender, occupation, marital_status, sleep_duration,
            sleep_quality, wake_up_time, bed_time, physical_activity, screen_time
        )

        # Return result as JSON
        return jsonify({
            "stressImpact": result,
            "suggestion": f"Predicted stress level: {result}"  # Placeholder suggestion
        })
    except Exception as e:
        return jsonify({
            "stressImpact": "Unknown",
            "suggestion": f"Error processing request: {str(e)}"
        }), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)