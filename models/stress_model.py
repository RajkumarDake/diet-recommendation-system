# models/stress_model.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset with basic validation
data_path = "E:/Gayatri_project/datasets/mental_health_dataset.csv"
if not pd.io.common.file_exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")
data = pd.read_csv(data_path)  # Age, Gender, Occupation, etc.

# Print columns for debugging
print("Dataset columns:", data.columns.tolist())

# Select the 10 features the script expects
features = ['Age', 'Gender', 'Occupation', 'Marital_Status', 'Sleep_Duration', 
            'Sleep_Quality', 'Wake_Up_Time', 'Bed_Time', 'Physical_Activity', 'Screen_Time']
target_column = "Stress_Detection"  # Updated to match the actual column name

# Extract features and target
X = data[features].values  # Select only the specified features
y = data[target_column].values  # Target: Low, Medium, High

# Preprocess data
le = LabelEncoder()
for i in [1, 2, 3, 5]:  # Encode categorical: Gender, Occupation, Marital_Status, Sleep_Quality
    X[:, i] = le.fit_transform(X[:, i])
X = X.astype(float)
X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM: [samples, timesteps, features]
y = le.fit_transform(y)  # Encode target: Low=0, Medium=1, High=2

# Split data for training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, 10), activation="relu"))  # 10 features, 1 timestep
model.add(Dense(3, activation="softmax"))  # 3 output classes
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model with validation
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("stress_model.h5")

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
    pred = model.predict(input_data)
    return ["Low", "Medium", "High"][np.argmax(pred)]