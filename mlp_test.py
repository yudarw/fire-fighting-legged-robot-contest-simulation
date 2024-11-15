import pandas as pd
import numpy as np
import joblib

# Load the saved model
mlp = joblib.load('room_detection_mlp_model.pkl')

# Load the saved scaler
scaler = joblib.load('scaler.pkl')
#label_encoder = LabelEncoder()
#label_encoder.classes_ = np.load('label_classes.npy')

# Define the input sensors data
dist = np.array([[60,35,62,50,38,16,11,14]])

# Convert the sensors array to a DataFrame
X_test2 = scaler.transform(dist)
y_test2 = mlp.predict(X_test2)

# Print the predicted room label
print("Predicted Room Label for sensors:", y_test2)