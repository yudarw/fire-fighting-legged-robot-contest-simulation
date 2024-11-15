import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Load the data
room1A = pd.read_csv('room1A.csv', header=None)
room1B = pd.read_csv('room1B.csv', header=None)
room2 = pd.read_csv('room2.csv', header=None)
room3 = pd.read_csv('room3.csv', header=None)
room4 = pd.read_csv('room4.csv', header=None)

# Combine all data into a single DataFrame
data = pd.concat([room1A, room1B, room2, room3, room4], ignore_index=True)

# Print the first few rows of the data
print('Data Shape:', data.shape)
print(data.head())

# Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Print the shapes of x and y:
print('Shape of X:', X.shape)
print('Shape of y:', y.shape)
# print a few rows of X and y
print('X:', X[:5])
print('y:', y[:5])

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("============= Train Data =============")
# print the shapes of X_train and y_train
print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
# print a few rows of X_train and y_train
print('X_train:', X_train[:5])
print('y_train:', y_train[:5])

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')

# Create the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# Predict the room label for a given set of sensors data
dist = np.array([[61,35,63,50,36,14,10,12]])
X_test2 = scaler.transform(dist)
y_test2 = mlp.predict(X_test2)
print("Predicted Room Label for sensors:", label_encoder.inverse_transform(y_test2))

np.save('label_classes.npy', label_encoder.classes_)


joblib.dump(mlp, 'room_detection_mlp_model.pkl')
# # Plot the training loss curve
plt.figure(figsize=(10, 5))
plt.plot(mlp.loss_curve_, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

