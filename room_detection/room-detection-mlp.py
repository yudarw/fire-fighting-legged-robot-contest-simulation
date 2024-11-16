import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Prepare the data for training the MLP model
# ==============================================================================

# Load the room data
data = pd.read_csv('room_data.csv', header=None)
print('Data Shape:', data.shape)
print(data.head())
print('\n')

# Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print('Shape of X:', X.shape)
print('Shape of y:', y.shape)
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

print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
print('X_train:', X_train[:5])
print('y_train:', y_train[:5])


# Train the data by using MLP classifier
# ==============================================================================

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)

# Print the architecture of the model
# print("Number of input features:", mlp.coefs_[0].shape[0])
# print("Number of layers (including input and output):", mlp.n_layers_)
# print("Number of outputs:", mlp.n_outputs_)
# print("Number of neurons in each layer:", [coef.shape[1] for coef in mlp.coefs_])
# print("Activation function:", mlp.activation)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot the training loss curve
# plt.figure(figsize=(10, 5))
# plt.plot(mlp.loss_curve_, label='Training Loss')
# plt.title('Training Loss Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# Save the model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(mlp, 'room_detection_mlp_model.pkl')