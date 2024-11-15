import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load the data
room1A = pd.read_csv('room1A.csv', header=None)
room1B = pd.read_csv('room1B.csv', header=None)
room2 = pd.read_csv('room2.csv', header=None)
room4 = pd.read_csv('room4.csv', header=None)

# Combine all data into a single DataFrame
data = pd.concat([room1A, room1B, room2, room4], ignore_index=True)

# Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot the training loss curve
plt.figure(figsize=(10, 5))
plt.plot(mlp.loss_curve_, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

