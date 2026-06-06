import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load datasets
train_df = pd.read_csv("Training.csv")
test_df = pd.read_csv("Testing.csv")

# Target column (last column)
target_column = train_df.columns[-1]

# Features and Labels
X_train = train_df.drop(target_column, axis=1)
y_train = train_df[target_column]

X_test = test_df.drop(target_column, axis=1)
y_test = test_df[target_column]

# Encode diseases
encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Train Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predict on Testing Dataset
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n========================")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("========================")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save files
pickle.dump(model, open("disease_model.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))
pickle.dump(list(X_train.columns), open("symptoms.pkl", "wb"))

print("\nModel Saved Successfully!")
