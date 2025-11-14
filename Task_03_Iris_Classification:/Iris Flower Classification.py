# ==========================================================
# IRIS FLOWER CLASSIFICATION (TASK 3)
# Predict flower type using Logistic Regression
# ==========================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# File location
file_path = 'IRIS.csv'

# Step 1: Load the data
try:
    df = pd.read_csv(file_path)
    print("--- Data loaded successfully ---")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Show few rows and info about data
print(df.head())
print(df.info())


# ==========================================================
# Step 2: Data Visualization
# ==========================================================

# Show how features relate to each other by species
print("\nMaking Pair Plot...")
sns.pairplot(df, hue='species')
plt.suptitle('Pair Plot of Features', y=1.02)
plt.savefig('iris_pair_plot.png')
plt.close()
print("[Image: iris_pair_plot.png] saved.")


# Step 3: Split the data

# X = input features, y = target
X = df.drop('species', axis=1)
y = df['species']

# Split into 80% train and 20% test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData split done: Train {X_train.shape}, Test {X_test.shape}")


# Step 4: Train the model

# Create Logistic Regression model
model = LogisticRegression(max_iter=200)

# Train it
model.fit(X_train, y_train)
print("\nModel training complete.")


# Step 5: Test the model

# Predict test results
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Results ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Show more detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================================
# Step 6: Evaluation and Output
# ==========================================================

# Make confusion matrix
cm = confusion_matrix(y_test, y_pred)
species_labels = np.unique(y)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=species_labels,
            yticklabels=species_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('iris_confusion_matrix.png')
plt.close()
print("\n[Image: iris_confusion_matrix.png] saved.")


# Show sample actual vs predicted results
print("\nSample Predictions:")
prediction_results = pd.DataFrame({
    'Actual Species': y_test,
    'Predicted Species': y_pred
})
print(prediction_results.sample(10, random_state=42).sort_index())
