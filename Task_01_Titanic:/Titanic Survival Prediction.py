# ================================================================
# TITANIC SURVIVAL PREDICTION (TASK 1)
# Data cleaning, model training, and visualization
# ================================================================

# --- Importing libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


# --- Load the data ---

# Make sure all these CSV files are in the same folder
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    gender_submission_df = pd.read_csv('gender_submission.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Some files are missing.")
    exit()

# Save target column and passenger IDs
y_train_target = train_df['Survived']
test_passenger_ids = test_df['PassengerId']

# Combine train and test data for easy cleaning
combined_df = pd.concat([train_df.drop('Survived', axis=1), test_df], ignore_index=True)

train_len = len(train_df)
print(f"Training data rows: {train_len}")
print(f"Combined data rows: {len(combined_df)}")


# --- Clean and prepare the data ---

# Fill missing values
combined_df['Age'].fillna(combined_df['Age'].mean(), inplace=True)
combined_df['Fare'].fillna(combined_df['Fare'].mean(), inplace=True)
combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' because it has too many missing values
combined_df.drop('Cabin', axis=1, inplace=True)
print("\nMissing values fixed and 'Cabin' removed.")

# Change 'Sex' to numbers
combined_df['Sex'] = combined_df['Sex'].map({'male': 1, 'female': 0})

# Convert 'Embarked' to dummy columns
embarked_dummies = pd.get_dummies(combined_df['Embarked'], prefix='Embarked', drop_first=True, dtype=int)
combined_df = pd.concat([combined_df, embarked_dummies], axis=1)

# Drop columns we don’t need
combined_df.drop(['Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

print("Data cleaned and ready.")
print(combined_df.head())


# ================================================================
# Exploratory Data Analysis (EDA)
# ================================================================

# Survival rate by passenger class
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')
plt.grid(axis='y')
plt.savefig('survival_by_pclass.png')
plt.close()
print("\n[Image: survival_by_pclass.png] saved.")

# Survival rate by gender
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=train_df, palette={'male': 'blue', 'female': 'red'})
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.grid(axis='y')
plt.savefig('survival_by_sex.png')
plt.close()
print("[Image: survival_by_sex.png] saved.")


# --- Split and train the model ---

# Split data back into train and test sets
X_train_full = combined_df[:train_len]
X_test = combined_df[train_len:]

# Create validation split
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_full, y_train_target, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("\nFeature scaling done.")

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train_split)
print("Model trained successfully.")


# --- Check model performance ---

y_val_pred = model.predict(X_val_scaled)
validation_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {validation_accuracy * 100:.2f}%")


# ================================================================
# Model Results and Visualization
# ================================================================

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix (Validation Data)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
plt.close()
print("[Image: confusion_matrix.png] saved.")

# Show a few prediction samples
print("\nSample Predictions:")
validation_results = pd.DataFrame({
    'Actual': y_val,
    'Predicted': y_val_pred
})
print(validation_results.head(10))


# --- Create final submission file ---

# Predict on test data
final_predictions = model.predict(X_test_scaled)

# Make submission file
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': final_predictions.astype(int)
})
submission_df.to_csv('titanic_submission.csv', index=False)

print("\n--- Final Output ---")
print("File 'titanic_submission.csv' created successfully.")
print(submission_df.head())
