# ================================================================
# MOVIE RATING PREDICTION (TASK 2)
# Simple project: clean data, make model, and check results
# ================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import re
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the dataset ---
file_path = 'IMDb Movies India.csv'

try:
    # Using latin-1 encoding to read special characters
    df = pd.read_csv(file_path, encoding='latin-1')
    print("--- Data Loaded Successfully ---")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()


# --- Step 1: Data Cleaning ---

# Remove rows where Rating is missing
df.dropna(subset=['Rating'], inplace=True)
df['Rating'] = df['Rating'].astype(float)
print(f"Data size after removing missing ratings: {len(df)}")

# Reset index after dropping rows
df.reset_index(drop=True, inplace=True)

# Clean Year column and convert to numbers
df['Year'] = df['Year'].astype(str).str.extract('(\d{4})').astype(float)
df.dropna(subset=['Year'], inplace=True)
df['Year'] = df['Year'].astype(int)

# Clean Duration column and convert to numbers
df['Duration'] = df['Duration'].astype(str).str.replace(' min', '').str.strip()
mean_duration = df['Duration'].replace('NaN', np.nan).astype(float).mean()
df['Duration'].fillna(mean_duration, inplace=True)
df['Duration'] = df['Duration'].astype(float)

# Fill missing values in text columns
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col].fillna('Unknown', inplace=True)
print("Missing text values filled.")

# Drop columns not needed
df.drop(['Name', 'New_ID', 'Reviews', 'Votes'], axis=1, inplace=True, errors='ignore')


# --- Step 2: Feature Creation ---

# Convert Genre into multiple binary columns
df['Genre'] = df['Genre'].apply(lambda x: [g.strip() for g in x.split(',')])
mlb = MultiLabelBinarizer()
genre_features = mlb.fit_transform(df['Genre'])
genre_df = pd.DataFrame(genre_features, columns=[f'Genre_{c}' for c in mlb.classes_])

# Combine Director and Actors into one text column
df['cast_features'] = (
    df['Director'] + ' ' + df['Actor 1'] + ' ' + df['Actor 2'] + ' ' + df['Actor 3']
)

# Turn text data into numbers using TF-IDF
tfidf = TfidfVectorizer(max_features=100)
cast_matrix = tfidf.fit_transform(df['cast_features'])
print("Genre and cast features changed into numeric form.")


# ================================================================
# Step 3: Data Visualization
# ================================================================

# Show rating distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Rating'], kde=True, bins=20)
plt.title('Movie Ratings Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--')
plt.savefig('rating_distribution.png')
plt.close()
print("\n[Image: rating_distribution.png] saved.")


# --- Step 4: Combine All Features ---

# Pick numerical columns
numerical_features = df[['Year', 'Duration']]

# Scale (normalize) numerical data
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(numerical_features)
scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_features.columns)

# Combine all features together
X_combined = hstack([
    scaled_numerical_df.values,
    genre_df.values,
    cast_matrix
])

# Replace missing or infinity values
X_final = np.nan_to_num(X_combined.toarray())

# Target column
y = df['Rating']

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)
print(f"Data split done. Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# --- Step 5: Train and Test Model ---

# Create Ridge Regression model
regression_model = Ridge(alpha=1.0)
regression_model.fit(X_train, y_train)
print("\nModel trained (Ridge Regression).")

# Predict test data
y_pred = regression_model.predict(X_test)

# Check model accuracy
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Results ---")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
print("\nLower MSE and R2 closer to 1 means better model.")


# ================================================================
# Step 6: Show Results
# ================================================================

# Plot actual vs predicted ratings
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Ratings')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.grid(True, linestyle='--')
plt.savefig('actual_vs_predicted.png')
plt.close()
print("\n[Image: actual_vs_predicted.png] saved.")

# Plot residuals (difference between actual and predicted)
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=30)
plt.title('Residuals Distribution')
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--')
plt.savefig('residuals_distribution.png')
plt.close()
print("[Image: residuals_distribution.png] saved.")

# Show few prediction examples
print("\nSample Predictions (Actual vs Predicted):")
prediction_results = pd.DataFrame({
    'Actual Rating': y_test,
    'Predicted Rating': y_pred
})
print(prediction_results.sample(10, random_state=42).sort_index())
