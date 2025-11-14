# ==========================================================
# SALES PREDICTION (TASK 4)
# Predicting sales based on advertising spend
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
file_path = 'advertising.csv'

try:
    df = pd.read_csv(file_path)
    print("--- Data loaded successfully ---")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# The dataset has 'TV', 'Radio', 'Newspaper' as features and 'Sales' as target
feature_cols = ['TV', 'Radio', 'Newspaper']
target_col = 'Sales'

# Remove rows with missing values in important columns
df.dropna(subset=feature_cols + [target_col], inplace=True)


# ==========================================================
# Step 2: Data Visualization (EDA)
# ==========================================================

print("\nCreating EDA visuals...")

# Correlation heatmap to check relationships between variables
plt.figure(figsize=(8, 6))
sns.heatmap(df[feature_cols + [target_col]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between Advertising Channels and Sales')
plt.savefig('correlation_heatmap.png')
plt.close()
print("[Image: correlation_heatmap.png] saved.")

# Scatter plots to see how each ad type affects sales
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Advertising Spend vs. Sales', fontsize=16)

for i, feature in enumerate(feature_cols):
    sns.scatterplot(x=feature, y=target_col, data=df, ax=axes[i], color='teal')
    axes[i].set_title(f'{feature} Spend vs. Sales')
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('spend_vs_sales_scatter.png')
plt.close()
print("[Image: spend_vs_sales_scatter.png] saved.")


# ==========================================================
# Step 3: Split Data and Train Model
# ==========================================================

# Separate input (X) and output (y)
X = df[feature_cols]
y = df[target_col]

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nData split complete: Train {X_train.shape}, Test {X_test.shape}")

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully.")


# ==========================================================
# Step 4: Make Predictions and Evaluate
# ==========================================================

# Predict on test data
y_pred = model.predict(X_test)

# Check model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")
print("Higher R2 means better prediction results.")


# ==========================================================
# Step 5: Visualization of Results
# ==========================================================

# Plot distribution of residuals (errors)
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=20, color='darkorange')
plt.title('Residuals Distribution')
plt.xlabel('Residuals (Actual - Predicted)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('residuals_distribution.png')
plt.close()
print("[Image: residuals_distribution.png] saved.")

# Plot actual vs predicted sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='indigo')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=3)
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('actual_vs_predicted_sales.png')
plt.close()
print("[Image: actual_vs_predicted_sales.png] saved.")


# ==========================================================
# Step 6: Sample Predictions
# ==========================================================

# Show few actual vs predicted results
print("\nSample Predictions (Actual vs Predicted):")
prediction_results = pd.DataFrame({
    'Actual Sales': y_test,
    'Predicted Sales': y_pred.round(2)
})
print(prediction_results.sample(10, random_state=42).sort_index())
