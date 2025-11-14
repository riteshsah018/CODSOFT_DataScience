## 💰 Task 4: Sales Prediction 📈

### 🎯 Goal  
The objective of this task was to build a model that predicts **future sales** based on advertising expenditures across multiple channels—**TV, Radio, and Newspaper**.  
Since the target variable (Sales) is a continuous value, this is a **Regression problem**.

---

## 🛠️ Method Used (How the Task Was Completed)

### **1. Exploratory Data Analysis (EDA)**  
Before training the model, a detailed analysis was performed to understand the relationship between advertising budget and sales.

- **Correlation Analysis:**  
  A correlation matrix revealed that **TV** had the strongest positive correlation with Sales, followed by **Radio**, while Newspaper had a weaker relationship.

- **Visual Insights:**  
  Scatter plots were used to visually inspect linear trends.  
  The patterns confirmed strong linear relationships, supporting the choice of a regression model.

---

## 🤖 **2. Model Training**

- **Model Used:** *Linear Regression*  
- Linear Regression was selected due to its simplicity, interpretability, and ability to show how each advertising channel contributes to sales.

---

## 📊 **Evaluation & Visual Results**

### **1. Model Performance**

- **R-squared (R²):** `0.9059`  
  Indicates that over **90%** of the variation in sales is explained by the advertising budget—an excellent model fit.

- **Mean Squared Error (MSE):** `2.9078`  
  A low MSE value, showing that predictions are very close to the actual sales figures.

---

### **2. Actual vs Predicted Sales Plot**  
A scatter plot comparing real and predicted values.  
The points align closely around the diagonal red line (perfect prediction line), visually validating the high R² score.

---

### **3. Residuals Distribution**  
Residuals were centered around zero and followed a near-normal distribution—  
a sign of **unbiased predictions** and a **robust regression model**.

---

## 📦 Deliverables

- **Task4.py** → Full machine learning workflow (EDA, model training, evaluation).  
- **advertising.csv** → Dataset used for analysis and model building.  
- **4 Visualization Files** →  
  - Correlation Matrix  
  - Scatter Plots  
  - Actual vs Predicted Plot  
  - Residuals Distribution Plot  

---
