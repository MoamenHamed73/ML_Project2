# ML_Project2
House Price Prediction
# 🏠 House Price Prediction – King County Dataset

Predicting house prices using machine learning regression models based on various house features.

---

## 📘 Overview
This project aims to **predict house prices** in King County, USA using different regression models.  
It includes **data cleaning**, **feature engineering**, **model training**, and **evaluation**.

The dataset contains details such as the number of bedrooms, bathrooms, square footage, location, and more.

---

## 📊 Dataset
- **Source:** Kaggle – [King County House Sales Dataset](https://www.kaggle.com/harlfoxem/housesalesprediction)  
- **Rows:** ~21,000  
- **Columns:** 21 features including price, bedrooms, bathrooms, sqft_living, etc.

---

## ⚙️ Project Steps

### 1️⃣ Import Libraries  
Using essential libraries for data manipulation, visualization, and model building.  

### 2️⃣ Data Cleaning  
- Dropped missing and duplicate values  
- Removed outliers using the IQR method  
- Converted columns to appropriate data types  

### 3️⃣ Feature Engineering  
Added new meaningful features:
- `house_age` → how old the house is  
- `renovated` → whether the house was renovated  
- `price_per_sqft` → price per square foot  
- `living_lot_ratio` → ratio between living space and lot area  

### 4️⃣ Model Training  
Models used:
- **Linear Regression**  
- **Ridge Regression**  
- **Lasso Regression**  
- **Random Forest Regressor**  
- **XGBoost Regressor** (optional)

### 5️⃣ Evaluation Metrics  
- **R² Score** – Measures accuracy  
- **RMSE (Root Mean Squared Error)** – Measures prediction error  

---

## 🧾 Results

| Model | R² Score | RMSE |
|-------|----------|-------|
| Linear Regression | 0.692 | 109,691 |
| Ridge Regression | 0.692 | 109,691 |
| Lasso Regression | 0.692 | 109,691 |
| Random Forest | 0.863 | 73,160 |

**Observation:**  
- Random Forest is the best performing model with the highest R² and lowest RMSE.  
- Linear/Ridge/Lasso models provide a baseline for comparison.

---

## 📈 Visualization
- Correlation Heatmap  
- Model comparison bar chart  

Example:
```python
sns.heatmap(df[features + ["price"]].corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

