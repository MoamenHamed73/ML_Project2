# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 2️⃣ Load Dataset
# ==========================================
df = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\kc_house_data.csv")

# ==========================================
# 3️⃣ Data Cleaning
# ==========================================
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

def clean_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    for col in ["sqft_living", "bedrooms", "bathrooms", "price"]:
        df = remove_outliers(df, col)
    df["bedrooms"] = df["bedrooms"].astype(int)
    return df

df = clean_data(df)

# ==========================================
# 4️⃣ Feature Engineering
# ==========================================
df["house_age"] = 2025 - df["yr_built"]
df["renovated"] = (df["yr_renovated"] > 0).astype(int)
df["price_per_sqft"] = df["price"] / df["sqft_living"]
df["living_lot_ratio"] = df["sqft_living"] / df["sqft_lot"]

# ==========================================
# 5️⃣ Feature Selection
# ==========================================
features = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade",
    "sqft_above", "sqft_basement", "lat", "long",
    "house_age", "renovated", "living_lot_ratio"
]

X = df[features]
y = df["price"]

# ==========================================
# 6️⃣ Data Scaling
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 7️⃣ Split Data
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========================================
# 8️⃣ Train Models
# ==========================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append([name, r2, rmse])

results_df = pd.DataFrame(results, columns=["Model", "R2_Score", "RMSE"])
print("\n🔹 Model Performance:\n")
print(results_df)

# ==========================================
# 9️⃣ Cross Validation  
# ==========================================
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
    print(f"\n{name} Cross Validation R²: {scores.mean():.4f}")

 
# 🔹 Visualization  
# ==========================================
plt.figure(figsize=(8, 6))
sns.heatmap(df[features + ["price"]].corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
