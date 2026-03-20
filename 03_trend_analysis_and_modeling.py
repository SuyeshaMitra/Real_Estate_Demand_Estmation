import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os

print("Loading Greater London dataset...")
data_path = 'london_data.csv'
df = pd.read_csv(data_path)

# Drop missing critical Data
df = df.dropna(subset=['price', 'date_of_transfer', 'district'])

print("Parsing dates and engineering features...")
df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
df['year'] = df['date_of_transfer'].dt.year
df['month'] = df['date_of_transfer'].dt.month

# We will focus on 2008-2022 to give exactly a 15-year window.
# 10 years train (2008-2017) -> Predict 5 years ahead validation (2018-2022)
df = df[(df['year'] >= 2008) & (df['year'] <= 2022)].copy()

print(f"Dataset size after 2008-2022 filter: {len(df)}")

# Trend analysis: Price vs Year
yearly_trend = df.groupby('year')['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_trend, x='year', y='price', marker="o")
plt.title("Historical Price Trend in Greater London (2008 - 2022)")
plt.xlabel("Year")
plt.ylabel("Average Property Price (£)")
plt.grid(True)
plt.savefig("C:/Users/SuyeshaM/.gemini/antigravity/brain/43e1b8ec-08b5-4205-9e3c-edfcd7d0c5b0/historical_trend.png")
plt.close()

# Feature Engineering
# Replace text categoricals with numbers so models can run
df['property_code'] = df['property_type'].astype('category').cat.codes
df['old_new_code'] = df['old_new'].astype('category').cat.codes
df['duration_code'] = df['duration'].astype('category').cat.codes
df['district_code'] = df['district'].astype('category').cat.codes

features = ['year', 'month', 'property_code', 'old_new_code', 'duration_code', 'district_code']
target = 'price'

# To prevent sklearn memory limits and massive computation times on 1.5 million rows,
# we run on a random subsample for training (100k rows) while preserving validation integrity.
print("Splitting Data into Train (2008-2017) and Holdout Test (2018-2022) ...")
train_df = df[df['year'] <= 2017].sample(n=100000, random_state=42)
test_df = df[df['year'] >= 2018].sample(n=50000, random_state=42)

X_train = train_df[features]
y_train = np.log1p(train_df[target])  # Log transform the target

X_test = test_df[features]
y_test = test_df[target]  # We compare against actual real prices

print("Training Random Forest Regressor...")
start_time = time.time()
rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time
print(f"Random Forest Training time: {rf_time:.2f} seconds.")

print("Training Neural Network (MLP) Regressor...")
start_time = time.time()
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, activation='relu', random_state=42)
mlp_model.fit(X_train, y_train)
mlp_time = time.time() - start_time
print(f"Neural Network Training time: {mlp_time:.2f} seconds.")

def evaluate_model(name, model):
    # Predict in log scale, then convert back
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Model Evaluation: {name} ---")
    print(f"RMSE: £{rmse:,.2f}")
    print(f"MAE: £{mae:,.2f}")
    print(f"R-Squared (out-of-sample): {r2:.4f}")
    
    return y_pred

rf_pred = evaluate_model("Random Forest", rf_model)
mlp_pred = evaluate_model("Neural Network", mlp_model)

# Save evaluation plot
test_df['rf_predicted_price'] = rf_pred
yearly_test_trend = test_df.groupby('year').agg({'price': 'mean', 'rf_predicted_price': 'mean'}).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(yearly_test_trend['year'], yearly_test_trend['price'], marker="o", label="Actual Avg Price")
plt.plot(yearly_test_trend['year'], yearly_test_trend['rf_predicted_price'], marker="x", linestyle="--", label="Forecasted Price (RF)")
plt.title("5-Year Ahead Holdout Forecast Validation (2018-2022)")
plt.xlabel("Year")
plt.ylabel("Average Property Price (£)")
plt.legend()
plt.grid(True)
plt.savefig("C:/Users/SuyeshaM/.gemini/antigravity/brain/43e1b8ec-08b5-4205-9e3c-edfcd7d0c5b0/forecast_validation.png")
plt.close()

print("\nModeling and Analysis complete. Output charts are saved to artifacts.")
