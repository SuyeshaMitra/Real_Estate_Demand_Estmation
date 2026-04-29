import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("======================================================")
print(" 07C: EXECUTING LIGHTGBM ON ENRICHED DATASET")
print("======================================================")

# 1. Load Data
print("1. Loading 'london_geospatial_enriched_dataset.csv'...")
df = pd.read_csv('london_geospatial_enriched_dataset.csv')

# 2. Preprocess Categorical Features
print("2. Preprocessing categorical features...")
le = LabelEncoder()
df['property_type_encoded'] = le.fit_transform(df['property_type'].astype(str))
df['old_new_encoded'] = le.fit_transform(df['old_new'].astype(str))
df['duration_encoded'] = le.fit_transform(df['duration'].astype(str))

# Define Features
features = [
    'year', 'month', 'latitude', 'longitude', 
    'property_type_encoded', 'old_new_encoded', 'duration_encoded',
    'distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km',
    'distance_to_nearest_school_km', 'distance_to_nearest_station_km',
    'sbert_sentiment_index', 'google_trends_volume', 'boe_interest_rate'
]
target = 'price'

# 3. Train / Test Split (Time Series: Train 2008-2017, Test 2018-2022)
print("3. Executing Temporal Split: Train(2008-2017) -> Test(2018-2022)")
train_df = df[(df['year'] >= 2008) & (df['year'] <= 2017)]
test_df = df[(df['year'] >= 2018) & (df['year'] <= 2022)]

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# 4. Train Model
print("4. Training LightGBM Regressor (Leaves=31, Estimators=100)...")
model = LGBMRegressor(n_estimators=100, num_leaves=31, random_state=42, n_jobs=-1)

start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

# 5. Predict & Calculate Global Metrics
print("5. Predicting on 2018-2022 Holdout Set...")
start_time = time.time()
predictions = model.predict(X_test)
predict_time = time.time() - start_time

test_df['predicted_price'] = predictions
test_df['absolute_error'] = np.abs(test_df['price'] - test_df['predicted_price'])
# Accuracy Rule: 100 - (Error / Price) * 100, bounded at 0
test_df['accuracy_percentage'] = np.maximum(0, 100 - (test_df['absolute_error'] / test_df['price']) * 100)

global_mae = test_df['absolute_error'].mean()
global_acc = test_df['accuracy_percentage'].median()

print(f"\n[GLOBAL RESULTS] LightGBM")
print(f" -> Mean Absolute Error (MAE): £{global_mae:,.2f}")
print(f" -> Median Accuracy: {global_acc:.2f}%")
print(f" -> Training Time: {train_time:.2f} seconds")
print(f" -> Prediction Time: {predict_time:.2f} seconds")

# 6. Granular Metric Breakdowns
print("\n6. Exporting Granular Error Breakdowns...")

# Yearly
yearly_metrics = test_df.groupby('year').agg(
    MAE=('absolute_error', 'mean'),
    Median_Accuracy=('accuracy_percentage', 'median')
).reset_index()
print("\n--- Yearly Breakdown ---")
print(yearly_metrics.to_string(index=False))

# Monthly
monthly_metrics = test_df.groupby('month').agg(
    MAE=('absolute_error', 'mean'),
    Median_Accuracy=('accuracy_percentage', 'median')
).reset_index()
print("\n--- Monthly Breakdown ---")
print(monthly_metrics.to_string(index=False))

# Postcode
postcode_metrics = test_df.groupby('postcode').agg(
    Avg_Price=('price', 'mean'),
    MAE=('absolute_error', 'mean'),
    Median_Accuracy=('accuracy_percentage', 'median')
).reset_index().sort_values(by='MAE', ascending=False)
postcode_metrics.to_csv("07C_LightGBM_Postcode_Errors.csv", index=False)

# 7. Rendering Validation Charts
print("\n7. Rendering Validation Visualizations...")

# Chart 1: Historical vs Forecast
plt.figure(figsize=(14, 7))
historical_yearly = df.groupby('year')['price'].mean().reset_index()
forecast_yearly = test_df.groupby('year')['predicted_price'].mean().reset_index()

plt.plot(historical_yearly['year'], historical_yearly['price'], label='Actual Historical Price', color='blue', marker='o', linewidth=2)
plt.plot(forecast_yearly['year'], forecast_yearly['predicted_price'], label='LightGBM Forecast (2018-2022)', color='orange', marker='X', linestyle='--', linewidth=3)
plt.title('LightGBM: Actual vs Forecasted Real Estate Prices (London)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Price (£)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.savefig('07C_Historical_vs_Forecast_Prices.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 2: Monthly vs Yearly Error Heatmap
heatmap_data = test_df.pivot_table(index='month', columns='year', values='accuracy_percentage', aggfunc='median')
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Median Accuracy %'})
plt.title('LightGBM: Median Accuracy % by Month & Year', fontsize=16)
plt.ylabel('Month', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.savefig('07C_Yearly_Monthly_Error_Heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 3: Postal Code Error Distribution
plt.figure(figsize=(12, 6))
top_postcodes = postcode_metrics.dropna().head(50) # top 50 worst performing postcodes
plt.scatter(top_postcodes['Avg_Price'], top_postcodes['MAE'], alpha=0.6, color='orange')
plt.title('LightGBM: MAE Distribution across London Postcodes', fontsize=16)
plt.xlabel('Average Postcode Property Price (£)', fontsize=12)
plt.ylabel('Mean Absolute Error (£)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('07C_Postal_Code_Error_Distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(" -> [SAVED] 07C_Historical_vs_Forecast_Prices.png")
print(" -> [SAVED] 07C_Yearly_Monthly_Error_Heatmap.png")
print(" -> [SAVED] 07C_Postal_Code_Error_Distribution.png")

print("\n======================================================")
print(" LIGHTGBM VALIDATION COMPLETELY EXECUTED!")
print("======================================================")
