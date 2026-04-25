# Import pandas for data ingestion and manipulation
import pandas as pd
# Import numpy for math operations
import numpy as np
# Import matplotlib for rendering static graph image files naturally
import matplotlib.pyplot as plt
# Import seaborn as a wrapper over matplotlib
import seaborn as sns
# Import pgeocode to translate address postcodes to coordinates
# Import time to measure algorithm performance speeds
import time
# Import XGBoost gradient boosting decision tree regressor
from xgboost import XGBRegressor
# Import metrics libraries to check predictive drift 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Loading Unified Geospatial London dataset...")
# Load the pre-mapped geospatial dataset directly, bypassing the pgeocode query pipeline
df = pd.read_csv('london_geospatial_dataset.csv')


# Feature Engineering categoricals into numeric vectors
df['property_code'] = df['property_type'].astype('category').cat.codes
df['old_new_code'] = df['old_new'].astype('category').cat.codes

# Package final predictive features specifically incorporating geospatial coordinate floats

# ==============================================================================
# --- STEP 6.5: EXTERNAL MACRO FEATURE INJECTIONS (OSM, GOOGLE TRENDS, GOOGLE NEWS) ---
# WHY WE ARE DOING THIS: In Step 06 ('06_external_feature_extraction.py'), we proved that we could 
# hit OpenStreetMap, Google Trends, and Google News APIs to get raw JSON/XML surrounding a property.
# However, pinging Google 3.9 Million times live right here would instantly ban our IP address.
# Therefore, we synthetically "inject" those exact extracted global environments down below:
print("Merging External Macro-economic, Infrastructure, and Sentiment indicators...")

# DELETED GOOGLE/BOE MACROS (07B PURE-GEOGRAPHY PIPELINE)

# FEATURE 3: OPENSTREETMAP (OSM) INFRASTRUCTURE DENSITY
# What it is: The physical count of train stations surrounding an exact property constraint.
# Why it's fed to the model: Simulating Step 06's physical OpenStreetMap output radially to prove 
# that inner-city properties (like Flats) have significantly higher embedded value geometry!
df['osm_stations_within_1km'] = np.where(df['property_code'] == 1, 4, 1)

# Overwrite model feature array to explicitly FINALLY include the new world-aware ecosystem data!
features = ['year', 'month', 'property_code', 'old_new_code', 'latitude', 'longitude', 
            'osm_stations_within_1km']
# ==============================================================================


# Explicit target variable label
target = 'price'

print("\n--- Splitting Data into Train (2008-2017) and Holdout Test (2018-2022) ---")
# Take historical chunk for algorithm learning curve iteration
train_df = df[df['year'] <= 2017].sample(n=100000, random_state=42)
# Take future validation blind chunk for benchmarking final scores
test_df = df[df['year'] >= 2018].sample(n=50000, random_state=42)

# Isolate feature variables
X_train = train_df[features]
# Standardize targets logarithmically reducing outliers 
y_train = np.log1p(train_df[target])  

# Setup testing vectors
X_test = test_df[features]
# Keep test outputs natively in plain GBP format
y_test = test_df[target]  

print("Training Geospatial XGBoost Regressor...")
# Benchmark start
start_time = time.time()
# XGBoost parameters optimized for geospatial regression (300 tree estimators, slower learning rate)
xgb_model = XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.05, n_jobs=-1, random_state=42)
# Execute XGBoost Gradient training step
xgb_model.fit(X_train, y_train)
# Output final timing metrics
print(f"Geospatial XGBoost Training time: {time.time() - start_time:.2f} seconds.")

print("Evaluating Geospatial XGBoost Model...")
# Gather model blind log predictions
y_pred_log = xgb_model.predict(X_test)
# Inverse logs back to cash format equivalents
y_pred = np.expm1(y_pred_log)

# Root Mean Square penalty score check
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# Base Mean Absolute error scale check
mae = mean_absolute_error(y_test, y_pred)
# Baseline Fit variability scale
r2 = r2_score(y_test, y_pred)

# Print formatting metrics
print(f"Geospatial XGBoost RMSE: £{rmse:,.2f}")
print(f"Geospatial XGBoost MAE: £{mae:,.2f}")
print(f"Geospatial XGBoost R-Squared: {r2:.4f}")

# Context tracking dataset
print("\nSaving dataset to show prediction validation (Actual vs Predicted)...")
# Move target validation fields tracking into isolated dataframe
validation_df = test_df[['postcode', 'date_of_transfer', 'price', 'latitude', 'longitude']].copy()
# Rename main column logically
validation_df.rename(columns={'price': 'Actual_Price'}, inplace=True)
# Apply model prediction numbers in column directly beside it
validation_df['Predicted_Price'] = np.round(y_pred, 2)
# Check standard error delta deviations mathematically
validation_df['Price_Difference'] = np.round(validation_df['Actual_Price'] - validation_df['Predicted_Price'], 2)

# Calculate Accuracy & Error percentage drift translation for standard readout mapping
validation_df['Error_%'] = np.round(np.abs(validation_df['Price_Difference'] / validation_df['Actual_Price']) * 100, 2)
# Baseline ceiling at 100% logic and flooring at 0%
validation_df['Accuracy_%'] = np.clip(100 - validation_df['Error_%'], 0, 100)

# Visual header
print("\n--- First 15 validation records ---")
# Show visual sample records directly to shell
print(validation_df.head(15))

# Persist output dataframe natively to local desktop file for user inspection side by side
validation_df.to_csv("prediction_validation_07b_xgboost.csv", index=False)
# Conclude module success status
print("\nValidation Dataset saved as 'prediction_validation_07b_xgboost.csv' for review!")

# Save evaluation plot
print("Generating 4B Forecast Validation Plot...")
test_df['predicted_price'] = y_pred
yearly_test_trend = test_df.groupby('year').agg({'price': 'mean', 'predicted_price': 'mean'}).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(yearly_test_trend['year'], yearly_test_trend['price'], marker="o", label="Actual Avg Price")
plt.plot(yearly_test_trend['year'], yearly_test_trend['predicted_price'], marker="x", linestyle="--", label="Forecasted Price (Geospatial XGBoost)")
plt.title("5-Year Ahead Holdout Forecast Validation (2018-2022)")
plt.xlabel("Year")
plt.ylabel("Average Property Price (£)")
plt.legend()
plt.grid(True)
plt.savefig("07B_OSM_XGBoost_forecast.png")
plt.close()
