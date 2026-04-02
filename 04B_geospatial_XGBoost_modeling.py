# Import pandas for data ingestion and manipulation
import pandas as pd
# Import numpy for math operations
import numpy as np
# Import pgeocode to translate address postcodes to coordinates
import pgeocode
# Import time to measure algorithm performance speeds
import time
# Import XGBoost gradient boosting decision tree regressor
from xgboost import XGBRegressor
# Import metrics libraries to check predictive drift 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Loading Greater London dataset...")
# Load basic filtered dataset into memory
df = pd.read_csv('london_data.csv')

# Drop missing critical data fields so model doesn't crash on nulls
df = df.dropna(subset=['price', 'date_of_transfer', 'postcode'])

print("Parsing dates...")
# Convert transfer strings to system datetime schema
df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
# Pull out discrete year numbers
df['year'] = df['date_of_transfer'].dt.year
# Pull out discrete month numbers
df['month'] = df['date_of_transfer'].dt.month

# Filter dataframe maintaining just the 15-year 2008-2022 dataset scope
df = df[(df['year'] >= 2008) & (df['year'] <= 2022)].copy()

print("Extracting unique postcodes for Geospatial mapping...")
# Retrieve unique postcodes
unique_postcodes = df['postcode'].unique()

print(f"Fetching Latitude and Longitude for {len(unique_postcodes)} unique postcodes using pgeocode...")
# Initialize UK localization for the geocoding dictionary
nom = pgeocode.Nominatim('gb')

# Split off inner-code strings focusing just on outer-codes for faster mass regional fetching
outcodes = pd.Series(unique_postcodes).str.split(' ').str[0]
# Fetch coordinate series based on the outcodes
geo_data = nom.query_postal_code(outcodes.tolist())

# Map results sequentially to an output dictionary binding real postcodes to fetched points
postcode_map = pd.DataFrame({
    'postcode': unique_postcodes,
    'latitude': geo_data['latitude'].values,
    'longitude': geo_data['longitude'].values
})

print("Merging Geospatial data back to main dataset...")
# Stitch physical map points back to large training dataframe joining on zip logic
df = df.merge(postcode_map, on='postcode', how='left')
# Purge unmapped rows
df = df.dropna(subset=['latitude', 'longitude'])

# Feature Engineering categoricals into numeric vectors
df['property_code'] = df['property_type'].astype('category').cat.codes
df['old_new_code'] = df['old_new'].astype('category').cat.codes

# Package final predictive features specifically incorporating geospatial coordinate floats
features = ['year', 'month', 'property_code', 'old_new_code', 'latitude', 'longitude']
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
validation_df.to_csv("prediction_validation_xgb.csv", index=False)
# Conclude module success status
print("\nValidation Dataset saved as 'prediction_validation_xgb.csv' for review!")
