# Import pandas to handle all tabular data loading
import pandas as pd
# Import mathematical operations library
import numpy as np
# Import matplotlib for rendering static graph image files naturally
import matplotlib.pyplot as plt
# Import seaborn as a wrapper over matplotlib
import seaborn as sns
# Import library that converts UK postcodes into precise GPS coordinates
import pgeocode
# Import standard library timer for assessing compute latency 
import time
# Import LightGBM gradient boosting framework (optimized for fast scaling)
from lightgbm import LGBMRegressor
# Import evaluation functions to check algorithmic performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Loading Greater London dataset...")
# Intake the main filtered dataset file
df = pd.read_csv('london_data.csv')

# Drop missing critical data fields
df = df.dropna(subset=['price', 'date_of_transfer', 'postcode'])

print("Parsing dates...")
# Convert transfer strings natively to pandas datetime elements
df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
# Pull out integer year identifiers
df['year'] = df['date_of_transfer'].dt.year
# Pull out integer month identifiers
df['month'] = df['date_of_transfer'].dt.month

# Filter dataframe maintaining just the 15-year 2008-2022 block
df = df[(df['year'] >= 2008) & (df['year'] <= 2022)].copy()

# Trend analysis: Price vs Year
print("Generating 4C Historical Trend Plot...")
yearly_trend = df.groupby('year')['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_trend, x='year', y='price', marker="o")
plt.title("Historical Price Trend in Greater London (2008 - 2022)")
plt.xlabel("Year")
plt.ylabel("Average Property Price (£)")
plt.grid(True)
plt.savefig("04C_historical_trend.png")
plt.close()

print("Extracting unique postcodes for Geospatial mapping...")
# Extract list of isolated generic postcodes 
unique_postcodes = df['postcode'].unique()

print(f"Fetching Latitude and Longitude for {len(unique_postcodes)} unique postcodes using pgeocode...")
# Point geographical fetcher at the Great Britain regional mapping registry
nom = pgeocode.Nominatim('gb')

# Slice string variable to take only frontward geographical blocks from zipcodes
outcodes = pd.Series(unique_postcodes).str.split(' ').str[0]
# Fetch coordinate series block mapped back to those exact outcodes
geo_data = nom.query_postal_code(outcodes.tolist())

# Map results sequentially to an output dictionary binding original strings to coordinate data
postcode_map = pd.DataFrame({
    'postcode': unique_postcodes,
    'latitude': geo_data['latitude'].values,
    'longitude': geo_data['longitude'].values
})

print("Merging Geospatial data back to main dataset...")
# Re-append geocoordinate floats directly onto root dataset rows matching by the 'postcode' key
df = df.merge(postcode_map, on='postcode', how='left')
# Drop records mathematically where the coordinate matching failed returning NaN values
df = df.dropna(subset=['latitude', 'longitude'])

# Feature Engineering categoricals into algorithmic numeric scales
df['property_code'] = df['property_type'].astype('category').cat.codes
df['old_new_code'] = df['old_new'].astype('category').cat.codes

# Define final array of input metrics the model views determining price
features = ['year', 'month', 'property_code', 'old_new_code', 'latitude', 'longitude']
# Explicit target target variable output string
target = 'price'

print("\n--- Splitting Data into Train (2008-2017) and Holdout Test (2018-2022) ---")
# Take historical chunk isolating random 100,000 element sampling
train_df = df[df['year'] <= 2017].sample(n=100000, random_state=42)
# Take future validation blind chunk via random 50,000 sampling points
test_df = df[df['year'] >= 2018].sample(n=50000, random_state=42)

# Set model features array targeting
X_train = train_df[features]
# Standardize heavily skewed targets via logging to force scale stabilization 
y_train = np.log1p(train_df[target])  

# Setup testing vectors mapping 
X_test = test_df[features]
# Test natively leaving targets as raw plain integer base GBP 
y_test = test_df[target]  

print("Training Geospatial LightGBM Regressor...")
# Flag initial benchmark tracking timestamp 
start_time = time.time()
# LightGBM parameters optimized for geospatial regression (Leaf-wise growth strategy using 400 branches)
lgbm_model = LGBMRegressor(n_estimators=400, num_leaves=64, learning_rate=0.05, n_jobs=-1, random_state=42)
# Fit mathematically onto vectors
lgbm_model.fit(X_train, y_train)
# Calculate speed footprint
print(f"Geospatial LightGBM Training time: {time.time() - start_time:.2f} seconds.")

print("Evaluating Geospatial LightGBM Model...")
# Gather model log space predictions
y_pred_log = lgbm_model.predict(X_test)
# Inverse exponent logs directly back out to real world values
y_pred = np.expm1(y_pred_log)

# Execute RMSE error assessment 
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# Execute MAE generic average point evaluation check 
mae = mean_absolute_error(y_test, y_pred)
# Execute out of bounds total variation accuracy bounds
r2 = r2_score(y_test, y_pred)

# Print formatting stat outputs 
print(f"Geospatial LightGBM RMSE: £{rmse:,.2f}")
print(f"Geospatial LightGBM MAE: £{mae:,.2f}")
print(f"Geospatial LightGBM R-Squared: {r2:.4f}")

# Standard status note message
print("\nSaving dataset to show prediction validation (Actual vs Predicted)...")
# Move reporting details from subset targeting directly isolating necessary rows internally into memory 
validation_df = test_df[['postcode', 'date_of_transfer', 'price', 'latitude', 'longitude']].copy()
# Rename header purely representing pure price points cleanly
validation_df.rename(columns={'price': 'Actual_Price'}, inplace=True)
# Apply LightGBM prediction vectors internally
validation_df['Predicted_Price'] = np.round(y_pred, 2)
# Create simple mathematical subtraction validation drift calculations
validation_df['Price_Difference'] = np.round(validation_df['Actual_Price'] - validation_df['Predicted_Price'], 2)

# Calculate Accuracy & Error percentage scaling for generalized business reporting standards 
validation_df['Error_%'] = np.round(np.abs(validation_df['Price_Difference'] / validation_df['Actual_Price']) * 100, 2)
# Baseline bounding edge ceiling limit to true real 100 scale limits
validation_df['Accuracy_%'] = np.clip(100 - validation_df['Error_%'], 0, 100)

# Visual standard header
print("\n--- First 15 validation records ---")
# Log top samples via shell visually validating general success layout
print(validation_df.head(15))

# Dump total calculated logic straight completely to local csv for dashboard views 
validation_df.to_csv("prediction_validation_lightgbm.csv", index=False)
# Show exiting completed notification status success cleanly   
print("\nValidation Dataset saved as 'prediction_validation_lightgbm.csv' for review!")

# Save evaluation plot
print("Generating 4C Forecast Validation Plot...")
test_df['predicted_price'] = y_pred
yearly_test_trend = test_df.groupby('year').agg({'price': 'mean', 'predicted_price': 'mean'}).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(yearly_test_trend['year'], yearly_test_trend['price'], marker="o", label="Actual Avg Price")
plt.plot(yearly_test_trend['year'], yearly_test_trend['predicted_price'], marker="x", linestyle="--", label="Forecasted Price (Geospatial LightGBM)")
plt.title("5-Year Ahead Holdout Forecast Validation (2018-2022)")
plt.xlabel("Year")
plt.ylabel("Average Property Price (£)")
plt.legend()
plt.grid(True)
plt.savefig("04C_forecast_validation.png")
plt.close()
