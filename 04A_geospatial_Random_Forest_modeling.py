# Import pandas for data manipulation
import pandas as pd
# Import numpy for numerical and array operations
import numpy as np
# Import pgeocode for translating postcodes into latitude/longitude coordinates
import pgeocode
# Import time module to benchmark
import time
# Import Random Forest machine learning algorithm
from sklearn.ensemble import RandomForestRegressor
# Import evaluation metrics to test accuracy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Print status to terminal
print("Loading Greater London dataset...")
# Load the filtered London CSV data into a pandas dataframe
df = pd.read_csv('london_data.csv')

# Drop any rows missing key critical data ensuring quality inputs
df = df.dropna(subset=['price', 'date_of_transfer', 'postcode'])

# Print status to terminal
print("Parsing dates...")
# Convert transfer date string column to a datetime object
df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
# Extract the year into a numeric column feature
df['year'] = df['date_of_transfer'].dt.year
# Extract the month into a numeric column feature
df['month'] = df['date_of_transfer'].dt.month

# Filter dataset to only include the 2008-2022 15-year window
df = df[(df['year'] >= 2008) & (df['year'] <= 2022)].copy()

# Print status to terminal
print("Extracting unique postcodes for Geospatial mapping...")
# Isolate all unique postcodes to minimize API calls to pgeocode
unique_postcodes = df['postcode'].unique()

print(f"Fetching Latitude and Longitude for {len(unique_postcodes)} unique postcodes using pgeocode...")
# Initialize the UK specific postcode geocoder
nom = pgeocode.Nominatim('gb')

# To speed up the process we split the postcode string and just take the first half (the outcode)
# This provides generalized spatial mapping suitable for large-scale block regression
outcodes = pd.Series(unique_postcodes).str.split(' ').str[0]
# Query the geocoder API for coordinates mapping to those outcodes
geo_data = nom.query_postal_code(outcodes.tolist())

# Build a fast mapping dictionary dataframe using exact original postcodes against fetched coordinates
postcode_map = pd.DataFrame({
    'postcode': unique_postcodes,
    'latitude': geo_data['latitude'].values,
    'longitude': geo_data['longitude'].values
})

print("Merging Geospatial data back to main dataset...")
# Merge the coordinates back into the main dataset matching by 'postcode'
df = df.merge(postcode_map, on='postcode', how='left')

# Track dataset size before dropping bad geospatial data
initial_len = len(df)
# Drop any rows where we failed to convert the postcode to valid coordinates
df = df.dropna(subset=['latitude', 'longitude'])
# Print how many rows were dropped due to mapping failures
print(f"Dropped {initial_len - len(df)} records due to missing geospatial data. Remaining: {len(df)}")

# Feature Engineering step map text categorical types into integer codes
df['property_code'] = df['property_type'].astype('category').cat.codes
df['old_new_code'] = df['old_new'].astype('category').cat.codes

# Define final input features feeding into the AI model, including lat/lon
features = ['year', 'month', 'property_code', 'old_new_code', 'latitude', 'longitude']
# Define the prediction target variable
target = 'price'

print("\n--- Splitting Data into Train (2008-2017) and Holdout Test (2018-2022) ---")
# Take a random 100k subset from 2008 to 2017 to train the algorithm on 
train_df = df[df['year'] <= 2017].sample(n=100000, random_state=42)
# Take a random 50k subset from 2018 to 2022 to bench its forward predicting abilities
test_df = df[df['year'] >= 2018].sample(n=50000, random_state=42)

# Set up features dataframe
X_train = train_df[features]
# Standardize the highly varied prices using log transformation to improve learning slope
y_train = np.log1p(train_df[target])  

# Setup testing features dataframe
X_test = test_df[features]
# Set testing targets as raw base units (pure GBP format)
y_test = test_df[target]  

print("Training Geospatial Random Forest Regressor...")
# Start recording processing time
start_time = time.time()
# Instantiate the regressor. Note: Deeper tree (max_depth=20) used because Lat/Lon spatial grouping benefits from depth
rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
# Trigger the actual mathematical training step
rf_model.fit(X_train, y_train)
# Print out specific elapsed computation time
print(f"Geospatial RF Training time: {time.time() - start_time:.2f} seconds.")

# Predict step over test data
print("Evaluating Geospatial Model...")
# Gather log-based answers
y_pred_log = rf_model.predict(X_test)
# Inverse the log math to yield baseline GBP pricing 
y_pred = np.expm1(y_pred_log)

# Calculate standard regression metrics against raw testing base unit price targets
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display specific formatted statistics out
print(f"Geospatial RMSE: £{rmse:,.2f}")
print(f"Geospatial MAE: £{mae:,.2f}")
print(f"Geospatial R-Squared: {r2:.4f}")

# Show Side-by-Side Validation Output context
print("\nSaving dataset to show prediction validation (Actual vs Predicted)...")
# Copy mapping details to output frame tracking physical outputs
validation_df = test_df[['postcode', 'date_of_transfer', 'price', 'latitude', 'longitude']].copy()
# Rename current active price column for output reporting clarity 
validation_df.rename(columns={'price': 'Actual_Price'}, inplace=True)
# Append newly predicted numbers mapped side-by-side rounded slightly 
validation_df['Predicted_Price'] = np.round(y_pred, 2)
# Mathematically formulate raw drift metrics
validation_df['Price_Difference'] = np.round(validation_df['Actual_Price'] - validation_df['Predicted_Price'], 2)

# Calculate Error precision percentages for simple business translation representing % off
validation_df['Error_%'] = np.round(np.abs(validation_df['Price_Difference'] / validation_df['Actual_Price']) * 100, 2)
# Floor accuracy mathematically mapping at 0% bounding edge instead of negative
validation_df['Accuracy_%'] = np.clip(100 - validation_df['Error_%'], 0, 100)

# Display terminal preview table for review
print("\n--- First 15 validation records ---")
print(validation_df.head(15))

# Export the entire detailed validation log frame to CSV
validation_df.to_csv("prediction_validation_randomforest.csv", index=False)
print("\nValidation Dataset saved as 'prediction_validation_randomforest.csv' for review!")
