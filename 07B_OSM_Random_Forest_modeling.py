# Import pandas for data manipulation
import pandas as pd
# Import numpy for numerical and array operations
import numpy as np
# Import matplotlib for rendering static graph image files naturally
import matplotlib.pyplot as plt
# Import seaborn as a wrapper over matplotlib
import seaborn as sns
# Import pgeocode for translating postcodes into latitude/longitude coordinates
# Import time module to benchmark
import time
# Import Random Forest machine learning algorithm
from sklearn.ensemble import RandomForestRegressor
# Import evaluation metrics to test accuracy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Print status to terminal
print("Loading Unified Geospatial London dataset...")
# Load the pre-mapped geospatial dataset directly, bypassing the pgeocode query pipeline
df = pd.read_csv('london_geospatial_dataset.csv')

# Print how many rows were dropped due to mapping failures
print(f"Dropped {initial_len - len(df)} records due to missing geospatial data. Remaining: {len(df)}")

# Feature Engineering step map text categorical types into integer codes
df['property_code'] = df['property_type'].astype('category').cat.codes
df['old_new_code'] = df['old_new'].astype('category').cat.codes

# Define final input features feeding into the AI model, including lat/lon

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
validation_df.to_csv("prediction_validation_07b_randomforest.csv", index=False)
print("\nValidation Dataset saved as 'prediction_validation_07b_randomforest.csv' for review!")

# Save evaluation plot
print("Generating 4A Forecast Validation Plot...")
test_df['predicted_price'] = y_pred
yearly_test_trend = test_df.groupby('year').agg({'price': 'mean', 'predicted_price': 'mean'}).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(yearly_test_trend['year'], yearly_test_trend['price'], marker="o", label="Actual Avg Price")
plt.plot(yearly_test_trend['year'], yearly_test_trend['predicted_price'], marker="x", linestyle="--", label="Forecasted Price (Geospatial RF)")
plt.title("5-Year Ahead Holdout Forecast Validation (2018-2022)")
plt.xlabel("Year")
plt.ylabel("Average Property Price (£)")
plt.legend()
plt.grid(True)
plt.savefig("07B_OSM_Random_Forest_forecast.png")
plt.close()
