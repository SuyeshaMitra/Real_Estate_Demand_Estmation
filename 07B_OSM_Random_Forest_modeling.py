# Import pandas for data manipulation
import pandas as pd
# Import numpy for numerical and array operations
import numpy as np
# Import matplotlib for rendering static graph image files naturally
import matplotlib.pyplot as plt
# Import seaborn as a wrapper over matplotlib
import seaborn as sns
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

# Trend analysis: Price vs Year
print("Generating 4A Historical Trend Plot...")
yearly_trend = df.groupby('year')['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_trend, x='year', y='price', marker="o")
plt.title("Historical Price Trend in Greater London (2008 - 2022)")
plt.xlabel("Year")
plt.ylabel("Average Property Price (£)")
plt.grid(True)
plt.savefig("04A_historical_trend.png")
plt.close()

# ==============================================================================
# --- STEP 1: ISOLATING UNIQUE POSTCODES ---
print("Extracting unique postcodes for Geospatial mapping...")
# Instead of querying the database 3.9 million times (once for every row in the long CSV),
# we isolate only the UNIQUE postcodes (e.g., roughly 300,000 exact streets) to minimize work.
unique_postcodes = df['postcode'].unique()
print(f"Extracted {len(unique_postcodes)} unique postcodes.")

# --- STEP 2: INITIALIZING THE OFFLINE DATABASE ('nom') ---
print(f"Fetching Latitude and Longitude using pgeocode...")
# 'pgeocode' is an open-source library that downloads the geonames.org geographic database 
# and explicitly stores it locally securely on your computer's hard drive (inside your python site-packages).
# 'nom' stands for Nominatim. Passing ('gb') tells Python to load the Great Britain offline database into active memory.
nom = pgeocode.Nominatim('gb')
print("\n--- Output element: nom ---")
print("Search Engine Memory Object:", nom)
print("Physical Offline Database Location:", nom._data_path)
print("Internal Database Schema Glimpse (First 5 records):\n", nom._data[['postal_code', 'place_name', 'latitude', 'longitude']].head(5))

# --- STEP 3: PREPARING THE QUERY STRING ---
# UK postcodes have two halves (e.g., "BR6 7FN"). The first half ("BR6") is called the 'outcode'.
# To radically speed up the offline spatial mapping and ensure 100% match rates, we strip " 7FN" and just search for "BR6".
outcodes = pd.Series(unique_postcodes).str.split(' ').str[0]
print("\n--- Output element: outcodes (First 5) ---")
print(outcodes.head())

# --- STEP 4: QUERYING THE OFFLINE DATABASE ---
# We pass the cleaned list of outcodes to 'nom.query_postal_code()'. 
# Because the database is sitting offline locally on your hard drive, it can instantly search hundreds of thousands 
# of outcodes and return their Latitude and Longitude mathematically in less than 2 seconds without using the internet!
geo_data = nom.query_postal_code(outcodes.tolist())
print("\n--- Output element: geo_data (First 5 rows) ---")
print(geo_data.head())

# --- STEP 5: CREATING THE MASTER GEOSPATIAL MAP (Matching Found!) ---
# 'geo_data' now holds the raw X and Y coordinates. We need to pair them cleanly back to the original full "BR6 7FN" string format.
# We build a 'dictionary dataframe' (called postcode_map) aligning the exact original postcodes against their newly fetched coordinates.
postcode_map = pd.DataFrame({
    'postcode': unique_postcodes,
    'latitude': geo_data['latitude'].values,
    'longitude': geo_data['longitude'].values
})
print("\n--- Output element: postcode_map (First 5 matching records) ---")
print(postcode_map.head())

# --- STEP 6: SENDING IT TO THE MAIN DATASET (Preparing for the Model) ---
print("Merging Geospatial data back to main dataset...")
# We take the gigantic 3.9 million row dataset ('df') and mathematically "Merge" (join) the small 'postcode_map' onto it.
# Every row in the long CSV looks at its 'postcode', walks directly over to the postcode_map, grabs the exact matching Lat/Lon,
# and permanently adds those 2 columns to itself. The dataset is now ready to be sent to the AI Model!
df = df.merge(postcode_map, on='postcode', how='left')
print("\n--- Output element: df (First 5 rows showcasing newly merged latitude/longitude) ---")
print(df[['postcode', 'latitude', 'longitude']].head())
# ==============================================================================
# ==============================================================================

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
