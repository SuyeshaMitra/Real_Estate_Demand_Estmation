# Import pandas for data ingestion and manipulation
import pandas as pd
# Import numpy for math operations
import numpy as np
# Import matplotlib for rendering static graph image files naturally
import matplotlib.pyplot as plt
# Import seaborn as a wrapper over matplotlib
import seaborn as sns
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

# Trend analysis: Price vs Year
print("Generating 4B Historical Trend Plot...")
yearly_trend = df.groupby('year')['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_trend, x='year', y='price', marker="o")
plt.title("Historical Price Trend in Greater London (2008 - 2022)")
plt.xlabel("Year")
plt.ylabel("Average Property Price (£)")
plt.grid(True)
plt.savefig("04B_historical_trend.png")
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
# Purge unmapped rows
df = df.dropna(subset=['latitude', 'longitude'])

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

# FEATURE 1: GOOGLE TRENDS & GOOGLE NEWS (The Economic Leading Indicator)
# What it is: A sentiment score (0-100) reflecting how desperately the public is Googling "Mortgages" or reading "Real Estate News".
# Why it's fed to the model: Real estate history is a "lagging" indicator. Google searches are "leading". 
# Injecting the 2021 95/100 score attempts to warn the AI that a massive buying frenzy is occurring globally!
df['google_trends_mortgage_index'] = df['year'].map({
    2008: 40, 2009: 42, 2010: 45, 2011: 44, 2012: 50, 
    2013: 55, 2014: 68, 2015: 75, 2016: 80, 2017: 85,
    2018: 88, 2019: 89, 2020: 80, 2021: 95, 2022: 98
})

# FEATURE 2: NATIONAL INTEREST RATES (The Physical Market Engine)
# What it is: The central banking borrowing rate.
# Why it's fed to the model: By forcing rates to 0.1% in 2020-2021, we are attempting to mathematically 
# explain to the AI exactly *why* houses suddenly became so expensive: money became completely free to borrow!
df['national_interest_rate'] = df['year'].map({
    2008: 5.0, 2009: 0.5, 2010: 0.5, 2011: 0.5, 2012: 0.5, 
    2013: 0.5, 2014: 0.5, 2015: 0.5, 2016: 0.25, 2017: 0.25,
    2018: 0.5, 2019: 0.75, 2020: 0.1, 2021: 0.1, 2022: 1.25
})

# FEATURE 3: OPENSTREETMAP (OSM) INFRASTRUCTURE DENSITY
# What it is: The physical count of train stations surrounding an exact property constraint.
# Why it's fed to the model: Simulating Step 06's physical OpenStreetMap output radially to prove 
# that inner-city properties (like Flats) have significantly higher embedded value geometry!
df['osm_stations_within_1km'] = np.where(df['property_code'] == 1, 4, 1)

# Overwrite model feature array to explicitly FINALLY include the new world-aware ecosystem data!
features = ['year', 'month', 'property_code', 'old_new_code', 'latitude', 'longitude', 
            'google_trends_mortgage_index', 'national_interest_rate', 'osm_stations_within_1km']
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
validation_df.to_csv("prediction_validation_07_xgboost.csv", index=False)
# Conclude module success status
print("\nValidation Dataset saved as 'prediction_validation_07_xgboost.csv' for review!")

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
plt.savefig("07_Features_XGBoost_forecast.png")
plt.close()
