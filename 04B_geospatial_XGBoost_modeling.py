import pandas as pd
import numpy as np
import pgeocode
import time
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Loading Greater London dataset...")
df = pd.read_csv('london_data.csv')

# Drop missing critical data
df = df.dropna(subset=['price', 'date_of_transfer', 'postcode'])

print("Parsing dates...")
df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
df['year'] = df['date_of_transfer'].dt.year
df['month'] = df['date_of_transfer'].dt.month

# Filter for the 2008-2022 window as before
df = df[(df['year'] >= 2008) & (df['year'] <= 2022)].copy()

print("Extracting unique postcodes for Geospatial mapping...")
unique_postcodes = df['postcode'].unique()

print(f"Fetching Latitude and Longitude for {len(unique_postcodes)} unique postcodes using pgeocode...")
nom = pgeocode.Nominatim('gb')

outcodes = pd.Series(unique_postcodes).str.split(' ').str[0]
geo_data = nom.query_postal_code(outcodes.tolist())

postcode_map = pd.DataFrame({
    'postcode': unique_postcodes,
    'latitude': geo_data['latitude'].values,
    'longitude': geo_data['longitude'].values
})

print("Merging Geospatial data back to main dataset...")
df = df.merge(postcode_map, on='postcode', how='left')
df = df.dropna(subset=['latitude', 'longitude'])

# Feature Engineering
df['property_code'] = df['property_type'].astype('category').cat.codes
df['old_new_code'] = df['old_new'].astype('category').cat.codes

features = ['year', 'month', 'property_code', 'old_new_code', 'latitude', 'longitude']
target = 'price'

print("\n--- Splitting Data into Train (2008-2017) and Holdout Test (2018-2022) ---")
train_df = df[df['year'] <= 2017].sample(n=100000, random_state=42)
test_df = df[df['year'] >= 2018].sample(n=50000, random_state=42)

X_train = train_df[features]
y_train = np.log1p(train_df[target])  

X_test = test_df[features]
y_test = test_df[target]  

print("Training Geospatial XGBoost Regressor...")
start_time = time.time()
# XGBoost parameters optimized for geospatial regression
xgb_model = XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.05, n_jobs=-1, random_state=42)
xgb_model.fit(X_train, y_train)
print(f"Geospatial XGBoost Training time: {time.time() - start_time:.2f} seconds.")

print("Evaluating Geospatial XGBoost Model...")
y_pred_log = xgb_model.predict(X_test)
y_pred = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Geospatial XGBoost RMSE: £{rmse:,.2f}")
print(f"Geospatial XGBoost MAE: £{mae:,.2f}")
print(f"Geospatial XGBoost R-Squared: {r2:.4f}")

print("\nSaving dataset to show prediction validation (Actual vs Predicted)...")
validation_df = test_df[['postcode', 'date_of_transfer', 'price', 'latitude', 'longitude']].copy()
validation_df.rename(columns={'price': 'Actual_Price'}, inplace=True)
validation_df['Predicted_Price'] = np.round(y_pred, 2)
validation_df['Price_Difference'] = np.round(validation_df['Actual_Price'] - validation_df['Predicted_Price'], 2)

# Calculate Accuracy & Error
validation_df['Error_%'] = np.round(np.abs(validation_df['Price_Difference'] / validation_df['Actual_Price']) * 100, 2)
validation_df['Accuracy_%'] = np.clip(100 - validation_df['Error_%'], 0, 100)

print("\n--- First 15 validation records ---")
print(validation_df.head(15))

validation_df.to_csv("prediction_validation_xgb.csv", index=False)
print("\nValidation Dataset saved as 'prediction_validation_xgb.csv' for review!")
