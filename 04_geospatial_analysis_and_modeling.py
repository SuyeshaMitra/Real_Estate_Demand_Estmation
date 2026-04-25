# Import pandas for data manipulation
import pandas as pd
# Import numpy for numerical and array operations
import numpy as np
# Import matplotlib for rendering static graph image files
import matplotlib.pyplot as plt
# Import seaborn for beautiful graph styling
import seaborn as sns
# Import pgeocode for translating postcodes into latitude/longitude coordinates
import pgeocode
# Import time module to benchmark speed execution
import time

# Import machine learning algorithms
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Import evaluation metrics to test accuracy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Apply clean visual styles
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="muted")

# Print status to terminal
print("Loading Greater London dataset...")
# Load the filtered London CSV data into a pandas dataframe
df = pd.read_csv('london_data.csv')

# Drop any rows missing key critical data ensuring quality inputs
df = df.dropna(subset=['price', 'date_of_transfer', 'postcode'])

# Print status to terminal
print("Parsing dates...")
df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
df['year'] = df['date_of_transfer'].dt.year
df['month'] = df['date_of_transfer'].dt.month

# Filter dataset to only include the 2008-2022 15-year window
df = df[(df['year'] >= 2008) & (df['year'] <= 2022)].copy()

# ==============================================================================
# --- STEP 1: ISOLATING UNIQUE POSTCODES ---
print("Extracting unique postcodes for Geospatial mapping...")
unique_postcodes = df['postcode'].unique()
print(f"Extracted {len(unique_postcodes)} unique postcodes.")

# --- STEP 2: INITIALIZING THE OFFLINE DATABASE ('nom') ---
print("Fetching Latitude and Longitude using pgeocode (Offline DB)...")
nom = pgeocode.Nominatim('gb')

# --- STEP 3 & 4: PREPARING AND QUERYING ---
outcodes = pd.Series(unique_postcodes).str.split(' ').str[0]
geo_data = nom.query_postal_code(outcodes.tolist())

# --- STEP 5: CREATING THE MASTER GEOSPATIAL MAP ---
postcode_map = pd.DataFrame({
    'postcode': unique_postcodes,
    'latitude': geo_data['latitude'].values,
    'longitude': geo_data['longitude'].values
})

# --- STEP 6: MERGING IT TO THE MAIN DATASET ---
print("Merging Geospatial coordinates back to main dataset...")
df = df.merge(postcode_map, on='postcode', how='left')

# Drop rows missing valid lat/lon
df = df.dropna(subset=['latitude', 'longitude'])
print(f"Remaining records with valid coordinates: {len(df)}")

# Feature Engineering step map text categorical types into integer codes
df['property_code'] = df['property_type'].astype('category').cat.codes
df['old_new_code'] = df['old_new'].astype('category').cat.codes
df['duration_code'] = df['duration'].astype('category').cat.codes

# Define final input features feeding into the AI model, matching baseline exactly but adding lat/lon
features = ['year', 'month', 'property_code', 'old_new_code', 'duration_code', 'latitude', 'longitude']
target = 'price'

print("\n--- Splitting Data into Train (2008-2017) and Holdout Test (2018-2022) ---")
train_df = df[df['year'] <= 2017].sample(n=100000, random_state=42)
test_df = df[df['year'] >= 2018].sample(n=50000, random_state=42)

X_train = train_df[features]
y_train = np.log1p(train_df[target])  

X_test = test_df[features]
y_test = test_df[target]  

# ==============================================================================
# --- MODEL TRAINING AND BENCHMARKING ---
# ==============================================================================

def evaluate_model(y_true, y_pred_log):
    y_pred = np.expm1(y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate Accuracy using the Accuracy Bounding Rule (Floor at 0%)
    abs_percentage_error = np.abs((y_true - y_pred) / y_true) * 100
    accuracy = 100 - abs_percentage_error
    bounded_accuracy = np.clip(accuracy, 0, 100)
    median_accuracy = np.median(bounded_accuracy)
    
    return rmse, mae, r2, median_accuracy, y_pred, bounded_accuracy

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, n_jobs=-1, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, n_jobs=-1, random_state=42)
}

results = {}
test_df_predictions = test_df.copy()

for name, model in models.items():
    print(f"\nTraining Geospatial {name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred_log = model.predict(X_test)
    rmse, mae, r2, med_acc, y_pred, bounded_acc = evaluate_model(y_test, y_pred_log)
    
    results[name] = {
        'MAE': mae,
        'Median_Accuracy': med_acc,
        'Speed': training_time
    }
    
    # Store predictions and accuracy bounded logic in the dataframe for time-series and postal charting
    pred_col = f'{name}_pred'
    acc_col = f'{name}_acc'
    err_col = f'{name}_error'
    test_df_predictions[pred_col] = y_pred
    test_df_predictions[acc_col] = bounded_acc
    test_df_predictions[err_col] = np.abs(test_df_predictions['price'] - y_pred)
    
    print(f"--- Model Evaluation: {name} ---")
    print(f"Speed: {training_time:.2f}s")
    print(f"MAE: £{mae:,.2f}")
    print(f"Median Accuracy: {med_acc:.2f}%")

# ==============================================================================
# --- TIME-SERIES AND GEOSPATIAL ERROR AGGREGATION ---
# ==============================================================================

print("\nAggregating time-series and spatial patterns...")
# 1. Yearly Aggregation
yearly_cols = {'price': 'mean'}
for name in models.keys():
    yearly_cols[f'{name}_pred'] = 'mean'
    yearly_cols[f'{name}_error'] = 'mean'
    yearly_cols[f'{name}_acc'] = 'median'
yearly_trend = test_df_predictions.groupby('year').agg(yearly_cols).reset_index()

# 2. Monthly Aggregation
monthly_cols = {'price': 'mean'}
for name in models.keys():
    monthly_cols[f'{name}_pred'] = 'mean'
    monthly_cols[f'{name}_error'] = 'mean'
    monthly_cols[f'{name}_acc'] = 'median'
monthly_trend = test_df_predictions.groupby('month').agg(monthly_cols).reset_index()

# 3. Postcode Aggregation (Geospatial Variance)
postcode_cols = {'price': 'mean'}
for name in models.keys():
    postcode_cols[f'{name}_error'] = 'mean'
    postcode_cols[f'{name}_acc'] = 'median'
# We group strictly by outcode (first half of postcode) for charting density
test_df_predictions['outcode'] = test_df_predictions['postcode'].str.split(' ').str[0]
postcode_trend = test_df_predictions.groupby('outcode').agg(postcode_cols).reset_index()

# Filter to only the top 50 postcodes by volume so the chart is readable
top_postcodes = test_df_predictions['outcode'].value_counts().nlargest(50).index
postcode_trend_filtered = postcode_trend[postcode_trend['outcode'].isin(top_postcodes)]
# Sort geographically or by highest error
postcode_trend_filtered = postcode_trend_filtered.sort_values(by='LightGBM_error', ascending=False)


# ==============================================================================
# --- GENERATING VISUALIZATIONS ---
# ==============================================================================
print("\nGenerating physically mapped visualization charts...")

# 1. Historical Data Chart (2008-2022)
hist_trend = df.groupby('year')['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=hist_trend, x='year', y='price', marker="o", color="black")
plt.title("Geospatial Historical Price Trend (2008 - 2022)")
plt.xlabel("Year")
plt.ylabel("Average Property Price (£)")
plt.grid(True)
plt.savefig("04_historical_trend.png")
plt.close()

# 2. Forecast Validation - Yearly
plt.figure(figsize=(10, 6))
plt.plot(yearly_trend['year'], yearly_trend['price'], marker="o", linewidth=2, color="black", label="Actual Avg Price")
plt.plot(yearly_trend['year'], yearly_trend['Random Forest_pred'], marker="x", linestyle="--", label="Forecasted (Random Forest)")
plt.plot(yearly_trend['year'], yearly_trend['XGBoost_pred'], marker="^", linestyle="--", label="Forecasted (XGBoost)")
plt.plot(yearly_trend['year'], yearly_trend['LightGBM_pred'], marker="d", linestyle="--", label="Forecasted (LightGBM)")
plt.title("Geospatial Yearly Forecast Validation (2018-2022)")
plt.xlabel("Year")
plt.ylabel("Average Property Price (£)")
plt.legend()
plt.grid(True)
plt.savefig("04_forecast_validation_yearly.png")
plt.close()

# 3. Forecast Validation - Monthly Seasonality
plt.figure(figsize=(10, 6))
plt.plot(monthly_trend['month'], monthly_trend['price'], marker="o", linewidth=2, color="black", label="Actual Avg Price")
plt.plot(monthly_trend['month'], monthly_trend['Random Forest_pred'], marker="x", linestyle="--", label="Forecasted (Random Forest)")
plt.plot(monthly_trend['month'], monthly_trend['XGBoost_pred'], marker="^", linestyle="--", label="Forecasted (XGBoost)")
plt.plot(monthly_trend['month'], monthly_trend['LightGBM_pred'], marker="d", linestyle="--", label="Forecasted (LightGBM)")
plt.title("Geospatial Monthly Cyclical Forecast Validation (Months 1-12)")
plt.xlabel("Month")
plt.ylabel("Average Property Price (£)")
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True)
plt.savefig("04_forecast_validation_monthly.png")
plt.close()

# 4. Accuracy Trend - Yearly
plt.figure(figsize=(10, 6))
plt.plot(yearly_trend['year'], yearly_trend['Random Forest_acc'], marker="x", linestyle="-", label="Random Forest Accuracy")
plt.plot(yearly_trend['year'], yearly_trend['XGBoost_acc'], marker="^", linestyle="-", label="XGBoost Accuracy")
plt.plot(yearly_trend['year'], yearly_trend['LightGBM_acc'], marker="d", linestyle="-", label="LightGBM Accuracy")
plt.title("Geospatial Yearly Median Accuracy Trend (2018 - 2022)")
plt.xlabel("Year")
plt.ylabel("Median Accuracy (%)")
plt.legend()
plt.grid(True)
plt.savefig("04_accuracy_trend_yearly.png")
plt.close()

# 5. Accuracy Trend - Monthly Seasonality
plt.figure(figsize=(10, 6))
plt.plot(monthly_trend['month'], monthly_trend['Random Forest_acc'], marker="x", linestyle="-", label="Random Forest Accuracy")
plt.plot(monthly_trend['month'], monthly_trend['XGBoost_acc'], marker="^", linestyle="-", label="XGBoost Accuracy")
plt.plot(monthly_trend['month'], monthly_trend['LightGBM_acc'], marker="d", linestyle="-", label="LightGBM Accuracy")
plt.title("Geospatial Monthly Seasonality Median Accuracy Trend (Months 1-12)")
plt.xlabel("Month")
plt.ylabel("Median Accuracy (%)")
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True)
plt.savefig("04_accuracy_trend_monthly.png")
plt.close()

# 6. Postcode Error Distribution (NEW SPATIAL CHART)
plt.figure(figsize=(14, 7))
# Bar chart plotting the absolute physical MAE error distributed physically across the top 50 districts
x = np.arange(len(postcode_trend_filtered['outcode']))
width = 0.25

plt.bar(x - width, postcode_trend_filtered['Random Forest_error'], width, label='Random Forest Error')
plt.bar(x, postcode_trend_filtered['XGBoost_error'], width, label='XGBoost Error')
plt.bar(x + width, postcode_trend_filtered['LightGBM_error'], width, label='LightGBM Error')

plt.title("Geospatial Model Error Distribution (Top 50 Most Active Postcodes)")
plt.xlabel("London Postcode Outcode Districts")
plt.ylabel("Mean Absolute Error (£)")
plt.xticks(x, postcode_trend_filtered['outcode'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("04_error_distribution_by_postcode.png")
plt.close()

print("Execution and Analytics fully completed successfully!")
