# Import core data manipulation library
import pandas as pd
# Import numpy for math operations and array handling
import numpy as np
# Import matplotlib for generating static visualizations and charts
import matplotlib.pyplot as plt
# Import seaborn as a wrapper over matplotlib for better statistical plot aesthetics
import seaborn as sns
# Import machine learning train/test splitting functionality
from sklearn.model_selection import train_test_split
# Import the Random Forest regression algorithm
from sklearn.ensemble import RandomForestRegressor
# Import the Multi-Layer Perceptron (Neural Network) regression algorithm
from sklearn.neural_network import MLPRegressor
# Import XGBoost and LightGBM regression algorithms
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# Import metrics to evaluate how well our models perform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Import time module to benchmark model training speeds
import time
# Import os module for path operations
import os

# Inform the terminal user we are starting the data load
print("Loading Greater London dataset...")
# Define path to our previously prepared regional dataset
data_path = 'london_data.csv'
# Load the CSV data entirely into RAM as a Pandas DataFrame
df = pd.read_csv(data_path)

# Drop rows that lack a recorded price, transfer date, or district as they cannot be used
df = df.dropna(subset=['price', 'date_of_transfer', 'district'])

# Inform the terminal user we are formatting features
print("Parsing dates and engineering features...")
# Convert the plain text date string into a genuine datetime object for pandas
df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
# Extract just the year as an independent standalone number/feature
df['year'] = df['date_of_transfer'].dt.year
# Extract just the month as an independent standalone number/feature
df['month'] = df['date_of_transfer'].dt.month

# Filter the dataframe to only keep records occurring between 2008 and 2022
# This ensures we are only modeling a stable 15-year period for our scenario
df = df[(df['year'] >= 2008) & (df['year'] <= 2022)].copy()

# Print out exactly how many rows are remaining after our temporal filtering
print(f"Dataset size after 2008-2022 filter: {len(df)}")

# Create a new dataframe isolating the average property price grouped by year
yearly_trend = df.groupby('year')['price'].mean().reset_index()
# Instantiate a new blank Matplotlib figure with specific dimensions (10x6 inches)
plt.figure(figsize=(10, 6))
# Instruct Seaborn to draw a lineplot mapping years to prices, marking points with circles
sns.lineplot(data=yearly_trend, x='year', y='price', marker="o")
# Add the main title at the top of the chart
plt.title("Historical Price Trend in Greater London (2008 - 2022)")
# Label the X-axis mapping to time
plt.xlabel("Year")
# Label the Y-axis mapping to cost
plt.ylabel("Average Property Price (£)")
# Enable grid lines inside the chart so values are easier to read
plt.grid(True)
# Save the rendered graph out directly to a local image file
plt.savefig("03_historical_trend.png")
# Close out the figure to release RAM
plt.close()

# Begin engineering non-numeric attributes down to integers
# Overwrite property_type strings (like 'Detached') into categorical ID integers (like 1, 2, 3)
df['property_code'] = df['property_type'].astype('category').cat.codes
# Overwrite old_new strings into categorical ID integers
df['old_new_code'] = df['old_new'].astype('category').cat.codes
# Overwrite duration strings (Leasehold/Freehold) into categorical ID integers
df['duration_code'] = df['duration'].astype('category').cat.codes
# Overwrite distinct geographic district names into categorical ID integers
df['district_code'] = df['district'].astype('category').cat.codes

# Define the precise list of column attributes that will act as inputs identifying the price
features = ['year', 'month', 'property_code', 'old_new_code', 'duration_code', 'district_code']
# Define 'price' as the target column the models will try to guess or predict
target = 'price'

# Output status notification out to the user describing standard train/test split rules
print("Splitting Data into Train (2008-2017) and Holdout Test (2018-2022) ...")
# Take exactly 100,000 random records from 2008-2017 to train with to sidestep infinite compute loops
train_df = df[df['year'] <= 2017].sample(n=100000, random_state=42)
# Take exactly 50,000 random records spanning from 2018-2022 to benchmark predictions against
test_df = df[df['year'] >= 2018].sample(n=50000, random_state=42)

# Isolate training attributes dataframe
X_train = train_df[features]
# Isolate training answers, but convert standard price into log-space to handle skewed outliers
y_train = np.log1p(train_df[target])  

# Isolate validation attributes dataframe
X_test = test_df[features]
# Isolate validation answers natively as raw correct values instead of logs for pure visual comparison
y_test = test_df[target]  

# Array to hold charting metrics dynamically
plot_models = []
plot_maes = []
plot_times = []
plot_accuracies = []

# Output status
print("Training Random Forest Regressor...")
# --- OUTCOME DIAGNOSTICS: RANDOM FOREST ---
# Resulting MAE: ~£470k | Median Accuracy: 74.78% | Speed: ~0.39s
# Why?: Random forest offers a very stable baseline prediction. However, 
# simply mapping District text names lacks the spatial depth to tightly follow the pricing boom causing
# a high error rate overall.
# Start recording time
start_time = time.time()
# Instantiate the Random Forest algorithm configuration (50 trees, cap depth at 15 splits to dodge overfitting, use all CPU cores)
rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
# Execute mathematical model fitting against the provided training shapes
rf_model.fit(X_train, y_train)
# Calculate total duration by subtracting start from current time
rf_time = time.time() - start_time
# Display exact compute time taken
print(f"Random Forest Training time: {rf_time:.2f} seconds.")

# Output status
print("Training Neural Network (MLP) Regressor...")
# --- OUTCOME DIAGNOSTICS: NEURAL NETWORK ---
# Resulting MAE: ~£546k | Median Accuracy: 65.93% | Speed: ~2.22s
# Why?: Neural networks require intense depth and complexity. Because we are only passing
# basic categorical integers (like property code or district code), it fundamentally crashes,
# resulting in the worst accuracy out of the entire pipeline.
# Start recording time
start_time = time.time()
# Instantiate a Multi Layer Perceptron (two hidden layers sized 64 and 32 neurons, nonlinear RELU activation logic)
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, activation='relu', random_state=42)
# Execute mathematical model fitting
mlp_model.fit(X_train, y_train)
# Calculate duration
mlp_time = time.time() - start_time
# Display exact computing time
print(f"Neural Network Training time: {mlp_time:.2f} seconds.")

# Output status
print("Training XGBoost Regressor...")
# --- OUTCOME DIAGNOSTICS: XGBOOST ---
# Resulting MAE: ~£494k | Median Accuracy: 73.82% | Speed: ~2.27s
# Why?: XGBoost sequentially hyper-focuses on errors. Without geospatial GPS coordinates, 
# chasing errors using only basic text vectors forces the model to heavily overfit, inflating the error limit.
# Start recording time
start_time = time.time()
xgb_model = XGBRegressor(n_estimators=100, max_depth=10, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start_time
print(f"XGBoost Training time: {xgb_time:.2f} seconds.")

# Output status
print("Training LightGBM Regressor...")
# --- OUTCOME DIAGNOSTICS: LIGHTGBM (BASELINE WINNER) ---
# Resulting MAE: ~£456k | Median Accuracy: 75.68% | Speed: ~1.63s
# Why?: Even without Latitude/Longitude coordinates, LightGBM dominates. Its leaf-wise 
# histogram bins can naturally isolate extreme wealth properties far better than text depth averages.
# Start recording time
start_time = time.time()
lgbm_model = LGBMRegressor(n_estimators=100, num_leaves=64, random_state=42)
lgbm_model.fit(X_train, y_train)
lgbm_time = time.time() - start_time
print(f"LightGBM Training time: {lgbm_time:.2f} seconds.")

# Define an isolated helper function that prints standard benchmark statistics taking the name/model directly
def evaluate_model(name, model, execution_time):
    # Generates log-based number answers based on validating against out-of-sample data points
    y_pred_log = model.predict(X_test)
    # Exponentiate the logs mathematically back to plain raw real GBP costs for interpretability
    y_pred = np.expm1(y_pred_log)
    
    # Calculate the Root Mean Squared benchmark (heavily penalizes massive wild outliers)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Calculate the Mean Absolute benchmark (a normal average of error margins)
    mae = mean_absolute_error(y_test, y_pred)
    # Calculate R^2 representing overall accuracy variance the model natively accounted for
    r2 = r2_score(y_test, y_pred)
    
    # A(i) Check Absolute Error (MAE), Aggregate Median Accuracy, Execution Processing Speed
    # A(iii) Explore Accuracy Rule: Accuracy is 100% minus the absolute percentage error.
    # A(iv) Check all calculations: We enforce a strict mathematical floor at 0% for all models 
    # across all features to guarantee values are accurate and perfectly consistent.
    absolute_percentage_errors = np.abs((y_test - y_pred) / y_test)
    accuracy_array = np.clip(100 - (absolute_percentage_errors * 100), 0, 100)
    median_accuracy_percentage = np.median(accuracy_array)
    
    # Store dynamic stats
    plot_models.append(name)
    plot_maes.append(mae)
    plot_times.append(execution_time)
    plot_accuracies.append(median_accuracy_percentage)
    
    # Spacing
    print(f"\n--- Model Evaluation: {name} ---")
    # Output the exact £ RMSE formatting visually with commas
    print(f"RMSE: £{rmse:,.2f}")
    # Output the exact £ MAE cleanly
    print(f"MAE: £{mae:,.2f}")
    # Show R2 scoring matrix out of 1.0 peak
    print(f"R-Squared (out-of-sample): {r2:.4f}")
    # Show median accuracy
    print(f"Median Accuracy: {median_accuracy_percentage:.2f}%")
    
    # Function returns the generated physical predictions array
    return y_pred

# Force function evaluations to execute over both loaded pre-trained models
rf_pred = evaluate_model("Random Forest", rf_model, rf_time)
mlp_pred = evaluate_model("Neural Network", mlp_model, mlp_time)
xgb_pred = evaluate_model("XGBoost", xgb_model, xgb_time)
lgbm_pred = evaluate_model("LightGBM", lgbm_model, lgbm_time)

# Map the exact predictions back into the original dataset frame for a pure unified view
test_df['rf_predicted_price'] = rf_pred
test_df['mlp_predicted_price'] = mlp_pred
test_df['xgb_predicted_price'] = xgb_pred
test_df['lgbm_predicted_price'] = lgbm_pred

# Compute structured, granular error metrics rigorously ensuring evaluation limits are uniform
models_dict = {'rf': 'Random Forest', 'mlp': 'Neural Network', 'xgb': 'XGBoost', 'lgbm': 'LightGBM'}

for code in models_dict.keys():
    # Calculate pure absolute error physically per single house record natively
    test_df[f'{code}_abs_err'] = np.abs(test_df['price'] - test_df[f'{code}_predicted_price'])
    # Floor accuracy strictly at 0% baseline removing negative drifts completely from skewing models
    test_df[f'{code}_accuracy'] = np.clip(100 - (test_df[f'{code}_abs_err'] / test_df['price'] * 100), 0, 100)

print("\n=======================================================")
print("          --- 5-YEAR AGGREGATE SUMMARY (BY YEAR) ---           ")
print("=======================================================")
# A(v) Calculate the Accuracy, Error and Speed - For 5 Years 
# A(vi) How is the Error pattern coming on Years wise: We track how the error naturally floats per year.
# Compute exact historical averages explicitly tracking the yearly error pattern completely isolated
yearly_test_trend = test_df.groupby('year').agg(
    price=('price', 'mean'),
    rf_predicted=('rf_predicted_price', 'mean'), rf_mae=('rf_abs_err', 'mean'), rf_acc=('rf_accuracy', 'median'),
    mlp_predicted=('mlp_predicted_price', 'mean'), mlp_mae=('mlp_abs_err', 'mean'), mlp_acc=('mlp_accuracy', 'median'),
    xgb_predicted=('xgb_predicted_price', 'mean'), xgb_mae=('xgb_abs_err', 'mean'), xgb_acc=('xgb_accuracy', 'median'),
    lgbm_predicted=('lgbm_predicted_price', 'mean'), lgbm_mae=('lgbm_abs_err', 'mean'), lgbm_acc=('lgbm_accuracy', 'median')
).reset_index()

# Display terminal readout
print(yearly_test_trend.to_string(index=False))

print("\n=======================================================")
print("      --- SEASONALITY AGGREGATE SUMMARY (BY MONTH) ---         ")
print("=======================================================")
# A(v) Calculate the Accuracy, Error and Speed - Every Monthly Average aswell
# A(vi) How is the Error pattern coming on Months wise separately:
# Compute exact historical monthly averages tracking the cyclic seasonality pattern error completely isolated
monthly_test_trend = test_df.groupby('month').agg(
    price=('price', 'mean'),
    rf_predicted=('rf_predicted_price', 'mean'), rf_mae=('rf_abs_err', 'mean'), rf_acc=('rf_accuracy', 'median'),
    mlp_predicted=('mlp_predicted_price', 'mean'), mlp_mae=('mlp_abs_err', 'mean'), mlp_acc=('mlp_accuracy', 'median'),
    xgb_predicted=('xgb_predicted_price', 'mean'), xgb_mae=('xgb_abs_err', 'mean'), xgb_acc=('xgb_accuracy', 'median'),
    lgbm_predicted=('lgbm_predicted_price', 'mean'), lgbm_mae=('lgbm_abs_err', 'mean'), lgbm_acc=('lgbm_accuracy', 'median')
).reset_index()

# Display terminal readout
print(monthly_test_trend.to_string(index=False))

# B) Build charts - Historical Chart and Forecast Validation Chart (Actual Average Vs Forecasted Price)
# Draw final graphs
plt.figure(figsize=(10, 6))
plt.plot(yearly_test_trend['year'], yearly_test_trend['price'], marker="o", color="black", linewidth=2, label="Actual Avg Price")
plt.plot(yearly_test_trend['year'], yearly_test_trend['rf_predicted'], marker="x", linestyle="--", color="blue", label="Random Forest")
plt.plot(yearly_test_trend['year'], yearly_test_trend['mlp_predicted'], marker="s", linestyle="--", color="red", label="Neural Network")
plt.plot(yearly_test_trend['year'], yearly_test_trend['xgb_predicted'], marker="^", linestyle="--", color="green", label="XGBoost")
plt.plot(yearly_test_trend['year'], yearly_test_trend['lgbm_predicted'], marker="d", linestyle="--", color="purple", label="LightGBM")
plt.title("5-Year Ahead Baseline Forecast Validation")
plt.xlabel("Year")
plt.ylabel("Average Property Price (£)")
plt.legend()
plt.grid(True)
plt.savefig("03_forecast_validation_yearly.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(monthly_test_trend['month'], monthly_test_trend['price'], marker="o", color="black", linewidth=2, label="Actual Avg Price")
plt.plot(monthly_test_trend['month'], monthly_test_trend['rf_predicted'], marker="x", linestyle="--", color="blue", label="Random Forest")
plt.plot(monthly_test_trend['month'], monthly_test_trend['mlp_predicted'], marker="s", linestyle="--", color="red", label="Neural Network")
plt.plot(monthly_test_trend['month'], monthly_test_trend['xgb_predicted'], marker="^", linestyle="--", color="green", label="XGBoost")
plt.plot(monthly_test_trend['month'], monthly_test_trend['lgbm_predicted'], marker="d", linestyle="--", color="purple", label="LightGBM")
plt.title("Monthly Seasonality Baseline Validation (Months 1-12)")
plt.xlabel("Month")
plt.ylabel("Average Property Price (£)")
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True)
plt.savefig("03_forecast_validation_monthly.png")
plt.close()

# ==============================================================================
# --- AUTOMATED BAR CHART GENERATION ---
# ==============================================================================
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="muted")
bar_colors = ['#FF9999', '#66B2FF', '#99FF99', '#DDA0DD']

# MAE Comparison
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=plot_models, y=plot_maes, hue=plot_models, palette=bar_colors, dodge=False)
plt.title('Baseline Model Error (MAE) Comparison\n(Lower Error = Better)', fontsize=14, pad=15)
plt.ylabel('Mean Absolute Error (£)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(plot_maes):
    ax.text(i, v + 2000, f'£{v:,.0f}', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('03_chart_model_mae_comparison.png', dpi=200)
plt.close()

# Speed Comparison
plt.figure(figsize=(10, 6))
ax2 = sns.barplot(x=plot_models, y=plot_times, hue=plot_models, palette=bar_colors, dodge=False)
plt.title('Execution Processing Speed (100k records)\n(Lower Time = Better)', fontsize=14, pad=15)
plt.ylabel('Training Time (Seconds)', fontsize=12)
for i, v in enumerate(plot_times):
    ax2.text(i, v + 0.05, f'{v:.2f}s', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('03_chart_model_speed_comparison.png', dpi=200)
plt.close()

# Accuracy Comparison
plt.figure(figsize=(10, 6))
ax3 = sns.barplot(x=plot_models, y=plot_accuracies, hue=plot_models, palette=bar_colors, dodge=False)
plt.title('Median Validation Accuracy %\n(Higher Accuracy = Better)', fontsize=14, pad=15)
plt.ylabel('Baseline Target Accuracy (%)', fontsize=12)
plt.ylim(50, 100) 
for i, v in enumerate(plot_accuracies):
    ax3.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('03_chart_model_accuracy_comparison.png', dpi=200)
plt.close()

# Exit cleanly informing process completed
print("\nModeling and Analytics complete. Output Line and Bar charts are saved locally.")
