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
plt.savefig("3_historical_trend.png")
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

# Output status
print("Training Random Forest Regressor...")
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

# Define an isolated helper function that prints standard benchmark statistics taking the name/model directly
def evaluate_model(name, model):
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
    
    # Spacing
    print(f"\n--- Model Evaluation: {name} ---")
    # Output the exact £ RMSE formatting visually with commas
    print(f"RMSE: £{rmse:,.2f}")
    # Output the exact £ MAE cleanly
    print(f"MAE: £{mae:,.2f}")
    # Show R2 scoring matrix out of 1.0 peak
    print(f"R-Squared (out-of-sample): {r2:.4f}")
    
    # Function returns the generated physical predictions array
    return y_pred

# Force function evaluations to execute over both loaded pre-trained models
rf_pred = evaluate_model("Random Forest", rf_model)
mlp_pred = evaluate_model("Neural Network", mlp_model)

# Map the exact predictions back into the original dataset frame for a pure unified view
test_df['rf_predicted_price'] = rf_pred
# Compute group level aggregates displaying exact historic averages against averaged historic predictions per year
yearly_test_trend = test_df.groupby('year').agg({'price': 'mean', 'rf_predicted_price': 'mean'}).reset_index()

# Draw one final 10x6 inch graph to demonstrate predictive performance
plt.figure(figsize=(10, 6))
# Create the true physical real data line using an 'O' marker for truth visualization
plt.plot(yearly_test_trend['year'], yearly_test_trend['price'], marker="o", label="Actual Avg Price")
# Create the Random Forest simulated future line using an 'X' marker and dashes mapped against the same timeline
plt.plot(yearly_test_trend['year'], yearly_test_trend['rf_predicted_price'], marker="x", linestyle="--", label="Forecasted Price (RF)")
# Name the graph appropriately
plt.title("5-Year Ahead Holdout Forecast Validation (2018-2022)")
# Label Axis
plt.xlabel("Year")
# Label Axis
plt.ylabel("Average Property Price (£)")
# Ensure the legend differentiates truth lines vs artificial lines clearly
plt.legend()
# Turn down grids to show values sharply
plt.grid(True)
# Directly record chart visualization straight to disk for users
plt.savefig("3_forecast_validation.png")
# Tidy RAM
plt.close()

# Exit cleanly informing process completed
print("\nModeling and Analysis complete. Output charts are saved to artifacts.")
