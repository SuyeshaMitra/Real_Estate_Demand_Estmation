import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("Loading 04 Spatial Prediction Validation files...")
# Define the files mapped to the 04 pipeline
files = {
    'Geospatial RF': 'prediction_validation_randomforest.csv',
    'Geospatial XGBoost': 'prediction_validation_xgb.csv',
    'Geospatial LightGBM': 'prediction_validation_lightgbm.csv'
}

# Dictionary to hold the grouped yearly trend aggregations
trends = {}
actual_trend = None

for model_name, filename in files.items():
    print(f"Parsing {filename}...")
    try:
        df = pd.read_csv(filename)
        # Convert date to year explicitly
        df['year'] = pd.to_datetime(df['date_of_transfer']).dt.year
        
        # Calculate yearly average for actual price (only need to do it once)
        if actual_trend is None:
            actual_trend = df.groupby('year')['Actual_Price'].mean()
            
        # Calculate yearly average for predicted price
        trends[model_name] = df.groupby('year')['Predicted_Price'].mean()
    except Exception as e:
        print(f"Warning: Could not process {filename}. Error: {e}")

if actual_trend is not None and trends:
    print("Generating 04_combined_spatial_forecast_validation.png...")
    plt.figure(figsize=(10, 6))
    
    # Plot true actual price baseline
    plt.plot(actual_trend.index, actual_trend.values, marker="8", markersize=8, linewidth=3, color='black', label="Actual True Avg Price")
    
    # Plot model prediction trends with distinct colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    for i, (model_name, pred_trend) in enumerate(trends.items()):
        plt.plot(pred_trend.index, pred_trend.values, marker="x", linestyle="--", linewidth=2, color=colors[i], label=f"Predicted: {model_name}")
        
    plt.title("5-Year Spatial Forecast Validation (04 Baseline Sweep)")
    plt.xlabel("Holdout Test Year")
    plt.ylabel("Average Property Price (£)")
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Save the output chart
    output_filename = "04_combined_spatial_forecast_validation.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Successfully generated {output_filename} comparing 3 spatial models simultaneously!")
else:
    print("Failed to generate chart. Ensure the CSV tracking files exist.")
