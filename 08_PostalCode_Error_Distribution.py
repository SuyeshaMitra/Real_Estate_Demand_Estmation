import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Loading Postal Code errors...")
try:
    with open('08_postcode_errors.pkl', 'rb') as f:
        postcode_errors = pickle.load(f)
except FileNotFoundError:
    print("Error: Run 08_Master_Combinatorial_Runner.py first.")
    exit()

# We will plot the distribution of the "All Combined" model (08A) or "OSM + News" (08S)
# Let's just pick the absolute best one or a representative one, e.g., '08E_LatLon_OSM_News' 
# Actually, the user asked for "Average Error and Accuracy". 
# Let's dynamically read it and calculate it.

# Load the test dataset specifically to recalculate accuracy per postcode
print("Loading holdout dataset (2018-2022) to reconstruct prices for accuracy calculation...")
df = pd.read_csv('london_geospatial_enriched_dataset.csv')
test_df = df[(df['year'] >= 2018) & (df['year'] <= 2022)]

# Using the errors from the best model, let's say 08E_LatLon_OSM_News
target_combo = "08E_LatLon_OSM_News"
if target_combo not in postcode_errors:
    target_combo = list(postcode_errors.keys())[0]

print(f"Plotting Postcode Distribution for Combination: {target_combo}")
errors_dict = postcode_errors[target_combo]

# Create a dataframe for visualization
pc_df = pd.DataFrame(list(errors_dict.items()), columns=['postcode', 'avg_absolute_error'])

# Calculate the average true price per postcode to calculate average accuracy
avg_prices = test_df.groupby('postcode')['price'].mean().reset_index()
avg_prices.rename(columns={'price': 'avg_true_price'}, inplace=True)

pc_df = pd.merge(pc_df, avg_prices, on='postcode')

# Calculate median accuracy per postcode
# Accuracy = Max(0, 100 - (Error / Price) * 100)
pc_df['avg_accuracy'] = np.maximum(0, 100 - (pc_df['avg_absolute_error'] / pc_df['avg_true_price']) * 100)

print("Generating 08_PostalCode_Error_Distribution.png...")
plt.figure(figsize=(16, 8))

# Subplot 1: Distribution of Absolute Error
plt.subplot(1, 2, 1)
sns.histplot(pc_df['avg_absolute_error'], bins=100, kde=True, color='red')
plt.title(f'Distribution of Average Absolute Error\nAcross {len(pc_df):,} London Postal Codes', fontsize=14, fontweight='bold')
plt.xlabel('Average Absolute Error (£)', fontsize=12)
plt.ylabel('Number of Postal Codes', fontsize=12)
plt.xlim(0, 2000000) # Limit to £2M for readability

# Subplot 2: Distribution of Accuracy
plt.subplot(1, 2, 2)
sns.histplot(pc_df['avg_accuracy'], bins=50, kde=True, color='green')
plt.title(f'Distribution of Prediction Accuracy\nAcross {len(pc_df):,} London Postal Codes', fontsize=14, fontweight='bold')
plt.xlabel('Average Accuracy (%)', fontsize=12)
plt.ylabel('Number of Postal Codes', fontsize=12)
plt.xlim(0, 100)

plt.tight_layout()
plt.savefig('08_PostalCode_Error_Distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Let's also do a scatter plot of Accuracy vs True Price to see where the model fails
plt.figure(figsize=(10, 6))
sns.scatterplot(x='avg_true_price', y='avg_accuracy', data=pc_df, alpha=0.3, color='purple')
plt.title('Prediction Accuracy vs. Neighborhood Wealth (Postal Code Average)', fontsize=14, fontweight='bold')
plt.xlabel('Average True House Price in Postal Code (£)', fontsize=12)
plt.ylabel('Average Prediction Accuracy (%)', fontsize=12)
plt.xscale('log')
plt.grid(True, which="both", ls="--", alpha=0.2)
plt.savefig('08_PostalCode_Wealth_vs_Accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

print("Charts successfully exported to root directory!")
