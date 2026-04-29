import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Loading test data...")
df = pd.read_csv('london_geospatial_enriched_dataset.csv')
test_df = df[(df['year'] >= 2018) & (df['year'] <= 2022)]
actual_yearly = test_df.groupby('year')['price'].mean().reset_index()
actual_monthly = test_df.groupby('month')['price'].mean().reset_index()

# We need the 30 combination predictions to plot them.
# BUT wait! My 08_Master_Combinatorial_Runner.py did NOT export the row-level predictions (to save 50GB memory).
# I must extract the top model logic or do something else. 
# Wait, let's just re-run the top 5 models quickly to generate the line plots!
# I will just write a script that does this.
