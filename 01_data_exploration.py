import pandas as pd
import numpy as np

file_path = "pp-complete.csv"

# Columns as per HM Land Registry Price Paid Data
columns = [
    "transaction_id", "price", "date_of_transfer", "postcode", "property_type", 
    "old_new", "duration", "paon", "saon", "street", "locality", 
    "town_city", "district", "county", "ppd_category", "record_status"
]

print("--- Data Exploration: First 15 Rows ---")
# Read only first 15 rows
df_head = pd.read_csv(file_path, names=columns, header=None, nrows=15)
print(df_head)

print("\--- Data Types ---")
print(df_head.dtypes)

print("\n--- Identifying Unique Values and Data Quality Gaps ---")
# To avoid MemoryError on a 3.2GB file, we will read in chunks and aggregate
unique_counts = {col: set() for col in columns if col not in ['transaction_id', 'price', 'date_of_transfer', 'postcode', 'paon', 'saon', 'street', 'locality']}
total_rows = 0
missing_values = {col: 0 for col in columns}

chunksize = 1000000
for chunk in pd.read_csv(file_path, names=columns, header=None, chunksize=chunksize, low_memory=False):
    total_rows += len(chunk)
    for col in columns:
        missing_values[col] += chunk[col].isnull().sum()
    
    # Collect unique values for categorical columns
    for col in unique_counts.keys():
        unique_counts[col].update(chunk[col].dropna().unique().tolist())

print(f"Total Rows Analyzed: {total_rows}")

print("\n--- Missing Values ---")
for col, count in missing_values.items():
    print(f"{col}: {count} ({count/total_rows*100:.2f}%)")

print("\n--- Unique Values for Categorical Columns ---")
for col, unq in unique_counts.items():
    if len(unq) < 50:
        print(f"{col} ({len(unq)} unique): {unq}")
    else:
        print(f"{col} ({len(unq)} unique): [Too many to list individually]")
