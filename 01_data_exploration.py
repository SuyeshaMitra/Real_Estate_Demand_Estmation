# Import the pandas library for data manipulation and analysis
import pandas as pd
# Import the numpy library for numerical operations
import numpy as np

# Define the path to the main CSV dataset containing property data
file_path = "pp-complete.csv"

# Define a list of column names representing the structure of the HM Land Registry Price Paid Data
columns = [
    "transaction_id", "price", "date_of_transfer", "postcode", "property_type", 
    "old_new", "duration", "paon", "saon", "street", "locality", 
    "town_city", "district", "county", "ppd_category", "record_status"
]

# Print a header for the console output indicating the first part of data exploration
print("--- Data Exploration: First 15 Rows ---")
# Read only the first 15 rows of the CSV file using pandas, assigning the predefined column names
df_head = pd.read_csv(file_path, names=columns, header=None, nrows=15)
# Print the dataframe containing these first 15 rows to the console
print(df_head)

# Print a header indicating that data types will be displayed next
print("--- Data Types ---")
# Print the pandas-inferred data types for each column in the first 15 rows
print(df_head.dtypes)

# Print a header indicating the start of a deeper data quality assessment
print("\n--- Identifying Unique Values and Data Quality Gaps ---")
# Create a dictionary of sets to track unique values for categorical features. We ignore identifiers and highly unique text columns to save memory.
unique_counts = {col: set() for col in columns if col not in ['transaction_id', 'price', 'date_of_transfer', 'postcode', 'paon', 'saon', 'street', 'locality']}
# Initialize a counter for the total number of rows processed
total_rows = 0
# Initialize a dictionary to count the number of missing (null) values in each column
missing_values = {col: 0 for col in columns}

# Define the number of rows to process in each chunk to avoid memory overflow limits
chunksize = 1000000
# Iterate through the massive CSV file in chunks
for chunk in pd.read_csv(file_path, names=columns, header=None, chunksize=chunksize, low_memory=False):
    # Add the number of rows in the current chunk to our running total
    total_rows += len(chunk)
    # Loop through each column name
    for col in columns:
        # Sum up how many missing (null) values exist in this column for the current chunk and add it to our tracking dictionary
        missing_values[col] += chunk[col].isnull().sum()
    
    # Loop through the categorical columns we are tracking unique values for
    for col in unique_counts.keys():
        # Drop nulls, find unique values in the current chunk, convert them to a list, and update our set of unique values
        unique_counts[col].update(chunk[col].dropna().unique().tolist())

# After reading all chunks, print the total number of rows analyzed
print(f"Total Rows Analyzed: {total_rows}")

# Print a header for the missing values report
print("\n--- Missing Values ---")
# Iterate over the calculated missing values dictionary
for col, count in missing_values.items():
    # Print the column name, total missing count, and the percentage of rows missing this data
    print(f"{col}: {count} ({count/total_rows*100:.2f}%)")

# Print a header for the unique categorical values report
print("\n--- Unique Values for Categorical Columns ---")
# Iterate over our dictionary of unique values
for col, unq in unique_counts.items():
    # If a categorical column has less than 50 unique values (e.g. property_type), print them all out
    if len(unq) < 50:
        print(f"{col} ({len(unq)} unique): {unq}")
    # If a column has 50 or more unique values (to avoid console spam), just show the total count
    else:
        print(f"{col} ({len(unq)} unique): [Too many to list individually]")
