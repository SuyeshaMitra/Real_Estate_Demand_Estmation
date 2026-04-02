# Import pandas for optimal data manipulation and analysis
import pandas as pd
# Import time to measure how long script execution takes
import time
# Import os module for interacting with the file system (like deleting files)
import os

# Define the source input file containing the massive raw dataset
input_file = "pp-complete.csv"
# Define the destination output file to store only the filtered records
output_file = "london_data.csv"

# Pre-define the 15 columns that exist in our version of the CSV (no UUID included in this copy)
columns = [
    "price", "date_of_transfer", "postcode", "property_type", 
    "old_new", "duration", "paon", "saon", "street", "locality", 
    "town_city", "district", "county", "ppd_category", "record_status"
]

# Indicate to the console that the filtering process is starting
print("Filtering for GREATER LONDON to reduce dataset size...")
# Record the current exact time so we can calculate total duration later
start_time = time.time()

# Define chunk size to process 1,000,000 rows at a time to prevent memory overflow limits
chunksize = 1000000
# Initialize a counter to track the absolute sum of matching GREATER LONDON records
total_london_rows = 0

# Check if an older version of the output file exists already
if os.path.exists(output_file):
    # If it does, delete it so we start fresh and don't append to old data
    os.remove(output_file)

# Flag to track whether we are processing the very first chunk of data
first_chunk = True
# Loop directly through the massive CSV reading it continuously chunk-by-chunk using Pandas
for chunk_number, chunk in enumerate(pd.read_csv(input_file, names=columns, header=None, chunksize=chunksize, low_memory=False)):
    # Filter the current chunk of data, keeping only rows where the 'county' column equals 'GREATER LONDON'
    london_chunk = chunk[chunk['county'] == 'GREATER LONDON']
    # Add the count of matched london records in this iteration to our grand total
    total_london_rows += len(london_chunk)
    
   # print("--- london_chunk ---")
   # print(london_chunk)


   # print("--- total_london_rows ---")
   # print(total_london_rows)


    # If our filtered dataframe chunk has at least 1 record inside it...
    if not london_chunk.empty:
        # Determine whether to write ('w') over the file initially, or append ('a') to it subsequently
        mode = 'w' if first_chunk else 'a'
        # Output the column header row only if this is our very first run
        header = True if first_chunk else False
        # Save this specific chunk's Greater London data onto the local SSD as a CSV
        london_chunk.to_csv(output_file, mode=mode, header=header, index=False)
        # Flip the flag so future loops append to the file instead of overwriting it
        first_chunk = False
        
    # Inform the user via console about the completion of the current chunk
    print(f"Processed chunk {chunk_number + 1}. Found {len(london_chunk)} London records.")

# Record the exact finish time of the file operations
end_time = time.time()
# Output the final computed grand total of matching records to the console
print(f"Finished filtering. Total London records: {total_london_rows}")
# Display the exact elapsed processing time calculated in readable seconds
print(f"Time taken: {end_time - start_time:.2f} seconds.")
