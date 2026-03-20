import pandas as pd
import time
import os

input_file = "pp-complete.csv"
output_file = "london_data.csv"

# The original dataset length without UUID
# Note: we observed there is NO transaction unique identifier in this copy, so 15 columns.
columns = [
    "price", "date_of_transfer", "postcode", "property_type", 
    "old_new", "duration", "paon", "saon", "street", "locality", 
    "town_city", "district", "county", "ppd_category", "record_status"
]

print("Filtering for GREATER LONDON to reduce dataset size...")
start_time = time.time()

chunksize = 1000000
total_london_rows = 0

# If output exists, remove it first
if os.path.exists(output_file):
    os.remove(output_file)

first_chunk = True
for chunk_number, chunk in enumerate(pd.read_csv(input_file, names=columns, header=None, chunksize=chunksize, low_memory=False)):
    # Filter for GREATER LONDON
    london_chunk = chunk[chunk['county'] == 'GREATER LONDON']
    total_london_rows += len(london_chunk)
    
    # Append to output CSV
    if not london_chunk.empty:
        mode = 'w' if first_chunk else 'a'
        header = True if first_chunk else False
        london_chunk.to_csv(output_file, mode=mode, header=header, index=False)
        first_chunk = False
        
    print(f"Processed chunk {chunk_number + 1}. Found {len(london_chunk)} London records.")

end_time = time.time()
print(f"Finished filtering. Total London records: {total_london_rows}")
print(f"Time taken: {end_time - start_time:.2f} seconds.")
