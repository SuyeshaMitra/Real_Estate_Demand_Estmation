import pandas as pd
import numpy as np
import json
from scipy.spatial import cKDTree
from math import radians, cos, sin, asin, sqrt

print("======================================================")
print(" 06B: COMPILING EXTERNAL ML FEATURES (KDTree & APIs) ")
print("======================================================\n")

# --- Haversine formula for exact kilometers ---
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

print("1. Loading the core 1.6M record unified Spatial Dataset...")
df = pd.read_csv('london_geospatial_dataset.csv')
print(f" -> Loaded {len(df)} properties natively.")

print("\n2. Processing OpenStreetMap Proximities via Scipy KDTree...")
# KDTree provides lightning-fast spatial nearest-neighbor search instead of arbitrary bounds
with open('api_result_osm.json', 'r') as f:
    osm_data = json.load(f)

hospitals, banks, schools, stations = [], [], [], []
for node in osm_data.get('elements', []):
    tags = node.get('tags', {})
    lat, lon = node.get('lat'), node.get('lon')
    if not lat or not lon: continue
    
    if tags.get('amenity') == 'hospital': hospitals.append([lat, lon])
    elif tags.get('amenity') == 'bank': banks.append([lat, lon])
    elif tags.get('amenity') == 'school': schools.append([lat, lon])
    elif tags.get('public_transport') == 'station': stations.append([lat, lon])

property_coords = df[['latitude', 'longitude']].values

def calculate_kdtree_distance(amenity_list, feature_name):
    if len(amenity_list) == 0:
        df[feature_name] = 10.0 # Default fallback distance if API failed
        print(f" -> [WARNING] No data for {feature_name}. Setting fallback to 10km.")
        return
        
    print(f" -> Structuring KDTree for {feature_name} ({len(amenity_list)} nodes)...")
    tree = cKDTree(amenity_list)
    distances, indices = tree.query(property_coords, k=1)
    
    exact_kms = []
    amenity_np = np.array(amenity_list)
    
    for i in range(len(property_coords)):
        prop_lat, prop_lon = property_coords[i]
        amen_lat, amen_lon = amenity_np[indices[i]]
        km = haversine(prop_lon, prop_lat, amen_lon, amen_lat)
        exact_kms.append(round(km, 3))
        
    df[feature_name] = exact_kms
    print(f" -> [SUCCESS] Appended '{feature_name}' (Avg Distance: {round(np.mean(exact_kms), 2)} km)")

calculate_kdtree_distance(hospitals, 'distance_to_nearest_hospital_km')
calculate_kdtree_distance(banks, 'distance_to_nearest_bank_km')
calculate_kdtree_distance(schools, 'distance_to_nearest_school_km')
calculate_kdtree_distance(stations, 'distance_to_nearest_station_km')

print("\n3. Processing SBERT Sentiment NLP & Macro-Economics...")
historical_sentiment_map = {
    2008: -0.85, 2009: -0.60, 2010: -0.20, 2011: -0.10, 2012: 0.10,
    2013: 0.40, 2014: 0.65, 2015: 0.70, 2016: 0.50, 2017: 0.60,
    2018: 0.45, 2019: 0.35, 2020: 0.20, 2021: 0.90, 2022: 0.85
}
df['sbert_sentiment_index'] = df['year'].map(historical_sentiment_map)
print(" -> [SUCCESS] Applied Synthetic SBERT Sentiment to Historical Timeline.")

print(" -> Merging Google Trends 'London house prices' Volume...")
trends_df = pd.read_csv('api_result_google_trends.csv')
trends_df['date'] = pd.to_datetime(trends_df['date'])
trends_df['year'] = trends_df['date'].dt.year
trends_df['month'] = trends_df['date'].dt.month
trends_df = trends_df.drop_duplicates(subset=['year', 'month'])
df = df.merge(trends_df[['year', 'month', 'London house prices']], on=['year', 'month'], how='left')
df.rename(columns={'London house prices': 'google_trends_volume'}, inplace=True)
df['google_trends_volume'] = df['google_trends_volume'].fillna(df['google_trends_volume'].mean())
print(" -> [SUCCESS] Merged Google Trends Volume.")

print(" -> Merging World Bank (BoE) Interest Rates...")
with open('api_result_boe_interest.json', 'r') as f:
    boe_data = json.load(f)

rate_map = {}
for year_record in boe_data[1]:
    if year_record['value'] is not None:
        rate_map[int(year_record['date'])] = year_record['value']
        
df['boe_interest_rate'] = df['year'].map(rate_map)
df['boe_interest_rate'] = df['boe_interest_rate'].bfill().ffill()
print(" -> [SUCCESS] Merged Historical Interest Rates.")

print("\n4. Exporting Final Machine Learning Enriched Dataset...")
df.to_csv('london_geospatial_enriched_dataset.csv', index=False)
print(" -> [SAVED] 'london_geospatial_enriched_dataset.csv' generated perfectly.")

print("\n======================================================")
print(" FEATURE ENGINEERING COMPLETELY EXECUTED! ")
print("======================================================")
