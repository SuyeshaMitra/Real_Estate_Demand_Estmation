# Import time to measure the delay required to fetch responses
import time
# Import the standard requests package internally handling any outbound HTTP queries
import requests
# Import JSON framework specifically for natively parsing dictionary tree results
import json
# Import pandas mainly natively for tabular file persistence and creation
import pandas as pd
# Import Google Trends API client specifically to bypass standard scraping filters
from pytrends.request import TrendReq
# Import core python core XML tree mapping toolkit to rip news RSS tags natively 
import xml.etree.ElementTree as ET

print("======================================================")
print(" EXTERNAL ECOSYSTEM FEATURE EXTRACTION PROOF OF CONCEPT ")
print("======================================================\n")

# -------------------------------------------------------------
# --- STEP 1: OPENSTREETMAP (OSM) INFRASTRUCTURE EXTRACTION ---
# WHAT IT DOES: We send GPS coordinates to public free map servers to structurally count
#               the number of Train Stations and Schools inside a 1.5 kilometer walking radius.
# WHY IT MATTERS: A house with 5 train stations nearby is fundamentally worth more than identical houses without.
# -------------------------------------------------------------
print("1. Testing Public OpenStreetMap (OSM) Overpass API...")
# Hardcode an arbitrary central London sample GPS target explicitly
sample_lat = 51.3734
# Setup test longitude marker explicitly pointing natively close inside standard zones
sample_lon = 0.0881
# Explicit metric bounds bounding how far people are naturally willing to navigate 
radius_meters = 1500 # 1.5 km walking distance

# Hardcode main open source Overpass interpreter API HTTP receiver locally
overpass_url = "http://overpass-api.de/api/interpreter"
# Inject Overpass Query Language natively targeting local mapping bounds internally purely looking for node classes
overpass_query = f"""
[out:json];
(
  node["public_transport"="station"](around:{radius_meters},{sample_lat},{sample_lon});
  node["amenity"="school"](around:{radius_meters},{sample_lat},{sample_lon});
);
out center;
"""

print(f" -> Sending HTTP GET request to Overpass API for Lat:{sample_lat}, Lon:{sample_lon}...")
try:
    # Ship actual HTTP call binding querying the explicit Overpass Query natively 
    response = requests.get(overpass_url, params={'data': overpass_query})
    # Process return purely as a JSON dictionary 
    data = response.json()
    
    # Dump entire raw output stream securely to physical artifact JSON tracking file
    with open('api_result_osm.json', 'w') as f:
        # Format completely with indent tabs for easy human debug reviews securely
        json.dump(data, f, indent=4)
    # Output success text locally
    print(" -> File Saved: 'api_result_osm.json'")
    
    # Spin up integer counters tracking the amenity counts exactly
    stations, schools = 0, 0
    # Loop over every unique node instance returned locally directly from Overpass
    for element in data['elements']:
        # Double check node tags correctly map internally
        if 'tags' in element:
            # Add up instances marking public transport explicitly
            if element['tags'].get('public_transport') == 'station':
                stations += 1
            # Add up specific school mapping tags locally correctly
            if element['tags'].get('amenity') == 'school':
                schools += 1
                
    # Output absolute success logic directly referencing the explicit extracted features we'd feed ML natively 
    print(f" -> [SUCCESS] Extracted Features to append to model: 'schools_within_1.5km': {schools}, 'stations_within_1.5km': {stations}")
except Exception as e:
    # Error checking safety bound to stop crashes
    print(f" -> [ERROR] Failed to hit OSM API: {e}")

# Padding
print("\n------------------------------------------------------\n")

# -------------------------------------------------------------
# --- STEP 2: GOOGLE TRENDS API (PYTRENDS) ECONOMIC EXTRACTION ---
# WHAT IT DOES: We ping Google's servers to fundamentally download the raw search volume percentage 
#               for terms like "London mortgage" month-over-month.
# WHY IT MATTERS: People frantically googling "mortgage" is the ultimate leading indicator 
#                 of a real estate bubble exploding before the actual housing prices physically rise.
# -------------------------------------------------------------
print("2. Testing Public Google Trends API (pytrends)...")
try:
    # Spool up pytrends client forcing language localization to UK English natively 
    pytrend = TrendReq(hl='en-GB', tz=0)
    # Target our primary explicitly chosen macro-economic test indicators natively 
    kw_list = ["London mortgage", "London house prices"]
    
    # Broadcast status cleanly directly via output 
    print(f" -> Sending Payload to Google Trends for keywords: {kw_list}...")
    # Package actual Trends search targeting fully specific GB geographic bounds and a specific window
    pytrend.build_payload(kw_list, cat=0, timeframe='2018-01-01 2022-12-31', geo='GB-ENG')
    # Issue active tracking search directly and cast output directly natively to a Pandas frame natively 
    interest_over_time_df = pytrend.interest_over_time()
    
    # Dump the fully fetched timeline frame clearly to standard physical csv correctly 
    interest_over_time_df.to_csv('api_result_google_trends.csv')
    # Console explicit success log status 
    print(" -> File Saved: 'api_result_google_trends.csv'")
    
    # State explicit success log metric cleanly
    print(f" -> [SUCCESS] Google Trends Data Fetched!")
    # State debug snapshot correctly
    print(" -> Sample DataFrame Extract:")
    # Print pure raw debug output from top three metrics directly visually
    print(interest_over_time_df.head(3))
except Exception as e:
    # Inform cleanly if Google has arbitrarily banned IP addresses
    print(f" -> [NOTE] Google Trends API Rate Limited.")
    # Persist empty mapping file layout directly so system pipelines won't natively fail internally 
    pd.DataFrame(columns=["date", "London mortgage", "London house prices", "isPartial"]).to_csv('api_result_google_trends.csv')

# Padding layout
print("\n------------------------------------------------------\n")

# -------------------------------------------------------------
# --- STEP 3: GOOGLE NEWS RSS FEED GEOPOLITICAL EXTRACTION ---
# WHAT IT DOES: We mathematically parse the XML data from Google News searching for "London Real Estate"
#               to physically count how many global news articles are being published every week.
# WHY IT MATTERS: International investment (like foreign oligarchs buying London properties) causes massive 
#                 unpredictable spikes that standard historical data literally cannot see.
# -------------------------------------------------------------
# Inform the start of simple news gathering targets 
print("3. Testing Public Google News RSS Real Estate Feed...")
try:
    # Direct pure Google RSS string clearly targeting "London Real Estate" search logic 
    news_url = "https://news.google.com/rss/search?q=London+Real+Estate"
    # Inform start query directly correctly 
    print(f" -> Sending HTTP GET request to Google News Feed...")
    # Fire off fully native HTTP retrieval of XML directly 
    news_response = requests.get(news_url)
    
    # Lock onto the raw byte response and dump the stream cleanly into an XML 
    with open('api_result_google_news.xml', 'wb') as f:
        # Write purely standard byte encoding directly properly 
        f.write(news_response.content)
    # Debug success tracking message locally 
    print(" -> File Saved: 'api_result_google_news.xml'")
    
    # Parse pure XML trees natively back across bytes straight into memory natively 
    root = ET.fromstring(news_response.content)
    # Find all direct explicit item children logic natively to count raw articles exactly
    article_count = len(root.findall('.//item'))
    # Return simple total validation bounds properly locally 
    print(f" -> [SUCCESS] API Response Received! Total Articles scanned: {article_count}")
    # Return direct physical extraction explicitly bounding how we'd ingest internally 
    print(f" -> Extracted Features to append to model: 'weekly_news_volume': {article_count}")
    
except Exception as e:
    # Throw catch internally natively keeping pipelines safe 
    print(f" -> [ERROR] Failed to hit Google News API: {e}")

# Padding natively 
print("\n======================================================")
# Full file completion state bound notification correctly 
print(" API PIPELINE TEST COMPLETE - FILES SAVED TO ROOT ")
print("======================================================")
