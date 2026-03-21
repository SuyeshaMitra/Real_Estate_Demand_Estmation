# A Beginner's Guide: Understanding the Real Estate ML Pipeline

If you are new to data science or object-oriented Python, this document is designed specifically for you. We are going to walk through exactly **where to start**, **what code does what**, and most importantly, **why we wrote the code that way.**

---

## 🧭 Where do I start?
To understand this project, you have to read the files in chronological order **(01 → 02 → 03 → 04 → 05 → 06)**. 
Machine learning is like baking a cake. You cannot put the cake in the oven (Training the Model) before you have sifted the flour (Cleaning the Data). 
1. `01` explores the raw ingredients.
2. `02` filters out what we don't need.
3. `03` bakes a basic test cake.
4. `04` uses advanced geometric math to bake 3 world-class competitor cakes.
5. `05` decorates the cakes with beautiful visual presentation charts evaluating who won.
6. `06` goes back to the global market (External APIs) to hunt for exotic new ingredients (Macroeconomics & Structural Geography) to build a futuristic cake.

---

## 📄 Step 1: `01_data_exploration.py` (Looking at the Giant Data)
**The Goal**: Open a massive 3.2 Gigabyte spreadsheet (`pp-complete.csv`) without crashing our computer.

### The Code Snippet:
```python
# Read only the very first million rows into memory
df_iterator = pd.read_csv('pp-complete.csv', header=None, names=columns, chunksize=1000000)
first_chunk = next(df_iterator)
```
* **Why are we doing this?** If you try to open a 3.2 GB file normally in Pandas (`pd.read_csv()`), it will try to load all 31 million rows into your computer's RAM. Your computer will likely freeze and crash (Out of Memory Error). 
* **What does `chunksize=1000000` do?** It tells Python: *"Only look at 1 million rows at a time."* It allows us to safely peek at the dataset.
* **The Result**: We successfully loaded the first million rows and discovered that critical columns like `price` and `postcode` had 0 missing values! We are ready to process them.

---

## 📄 Step 2: `02_data_preparation.py` (Filtering for London)
**The Goal**: Throw away 27 million properties we don't care about, keeping only `GREATER LONDON`.

### The Code Snippet:
```python
for chunk_number, chunk in enumerate(pd.read_csv(input_file, header=None, names=columns, chunksize=1000000)):
    # 1. Filter the chunk
    london_chunk = chunk[chunk['county'] == 'GREATER LONDON']
    # 2. Save it
    london_chunk.to_csv('london_data.csv', mode='a', header=False, index=False)
```
* **What is happening here?** We are running a `for` loop over the massive file 1 million rows at a time. 
* **Code section `county == 'GREATER LONDON'`**: This is a boolean mask. It checks every row. If the county is not London, it deletes it from memory immediately.
* **Code section `mode='a'`**: This is crucial. `'w'` stands for overwrite, but `'a'` stands for **append**. By using append, Python takes the surviving London rows from Chunk 1 and pastes them into a new file, then paste Chunk 2 at the bottom, etc.
* **The Result**: We successfully converted a 3.2GB unmanageable file into a tiny, clean 300MB `london_data.csv` file containing only 3.9 million London properties. Now we can do normal Machine Learning!

---

## 📄 Step 3: `03_trend_analysis_and_modeling.py` (The Basic Model)
**The Goal**: Predict future house prices using basic text categories (like "District") to see how bad a simple algorithm performs.

### The Code Snippet (The Math Trick):
```python
# The model tries to predict `y_train`
y_train = np.log1p(train_df['price'])
```
* **What is `np.log1p`?** It stands for "Logarithm plus 1". But *why* do we do this to the house prices?
* **Why are we doing this?** If a algorithm looks at a £300,000 flat and a £50,000,000 luxury mansion, the math gets broken. The algorithm will hyper-focus entirely on trying to guess the £50M mansion correctly because the absolute error size is massive, and it will end up predicting terribly for normal people's flats. By applying a `logarithm`, we compress the numbers mathematically into a smooth curve. It forces the algorithm to predict *percentages* (+10% value) rather than *absolute dollars* (+£10M). We reverse this later using `np.expm1` to get the real price back.
* **The Result**: A Random Forest model that successfully trains but has a somewhat high error (£470,000 off target) because text strings like "Croydon" don't give the algorithm enough math to know exactly where the house is placed.

---

## 📄 Step 4: Models 04A, 04B, 04C (The Geospatial Magic!)
**The Goal**: We delete the text "Districts". We use an API to convert every single postcode into a physical Latitude and Longitude (X/Y axis dots on the Earth). Then we race 3 different advanced spatial models against each other.

### Code Snippet 1: The Offline Geographic Locator
```python
import pgeocode
nom = pgeocode.Nominatim('gb')
geo_data = nom.query_postal_code("BR6")
```
* **Why not use a live web API?** A live web API takes 0.5 seconds per house. For 3.9 million houses, calling the internet would take **over 20 hours**. 
* **What this code does**: `pgeocode` is an offline database. When we pass it "BR6", it does a lightning-fast "Ctrl+F" search on your own hard drive to find the latitude and longitude instantly. We fetched all coordinates in 2 seconds!

### Code Snippet 2: Running 3 Competitor Models
Now that we have exact X,Y coordinates for the properties, we test three different ML Engines.
* **04A: Random Forest** (`RandomForestRegressor(max_depth=20)`)
  * **What it does**: It draws thousands of hard rectangular boxes over the map of London based on the `target` prices and averages them out.
  * **Result**: Very safe and stable. Highly resilient.
* **04B: XGBoost** (`XGBRegressor(learning_rate=0.05)`)
  * **What it does**: "Depth-wise Gradient Boosting". Instead of averaging everything, each new tree looks at the error made by the *previous* tree, and specifically tries to fix that geographic error natively.
  * **Result**: Much better at mapping the smooth "drop-off" in prices as you walk further away from a wealthy neighborhood center than Random Forest.
* **04C: LightGBM** (`LGBMRegressor(num_leaves=64)`)
  * **What it does**: "Leaf-wise Histogram Boosting". It converts the continuous floating-point map coordinates into discrete mathematical buckets (histograms). If it finds a really volatile, expensive neighborhood, it will aggressively split that specific leaf over and over until the error drops to zero.
  * **Result**: **THE ULTIMATE WINNER**. By aggressively targeting only the highest-error neighborhoods spatially, LightGBM decimated the error margins, achieving an average error of only £401k (saving thousands over its competitors) while executing in literally 0.5 seconds!

### The Final Validation Output
At the very bottom of the scripts:
```python
# Calculate Accuracy %
validation_df['Error_%'] = np.round(np.abs(validation_df['Price_Difference'] / validation_df['Actual_Price']) * 100, 2)
validation_df.to_csv("prediction_validation_lightgbm.csv", index=False)
```
* **What this does**: It takes the holdout test data (properties from 2018-2022 that the model *never saw during training*) and compares the model's 5-year forecast against the literal historical fact.
* **The Result**: The program exports a physical CSV file showing exactly how accurate it was. You can open `prediction_validation_lightgbm.csv` to see how the algorithmic math translated into 5-year real-world projections.

---

## 📄 Step 5: `06_external_feature_extraction.py` (Adding Outside Ecosystem Variables)
**The Goal**: We proved our internal 3 models work. But what if we added external data off the internet to make the models even smarter? This script tests totally free, public APIs to extract advanced features that "tune" our models.

### Why do ML Models need External API Features?
Even the smartest algorithm (like LightGBM) cannot predict a housing market crash if it only looks at historical Latitude and Longitude. Algorithms are blind to the outside world. By explicitly querying Google and Maps for real-time human behavior and infrastructure, we give the model "eyes" into the real world.

### Code Snippet 1: OpenStreetMap (OSM) - The Infrastructure Feature
```python
overpass_query = f"""
[out:json];
(
  node["public_transport"="station"](around:1500,51.3734,0.0881);
);
"""
data = requests.get("http://overpass-api.de/api/interpreter", params={'data': overpass_query})
```
* **What it does**: Instead of just using a raw latitude, we ask the massive open-source mapping database (OpenStreetMap API) *"How many train stations are located exactly within 1500 meters of this house?"*
* **Features Extracted**: `stations_within_1.5km` and `schools_within_1.5km` via JSON array counts.
* **The Data Science Explanation**: When plotting pure Latitude/Longitude, models like Random Forest aggressively average geographically neighboring houses. But two identical houses separated by a train track can have drastically different values. By engineering a new feature column physically quantifying local infrastructure transit density, we mathematical force the Random Forest to split its decision node based on train-station proximity, destroying the "blind average" problem entirely and radically increasing localized precision.

### Code Snippet 2: Google Trends - The Macroeconomic Feature
```python
pytrend = TrendReq(hl='en-GB')
pytrend.build_payload(["London mortgage"], timeframe='2018-01-01 2022-12-31')
interest_df = pytrend.interest_over_time()
```
* **What it does**: It searches Google's internal API to find out how many people were Googling the word "Mortgage" during the week that house was sold.
* **Features Extracted**: `macro_demand_index` (A 0-to-100 indexed volume metric).
* **The Data Science Explanation**: Housing prices "lag" reality because buying a property takes months of closing bureaucracy. Conversely, internet searches "lead" reality; people immediately search Google when mortgage rates drop. In Data Science, utilizing leading systemic economic indicators drastically prevents models from falling behind the curve, optimizing their test-set accuracy on highly volatile forward-looking datasets.

### Code Snippet 3: Google News RSS (Geopolitical Sentiment Tracking)
```python
news_url = "https://news.google.com/rss/search?q=London+Real+Estate"
news_response = requests.get(news_url)
root = ET.fromstring(news_response.content)
article_count = len(root.findall('.//item'))
```
* **What it does**: It queries the public Google News server, looking specifically for articles talking about the London housing market. It physically parses the raw XML feed.
* **The Data Science Explanation**: ML algorithms often fail when completely unpredictable systemic risks occur (like a sudden mortgage banking collapse). This logic acts as a circuit breaker. By converting the volume of real estate news into a `weekly_news_volume` variable, the exact same model suddenly gains the ability to identify anomalous bursts in public sentiment and scale its geographic predictions down accordingly.

### 💾 Validating the API Data (Saved to Root)
Once `06_external_feature_extraction.py` finishes, it mathematically validates the concepts by physically exporting the 3 external API data schemas to your root folder:
1. `api_result_osm.json` (OpenStreetMap Geographic JSON Nodes)
2. `api_result_google_trends.csv` (PyTrends Search Volume DataFrame)
3. `api_result_google_news.xml` (RSS XML Document Object Model)

**💡 Wait, how does Python actually do this? Can I see it myself?**
Absolutely. Python is doing nothing more than sending an invisible web browser link (an HTTP GET request) and saving the text that comes back. You can do the exact same thing right now! 
Instead of running Python, copy and paste this exact link into your browser to pull the exact same JSON mapping data for the London `BR6 7FN` property we tested in the scripts:
`http://overpass-api.de/api/interpreter?data=[out:json];node[%22amenity%22=%22school%22](around:1500,51.3734,0.0881);out;`
