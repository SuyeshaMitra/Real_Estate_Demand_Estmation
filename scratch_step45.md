## Step 4 & 5: Train Models below Data Feed using - (Latitude and Longitude)

> [!NOTE]
> **Why was the Neural Network omitted from Step 04 onwards?**
> During the Step 03 non-spatial baseline test, the Multi-Layer Perceptron (Neural Network) proved incredibly inefficient at processing tabular text features compared to Tree-based algorithms, resulting in the highest Absolute Error (£650,227). Neural Networks require normalized, dense floating-point matrices to function properly, making them computationally unstable when dealing with sparse categorical data like 30,000 unique postal districts. Therefore, it was logically decommissioned, and we structurally transitioned to racing only the 3 superior Tree-based algorithms (Random Forest, XGBoost, LightGBM) inside the geospatial arena.
> 
> Real estate pricing is dictated precisely by physical location. We transitioned the remaining **3 Tree-Based ML models** (`Random Forest`, `XGBoost`, `LightGBM`) to observe how varying mathematical approaches manage geometric spatial proximity differently once `Latitude` and `Longitude` are properly extracted.

**Pipeline Sequence & Training Data Generation (Generating `london_geospatial_dataset.csv`)**: 
To construct the spatial matrix, the pipeline executes the following rigid sequence:
1. **Temporal Filtering:** We first load the core `london_data.csv` dataset (which was physically filtered in Step 02 to strictly isolate only properties inside the `GREATER LONDON` county) and then we temporally filter those property records down to our target 15-year historical block (**Years: 2008 to 2022**). 
2. **Geospatial Mapping (No Time Dependency):** We extract the unique postal codes from those 15 years and pass them directly into the `pgeocode` module. `pgeocode` securely queries the **offline Great Britain (GB) geospatial resultset** (open-source [GeoNames GB.zip Dataset](http://download.geonames.org/export/zip/GB.zip)). *Note: This offline query operates strictly on static Postal Codes to retrieve physical **Latitude** and **Longitude** coordinates; it does not process or care about timestamp data.*
3. **Merging & Exporting the Unified Grid:** Once the static Lat/Lon coordinates are successfully extracted, they are merged back onto the timestamped property records. We then physically export this completely unified matrix to **[london_geospatial_dataset.csv](london_geospatial_dataset.csv)**. 

This resulting massive, offline geometric/time-series dataset is then permanently fed identically into the Machine Learning algorithms below (trained strictly on 2008-2017) and serves as the master baseline dataset for all future modeling.



### A) Comprehensive Geospatial Metric Validation

**i) Absolute Error (MAE), Aggregate Median Accuracy, and Execution Processing Speed -** We benchmarked the 3 spatial regression models. The spatial performance is:
* **LightGBM (Winner):** MAE: £395,634 | Accuracy: 78.81% | Speed: ~0.23s
* **XGBoost:** MAE: £404,452 | Accuracy: 78.09% | Speed: ~2.71s
* **Random Forest:** MAE: £430,946 | Accuracy: 76.07% | Speed: ~1.44s

*(Note: The integration of physical spatial coordinates massively dropped the baseline LightGBM error from £456k down to £395k, instantly proving that geographic mapping is fundamentally superior to text parsing).*

**ii) Accurate Values Across Models**
The execution strictly ensures that **values are completely accurate across models**, calculated identically over the identical spatial test dataset. All percentage variances are normalized identically.

**iii) Explore Accuracy: What defines an "Accurate" Prediction?**
To determine how "accurate" a spatial model is, we govern the percentage by three strict mathematical rules:
* **1. The Variance Rule (Calculating Error %):** We calculate the absolute difference between the Predicted Price and the True Sold Price, and divide it by the True Price.
* **2. The Inversion Rule (Error to Accuracy):** We subtract the percentage error from 100%.
* **3. The Bounding Rule (The Zero Floor):** Because spatial models can occasionally guess catastrophically on ultra-luxury mansions, we institute a rigorous mathematical floor explicitly at **0%** (`np.clip(accuracy, 0, 100)`). An absolutely wrong prediction is capped at 0%, protecting the median metric from unbounded negative drift.
* **Conclusion:** When LightGBM shows a `78.81% Median Accuracy`, its spatial predictions reliably land within a ~21.2% variance of the physical selling price.

**iv) Consistent Calculations Over All Features**
**Consistent Auditing Over Features:** The system executes explicit validation over all 3 spatial algorithms identically enforcing `Features = ['year', 'month', 'property_code', 'old_new_code', 'duration_code', 'latitude', 'longitude']`.

**v) Calculate Accuracy, Error and Speed - For 5 Years + Every Monthly Average**
Because standard globally averaged numbers can mathematically mask seasonal failures, the script natively breaks predictions out chronologically tracking structural degradation points.
* **5-Year View:** We explicitly track the absolute MAE and Accuracy dynamically across the 2018-2022 holdout block. 
* **Monthly View:** We fold the historic time-series back exclusively over the 12 calendar months (`1 to 12`) to explicitly track the seasonality pattern.

**vi) Error Pattern on Years and Months Wise Separately**

**A. 5-Year Yearly Breakdown Observation**
![Geospatial Yearly Accuracy Trend](04_accuracy_trend_yearly.png)
* **Which model is better and why:** **LightGBM** is universally the most accurate model across every single year. It dominates the 2021 post-pandemic spike. Its leaf-wise histogram bucketing algorithm is structurally superior to depth-wise (XGBoost) or tree-averaging (Random Forest) at isolating sudden chaotic shifts in housing wealth.

**B. Monthly Seasonality Breakdown Observation**
![Geospatial Monthly Accuracy Trend](04_accuracy_trend_monthly.png)
* **Which model is better and why:** **LightGBM** survives the massive Spring Real-Estate Rush (Month 3) better than the others. Pure transactional volume causes severe unpredictability, but LightGBM's binning handles volatile continuous spatial variables natively faster.

**vii) Average Error Distribution Over Postal Codes**
To mathematically prove exactly *where* the models struggle the most, we grouped the test dataset entirely by physical postal code (`outcode`) and plotted the Average Error distribution across the Top 50 most active London districts.

![Postcode Error Distribution](04_error_distribution_by_postcode.png)
* **Distribution & Pattern Plot Observation:** The error is absolutely not distributed evenly. Specific high-density, ultra-wealthy postal codes drastically spike the error bounds for XGBoost and Random Forest. 
* **Why LightGBM Wins Spatially:** Notice that the LightGBM error bars consistently remain visibly lower across the hardest-to-predict postcodes. This physically proves that leaf-wise splits isolate extremely expensive spatial bounding boxes mathematically better than block-averaging coordinates.

### B) Predictive Geospatial Visualizations

Below are the physically generated structured charts tracking the spatial drift:

**1. Historical Chart**
Visibly maps the 2008-2022 London property physical price graph using spatial logic.
![Geospatial Historical Trend](04_historical_trend.png)

**2. Forecast Validation Chart (Actual Average Vs Forecasted Price) - Yearly**
Directly plots exactly how the 3 spatial machine learning models historically drifted against the true price line over the 5-year validation bracket.
![Geospatial Yearly Validation](04_forecast_validation_yearly.png)

**3. Forecast Validation Chart (Actual Average Vs Forecasted Price) - Monthly**
Plots this same validation explicitly tracking the 1-12 month cyclical cycle.
![Geospatial Monthly Validation](04_forecast_validation_monthly.png)

