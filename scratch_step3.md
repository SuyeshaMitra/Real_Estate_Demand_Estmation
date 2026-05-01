## Step 3: The Baseline Models (Non-Spatial)
Before migrating to complex geographical mapping, we tested four algorithms (Random Forest, Neural Network, XGBoost, and LightGBM) on simple text-based district data (`03_trend_analysis_and_modeling.py`) to establish a non-spatial baseline.

**Training & Evaluation Setup:**
* **Training Data:** Historical property prices spanning from 2008 to 2017.
* **Testing Data:** Predicted blindly on the 5-year future holdout block (2018 to 2022).
* **MAE Calculation Engine:** The Mean Absolute Error strictly calculates the exact absolute financial difference (in £) between the True Selling Price and the AI's Predicted Price across all testing rows, producing a flat physical error margin.

| Baseline Model | Mean Absolute Error (MAE) | Median Accuracy | Training Speed | Pros | Cons |
|----------------|---------------------------|-----------------|----------------|------|------|
| **LightGBM 🏆 (Best)** | **£456,439** | **75.68%** | ~1.6 sec | Lightning fast and aggressively isolates extreme outliers natively. | Cannot define localized geometric spatial bounds without coordinates. |
| **Random Forest** | £470,591 | 74.78% | **~0.4 sec** | Highly stable and executes almost instantly on generic categorical datasets. | Relies on averaging, missing nuanced neighborhood wealth boundaries. |
| **XGBoost** | £494,479 | 73.82% | ~2.3 sec | Powerful gradient engine designed to deeply minimize localized errors. | Drastically overfits flat categorical text features, degrading accuracy. |
| **Neural Network** | £546,571 | 65.93% | ~2.2 sec | Theoretically mapped for high complexity non-linear relationships. | Systematically crashes. Flat tabular strings lack depth for neural logic. |

### Conclusion & Explanation: Why LightGBM Won the Baseline Contest

| Evaluation Aspect | LightGBM Result vs Competitors | Why LightGBM Dominated |
|-------------------|--------------------------------|------------------------|
| **Prediction Error (MAE)** | **£456,439** (Lowest Error) | It uniquely handles continuous data as discrete histograms, dropping massive mathematical noise that caused Neural Networks and XGBoost to massively overfit the baseline. |
| **Overall Accuracy** | **75.68%** (Highest Accuracy) | LightGBM uses Leaf-wise growth. While Random Forest averages broadly, LightGBM actively isolates the absolute hardest-to-predict luxury homes and dynamically hyper-focuses splits entirely on them. |
| **Compute Execution Speed** | **~1.6 sec** (Extremely Fast) | It bypasses forcing symmetric splits uniformly across the entire dataset like XGBoost, ignoring stable average zones completely to focus compute solely on breaking volatility. |

**Mathematical Metric Calculation Example:**
| Metric | Mathematical Formula Engine | Example Calculation Output (Based on 4 Models) |
|--------|-----------------------------|------------------------------------------------|
| **Mean Absolute Error (MAE)** | `MAE = Average( ABS(True_Price - Predicted_Price) )` | If LightGBM predicts a house is £400,000 but the True Sold Price was £500,000, the absolute physical error is recorded exactly as £100,000. |
| **Median Accuracy** | `Accuracy = 100 - (ABS(True - Predicted) / True) * 100` | Following the example above: (£100,000 error / £500,000 price) = 0.20 off. `100 - 20% = 80.00% Accuracy`. |

### Baseline Models Detailed Analytics (Time-Series Breakdown)

#### A) Comprehensive Metric Validation

**i) Absolute Error (MAE), Aggregate Median Accuracy, and Execution Processing Speed -** We benchmarked 4 regression models. The baseline performance is:
* **LightGBM (Winner):** MAE: £456,439 | Accuracy: 75.68% | Speed: ~1.94s
* **Random Forest:** MAE: £470,591 | Accuracy: 74.78% | Speed: ~0.48s
* **XGBoost:** MAE: £494,479 | Accuracy: 73.82% | Speed: ~2.34s
* **Neural Network:** MAE: £546,571 | Accuracy: 65.93% | Speed: ~2.93s

**ii) Accurate Values Across Models**
The execution strictly ensures that **values are completely accurate across models**, calculated identically over the identical test dataset. In prior local versions, unbound percentage skewing caused mathematical variance; the script now correctly normalizes the percentage variance uniformly across all models.

**iii) Explore Accuracy: What defines an "Accurate" Prediction?**
To determine how "accurate" a machine learning model is, we cannot just look at the raw physical error (like £50,000), because £50,000 is a massive error on a £100,000 home, but a tiny error on a £5,000,000 mansion. Therefore, we translate error into a normalized percentage, governed by three strict mathematical rules:
* **1. The Variance Rule (Calculating Error %):** First, we calculate the absolute difference between the Predicted Price and the True Sold Price, and divide it by the True Price. For example, if a house sells for £500,000 and the model predicts £400,000, the error is £100,000. `(100,000 / 500,000 = 0.20 or 20% Error)`.
* **2. The Inversion Rule (Error to Accuracy):** Accuracy is simply the mathematical opposite of error. We subtract the percentage error from 100%. Following the example above: `100% - 20% Error = 80% Accuracy`.
* **3. The Bounding Rule (The Zero Floor):** Models can occasionally make catastrophic guesses on strange outliers. If a model guesses £1,500,000 on a £500,000 home, the error is 200%. Mathematically, `100% - 200% = -100% Accuracy`. If we allow negative numbers, a single terrible guess could artificially ruin the overall baseline score. Therefore, we institute a rigorous mathematical floor explicitly at **0%** (`np.clip(accuracy, 0, 100)`). An absolutely wrong prediction is simply capped at 0%, protecting the final metric from unbounded negative drift.
* **Conclusion:** When the pipeline states that LightGBM has a `75.68% Median Accuracy`, it implies that if you select an average property from the dataset, the model's prediction will reliably be within a ~24.3% variance of the true physical selling price.

**iv) Consistent Calculations Over All Features**
**Consistent Auditing Over Features:** The system executes explicit validation over all 4 baseline algorithms identically enforcing `Features = ['year', 'month', 'property_code', 'old_new_code', 'duration_code', 'district_code']`.

**v) Calculate Accuracy, Error and Speed - For 5 Years + Every Monthly Average**
Because standard globally averaged numbers can mathematically mask seasonal failures, the Python script natively breaks predictions out chronologically tracking structural degradation points.
* **5-Year View:** We explicitly track the absolute MAE and Accuracy dynamically across the 2018-2022 holdout block. 
* **Monthly View:** We fold the historic time-series back exclusively over the 12 calendar months (`1 to 12`) to explicitly track the seasonality pattern.
  * **Explanation:** Instead of tracking 2018, 2019, 2020 chronologically, the pipeline takes *all* Januaries across the entire 5-year holdout block and averages their errors together into "Month 1". It then averages all Februaries into "Month 2", and so on up to "Month 12". 
  * **Why we do this:** This mathematically exposes the cyclical real-estate "seasonality" (e.g., the fact that housing transaction volumes and errors predictably spike every single Spring, regardless of what year it is). 
  * **Supporting Evidence:** You can visibly observe this cyclical pattern occurring in the **Forecast Validation Chart - Monthly** (embedded below in Section B), where the black true-price line and every model's prediction error sharply spikes specifically at `Month 3` (March).

**vi) Error Pattern on Years and Months Wise Separately**

**A. 5-Year Yearly Breakdown Observation**
When strictly plotting predicted median accuracy natively separated by the 2018-2022 holdout years independently:

![Yearly Accuracy Trend](03_accuracy_trend_yearly.png)

* **Which model is better and why:** **LightGBM** (the purple diamond line) is universally the most accurate model across every single year. It particularly excels during the **2021 Post-Pandemic Spike**, hitting a peak **76.68% Median Accuracy** (dropping error to £400,395). 
* **Why?** LightGBM uses leaf-wise histogram bucketing. When macro-economic events cause sudden chaotic shifts in housing wealth, Random Forest (blue line) attempts to average the chaos, and XGBoost (green line) chases the residual noise. LightGBM simply isolates the wild new outliers into their own specific leaves, keeping the median accuracy for standard homes exceptionally stable.

**B. Monthly Seasonality Breakdown Observation**
When actively sorting all historical holdout lines solely by individual chronological Month (`1 - 12`) to map the cyclical seasons:

![Monthly Accuracy Trend](03_accuracy_trend_monthly.png)

* **Which model is better and why:** **LightGBM** strictly dominates every single month of the year. The models uniformly hit their absolute highest accuracy/lowest error in the dead of Winter (**Month 2: February**), where LightGBM peaks at **76.46% accuracy** (£355,499 MAE) because market volatility and trading volumes are at their lowest.
* **Why?** Conversely, notice the massive, aggressive failure drift that triggers uniformly across all 4 machine learning pipelines exactly tracking the Spring Real-Estate Rush (**Month 3: March**). Pure transactional volume and aggressive bidding wars cause severe unpredictability. LightGBM survives this the best because it naturally handles volatile continuous variables (like unpredictable spring closing prices) faster than XGBoost's sequential boosting. Neural Networks (red line) fail completely (hovering near 65%), lacking the deep node logic to understand seasonal time cycles from flat text integers.

#### B) Predictive Baseline Visualizations

Below are the physically generated structured charts proving the analytics above for the 4 models:

**1. Historical Chart**
Visibly maps the 2008-2022 London property physical price graph.
![Historical Trend](03_historical_trend.png)

**2. Forecast Validation Chart (Actual Average Vs Forecasted Price) - Yearly**
Directly plots exactly how the 4 machine learning models (dashed lines) historically drifted against the solid black true price line over the 5-year validation bracket.
![Yearly Validation](03_forecast_validation_yearly.png)

**3. Forecast Validation Chart (Actual Average Vs Forecasted Price) - Monthly**
Plots this same validation against the 1-12 month cyclical cycle.
![Monthly Validation](03_forecast_validation_monthly.png)

### Baseline Output Visuals
To scientifically prove the categorical limits of the baseline model, the algorithms automatically dump their evaluation analytics locally into three charting arrays:
![Baseline MAE Comparison](03_chart_model_mae_comparison.png)
![Baseline Accuracy Comparison](03_chart_model_accuracy_comparison.png)
![Baseline Speed Comparison](03_chart_model_speed_comparison.png)

---

