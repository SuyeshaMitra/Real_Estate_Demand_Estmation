import os

combinations = [
    ("08A_LatLon_OSM_News_Trends", "['latitude', 'longitude', 'distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'sbert_sentiment_index', 'google_trends_volume']"),
    ("08B_LatLon_OSM_News_Rates", "['latitude', 'longitude', 'distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'sbert_sentiment_index', 'boe_interest_rate']"),
    ("08C_LatLon_OSM_Trends_Rates", "['latitude', 'longitude', 'distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'google_trends_volume', 'boe_interest_rate']"),
    ("08D_LatLon_News_Trends_Rates", "['latitude', 'longitude', 'sbert_sentiment_index', 'google_trends_volume', 'boe_interest_rate']"),
    ("08E_LatLon_OSM_News", "['latitude', 'longitude', 'distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'sbert_sentiment_index']"),
    ("08F_LatLon_OSM_Trends", "['latitude', 'longitude', 'distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'google_trends_volume']"),
    ("08G_LatLon_OSM_Rates", "['latitude', 'longitude', 'distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'boe_interest_rate']"),
    ("08H_LatLon_News_Trends", "['latitude', 'longitude', 'sbert_sentiment_index', 'google_trends_volume']"),
    ("08I_LatLon_News_Rates", "['latitude', 'longitude', 'sbert_sentiment_index', 'boe_interest_rate']"),
    ("08J_LatLon_Trends_Rates", "['latitude', 'longitude', 'google_trends_volume', 'boe_interest_rate']"),
    ("08K_LatLon_OSM", "['latitude', 'longitude', 'distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km']"),
    ("08L_LatLon_News", "['latitude', 'longitude', 'sbert_sentiment_index']"),
    ("08M_LatLon_Trends", "['latitude', 'longitude', 'google_trends_volume']"),
    ("08N_LatLon_Rates", "['latitude', 'longitude', 'boe_interest_rate']"),
    ("08O_OSM_News_Trends_Rates", "['distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'sbert_sentiment_index', 'google_trends_volume', 'boe_interest_rate']"),
    ("08P_OSM_News_Trends", "['distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'sbert_sentiment_index', 'google_trends_volume']"),
    ("08Q_OSM_News_Rates", "['distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'sbert_sentiment_index', 'boe_interest_rate']"),
    ("08R_OSM_Trends_Rates", "['distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'google_trends_volume', 'boe_interest_rate']"),
    ("08S_OSM_News", "['distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'sbert_sentiment_index']"),
    ("08T_OSM_Trends", "['distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'google_trends_volume']"),
    ("08U_OSM_Rates", "['distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'boe_interest_rate']"),
    ("08V_News_Trends_Rates", "['sbert_sentiment_index', 'google_trends_volume', 'boe_interest_rate']"),
    ("08W_News_Trends", "['sbert_sentiment_index', 'google_trends_volume']"),
    ("08X_News_Rates", "['sbert_sentiment_index', 'boe_interest_rate']"),
    ("08Y_Trends_Rates", "['google_trends_volume', 'boe_interest_rate']"),
    ("08Z_OSM_Only", "['distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km']"),
    ("08AA_News_Only", "['sbert_sentiment_index']"),
    ("08AB_Trends_Only", "['google_trends_volume']"),
    ("08AC_Rates_Only", "['boe_interest_rate']"),
    ("08AD_LatLon_Only", "['latitude', 'longitude']")
]

template = """import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

print("Executing Phase 08: {name}")

# 1. Load the Enriched Dataset
print("Loading dataset...")
df = pd.read_csv('london_geospatial_enriched_dataset.csv')

# 2. Strict Chronological Time-Series Split (2008-2017 Train, 2018-2022 Test)
train_df = df[(df['year'] >= 2008) & (df['year'] <= 2017)]
test_df = df[(df['year'] >= 2018) & (df['year'] <= 2022)]

y_train = train_df['price']
y_test = test_df['price']

# 3. Apply the specific ablation configuration:
features = {features}
train_cols = ['year', 'month'] + features
X_train = train_df[train_cols]
X_test = test_df[train_cols]

print(f"Training on columns: {{train_cols}}")

# 4. Train LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)
lgb_mae = mean_absolute_error(y_test, lgb_preds)
lgb_acc = np.median(np.maximum(0, 100 - (np.abs(y_test - lgb_preds) / y_test) * 100))
print(f"LightGBM MAE: £{{lgb_mae:,.0f}} | Accuracy: {{lgb_acc:.2f}}%")

# 5. Train XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_acc = np.median(np.maximum(0, 100 - (np.abs(y_test - xgb_preds) / y_test) * 100))
print(f"XGBoost MAE: £{{xgb_mae:,.0f}} | Accuracy: {{xgb_acc:.2f}}%")

# 6. Train Random Forest
rf_model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_acc = np.median(np.maximum(0, 100 - (np.abs(y_test - rf_preds) / y_test) * 100))
print(f"Random Forest MAE: £{{rf_mae:,.0f}} | Accuracy: {{rf_acc:.2f}}%")

print("Model execution completed!")
"""

for combo in combinations:
    filename = combo[0] + ".py"
    content = template.format(name=combo[0], features=combo[1])
    with open(filename, "w") as f:
        f.write(content)

print(f"Generated {len(combinations)} files successfully!")
