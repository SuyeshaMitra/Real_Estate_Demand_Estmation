import pandas as pd
import numpy as np
import time
import warnings
import json
import pickle
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

print("Loading massive 253MB enriched dataset into RAM once...")
df = pd.read_csv('london_geospatial_enriched_dataset.csv')

# Time-Series Split
train_df = df[(df['year'] >= 2008) & (df['year'] <= 2017)]
test_df = df[(df['year'] >= 2018) & (df['year'] <= 2022)]

y_train = train_df['price']
y_test = test_df['price']
test_years = test_df['year'].values
test_months = test_df['month'].values
test_postcodes = test_df['postcode'].values

# Feature groups
F_LATLON = ['latitude', 'longitude']
F_OSM = ['distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km']
F_NEWS = ['sbert_sentiment_index']
F_TRENDS = ['google_trends_volume']
F_RATES = ['boe_interest_rate']

combinations = [
    ("08A_LatLon_OSM_News_Trends", F_LATLON + F_OSM + F_NEWS + F_TRENDS),
    ("08B_LatLon_OSM_News_Rates", F_LATLON + F_OSM + F_NEWS + F_RATES),
    ("08C_LatLon_OSM_Trends_Rates", F_LATLON + F_OSM + F_TRENDS + F_RATES),
    ("08D_LatLon_News_Trends_Rates", F_LATLON + F_NEWS + F_TRENDS + F_RATES),
    ("08E_LatLon_OSM_News", F_LATLON + F_OSM + F_NEWS),
    ("08F_LatLon_OSM_Trends", F_LATLON + F_OSM + F_TRENDS),
    ("08G_LatLon_OSM_Rates", F_LATLON + F_OSM + F_RATES),
    ("08H_LatLon_News_Trends", F_LATLON + F_NEWS + F_TRENDS),
    ("08I_LatLon_News_Rates", F_LATLON + F_NEWS + F_RATES),
    ("08J_LatLon_Trends_Rates", F_LATLON + F_TRENDS + F_RATES),
    ("08K_LatLon_OSM", F_LATLON + F_OSM),
    ("08L_LatLon_News", F_LATLON + F_NEWS),
    ("08M_LatLon_Trends", F_LATLON + F_TRENDS),
    ("08N_LatLon_Rates", F_LATLON + F_RATES),
    ("08O_OSM_News_Trends_Rates", F_OSM + F_NEWS + F_TRENDS + F_RATES),
    ("08P_OSM_News_Trends", F_OSM + F_NEWS + F_TRENDS),
    ("08Q_OSM_News_Rates", F_OSM + F_NEWS + F_RATES),
    ("08R_OSM_Trends_Rates", F_OSM + F_TRENDS + F_RATES),
    ("08S_OSM_News", F_OSM + F_NEWS),
    ("08T_OSM_Trends", F_OSM + F_TRENDS),
    ("08U_OSM_Rates", F_OSM + F_RATES),
    ("08V_News_Trends_Rates", F_NEWS + F_TRENDS + F_RATES),
    ("08W_News_Trends", F_NEWS + F_TRENDS),
    ("08X_News_Rates", F_NEWS + F_RATES),
    ("08Y_Trends_Rates", F_TRENDS + F_RATES),
    ("08Z_OSM_Only", F_OSM),
    ("08AA_News_Only", F_NEWS),
    ("08AB_Trends_Only", F_TRENDS),
    ("08AC_Rates_Only", F_RATES),
    ("08AD_LatLon_Only", F_LATLON)
]

results = []
postcode_errors = {}

print(f"Starting execution of {len(combinations)} combinations...")
for combo_name, features in combinations:
    print(f"\n[{combo_name}] Running with features: {features}")
    
    base_features = ['year', 'month']
    train_cols = base_features + features
    
    X_train = train_df[train_cols]
    X_test = test_df[train_cols]
    
    combo_res = {"Combo": combo_name, "Features": len(features)}
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    lgb_mae = mean_absolute_error(y_test, lgb_preds)
    lgb_acc = np.median(np.maximum(0, 100 - (np.abs(y_test - lgb_preds) / y_test) * 100))
    combo_res["LGBM_MAE"] = lgb_mae
    combo_res["LGBM_Acc"] = lgb_acc
    
    abs_errors = np.abs(y_test - lgb_preds)
    pc_df = pd.DataFrame({'postcode': test_postcodes, 'error': abs_errors})
    pc_grouped = pc_df.groupby('postcode')['error'].mean().to_dict()
    postcode_errors[combo_name] = pc_grouped
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_acc = np.median(np.maximum(0, 100 - (np.abs(y_test - xgb_preds) / y_test) * 100))
    combo_res["XGB_MAE"] = xgb_mae
    combo_res["XGB_Acc"] = xgb_acc
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_acc = np.median(np.maximum(0, 100 - (np.abs(y_test - rf_preds) / y_test) * 100))
    combo_res["RF_MAE"] = rf_mae
    combo_res["RF_Acc"] = rf_acc
    
    print(f"LGBM MAE: {lgb_mae:,.0f} | XGB MAE: {xgb_mae:,.0f} | RF MAE: {rf_mae:,.0f}")
    results.append(combo_res)
    
    pd.DataFrame(results).to_csv('08_Master_Results.csv', index=False)

print("Finished all 30 Combinations! Saving postcode errors...")
with open('08_postcode_errors.pkl', 'wb') as f:
    pickle.dump(postcode_errors, f)
print("Done.")
