import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset for Chart & CSV generation...")
df = pd.read_csv('london_geospatial_enriched_dataset.csv')
train_df = df[(df['year'] >= 2008) & (df['year'] <= 2017)]
test_df = df[(df['year'] >= 2018) & (df['year'] <= 2022)]

y_train = train_df['price']
y_test = test_df['price']
actual_yearly = test_df.groupby('year')['price'].mean().reset_index()

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

for combo_name, features in combinations:
    print(f"Generating Chart & CSV for {combo_name}...")
    train_cols = ['year', 'month'] + features
    
    lgb_model = lgb.LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    lgb_model.fit(train_df[train_cols], y_train)
    preds = lgb_model.predict(test_df[train_cols])
    
    # Generate CSV
    out_df = pd.DataFrame({
        'year': test_df['year'],
        'month': test_df['month'],
        'actual_price': test_df['price'],
        'predicted_price': preds
    })
    
    # Save a lightweight aggregated CSV so GitHub doesn't crash
    agg_df = out_df.groupby(['year', 'month']).mean().reset_index()
    agg_df.to_csv(f"{combo_name}_Results.csv", index=False)
    
    # Generate Chart
    yearly_preds = out_df.groupby('year')['predicted_price'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    plt.plot(actual_yearly['year'], actual_yearly['price'], label='Actual Price', color='blue', marker='o', linewidth=2)
    plt.plot(yearly_preds['year'], yearly_preds['predicted_price'], label=f'Forecast ({combo_name})', color='red', linestyle='--', marker='x', linewidth=2)
    plt.title(f"Forecast vs Actual: {combo_name}", fontsize=12, fontweight='bold')
    plt.xlabel("Year")
    plt.ylabel("Average Price (£)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{combo_name}_Chart.png", dpi=100)
    plt.close()

print("All 30 CSVs and Charts generated successfully!")
