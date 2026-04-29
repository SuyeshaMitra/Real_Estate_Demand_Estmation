import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("======================================================")
print(" 07A: EXECUTING ABLATION (Lat/Lon + Years Only)")
print("======================================================")

df = pd.read_csv('london_geospatial_enriched_dataset.csv')

le = LabelEncoder()
df['property_type_encoded'] = le.fit_transform(df['property_type'].astype(str))
df['old_new_encoded'] = le.fit_transform(df['old_new'].astype(str))
df['duration_encoded'] = le.fit_transform(df['duration'].astype(str))

# Baseline Features ONLY
features = [
    'year', 'month', 'latitude', 'longitude', 
    'property_type_encoded', 'old_new_encoded', 'duration_encoded'
]
target = 'price'

train_df = df[(df['year'] >= 2008) & (df['year'] <= 2017)]
test_df = df[(df['year'] >= 2018) & (df['year'] <= 2022)]

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

models = {
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=100, num_leaves=31, random_state=42, n_jobs=-1)
}

results = {}
test_df_copy = test_df.copy()

for name, model in models.items():
    print(f"\n-> Training {name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    predictions = model.predict(X_test)
    test_df_copy[f'{name}_pred'] = predictions
    test_df_copy[f'{name}_error'] = np.abs(test_df_copy['price'] - predictions)
    test_df_copy[f'{name}_acc'] = np.maximum(0, 100 - (test_df_copy[f'{name}_error'] / test_df_copy['price']) * 100)
    
    global_mae = test_df_copy[f'{name}_error'].mean()
    global_acc = test_df_copy[f'{name}_acc'].median()
    
    results[name] = {
        'MAE': global_mae,
        'Accuracy': global_acc,
        'Train Time': train_time
    }
    print(f"[{name}] MAE: £{global_mae:,.2f} | Acc: {global_acc:.2f}% | Time: {train_time:.2f}s")

# Plotting Combined Historical vs Forecast Chart
plt.figure(figsize=(14, 7))
historical_yearly = df.groupby('year')['price'].mean().reset_index()
plt.plot(historical_yearly['year'], historical_yearly['price'], label='Actual Historical Price', color='black', marker='o', linewidth=3)

colors = ['red', 'green', 'orange']
for idx, name in enumerate(models.keys()):
    forecast_yearly = test_df_copy.groupby('year')[f'{name}_pred'].mean().reset_index()
    plt.plot(forecast_yearly['year'], forecast_yearly[f'{name}_pred'], label=f'{name} Forecast', color=colors[idx], marker='X', linestyle='--', linewidth=2)

plt.title('07A Control Baseline: Actual vs Forecasted Prices (Lat/Lon Only)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Price (£)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.savefig('07A_Historical_vs_Forecast_Prices.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n-> [SAVED] 07A_Historical_vs_Forecast_Prices.png")
print("======================================================")
