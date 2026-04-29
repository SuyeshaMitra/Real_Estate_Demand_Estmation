import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

print("Executing Phase 08: 08L_LatLon_News")

# 1. Load the Enriched Dataset
print("Loading dataset...")
df = pd.read_csv('london_geospatial_enriched_dataset.csv')

# 2. Strict Chronological Time-Series Split (2008-2017 Train, 2018-2022 Test)
train_df = df[(df['year'] >= 2008) & (df['year'] <= 2017)]
test_df = df[(df['year'] >= 2018) & (df['year'] <= 2022)]

y_train = train_df['price']
y_test = test_df['price']

# 3. Apply the specific ablation configuration:
features = ['latitude', 'longitude', 'sbert_sentiment_index']
train_cols = ['year', 'month'] + features
X_train = train_df[train_cols]
X_test = test_df[train_cols]

print(f"Training on columns: {train_cols}")

# 4. Train LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)
lgb_mae = mean_absolute_error(y_test, lgb_preds)
lgb_acc = np.median(np.maximum(0, 100 - (np.abs(y_test - lgb_preds) / y_test) * 100))
print(f"LightGBM MAE: £{lgb_mae:,.0f} | Accuracy: {lgb_acc:.2f}%")

# 5. Train XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_acc = np.median(np.maximum(0, 100 - (np.abs(y_test - xgb_preds) / y_test) * 100))
print(f"XGBoost MAE: £{xgb_mae:,.0f} | Accuracy: {xgb_acc:.2f}%")

# 6. Train Random Forest
rf_model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_acc = np.median(np.maximum(0, 100 - (np.abs(y_test - rf_preds) / y_test) * 100))
print(f"Random Forest MAE: £{rf_mae:,.0f} | Accuracy: {rf_acc:.2f}%")

print("Model execution completed!")
