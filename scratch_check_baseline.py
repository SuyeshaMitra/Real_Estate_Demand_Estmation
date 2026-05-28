import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

print("Loading data...")
df = pd.read_csv('london_data.csv')
df = df.dropna(subset=['price', 'date_of_transfer', 'district'])
df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
df['year'] = df['date_of_transfer'].dt.year
df['month'] = df['date_of_transfer'].dt.month
df = df[(df['year'] >= 2008) & (df['year'] <= 2022)].copy()

df['property_code'] = df['property_type'].astype('category').cat.codes
df['old_new_code'] = df['old_new'].astype('category').cat.codes
df['duration_code'] = df['duration'].astype('category').cat.codes
df['district_code'] = df['district'].astype('category').cat.codes

features = ['year', 'month', 'property_code', 'old_new_code', 'duration_code', 'district_code']
target = 'price'

train_df = df[df['year'] <= 2017].sample(n=100000, random_state=42)
test_df = df[df['year'] >= 2018].sample(n=50000, random_state=42)

X_train = train_df[features]
y_train_log = np.log1p(train_df[target])
y_train_raw = train_df[target]

X_test = test_df[features]
y_test_raw = test_df[target]

print("\n--- Training on Log-Transformed Target ---")
rf_log = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
rf_log.fit(X_train, y_train_log)
rf_pred_log = rf_log.predict(X_test)
rf_pred_from_log = np.expm1(rf_pred_log)

print("Mean of Actual Prices:", y_test_raw.mean())
print("Mean of Predictions (from Log Model):", rf_pred_from_log.mean())
print("MAE (Log Model):", mean_absolute_error(y_test_raw, rf_pred_from_log))

print("\n--- Training on Raw Target (No Log Transform) ---")
rf_raw = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
rf_raw.fit(X_train, y_train_raw)
rf_pred_raw = rf_raw.predict(X_test)

print("Mean of Predictions (from Raw Model):", rf_pred_raw.mean())
print("MAE (Raw Model):", mean_absolute_error(y_test_raw, rf_pred_raw))

# Let's check the percentage error
print("\n--- Bounded Accuracy & MAPE ---")
def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    med_acc = np.median(np.clip(100 - (np.abs((y_true - y_pred) / y_true) * 100), 0, 100))
    return mae, mape, med_acc

mae_l, mape_l, acc_l = get_metrics(y_test_raw, rf_pred_from_log)
print(f"Log Model: MAE = £{mae_l:,.2f}, MAPE = {mape_l:.2f}%, Median Acc = {acc_l:.2f}%")

mae_r, mape_r, acc_r = get_metrics(y_test_raw, rf_pred_raw)
print(f"Raw Model: MAE = £{mae_r:,.2f}, MAPE = {mape_r:.2f}%, Median Acc = {acc_r:.2f}%")
