import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

print("======================================================")
print(" 08: GENERATING UNIFIED ENRICHED CHARTS (Year & Month)")
print("======================================================")

# Load all 5 prediction sets
df_A = pd.read_csv('07A_Predictions.csv')
df_B = pd.read_csv('07B_Predictions.csv')
df_C = pd.read_csv('07C_Predictions.csv')
df_D = pd.read_csv('07D_Predictions.csv')
df_E = pd.read_csv('07E_Combined_LightGBM_Predictions.csv')

# Merge them all based on index/order (since test sets are identical)
df_master = df_A[['year', 'month', 'price']].copy()
df_master['7A_pred'] = df_A['LightGBM_pred']
df_master['7B_pred'] = df_B['LightGBM_pred']
df_master['7C_pred'] = df_C['LightGBM_pred']
df_master['7D_pred'] = df_D['LightGBM_pred']
df_master['7E_pred'] = df_E['LightGBM_pred']

# 1. YEARLY AGGREGATION
yearly = df_master.groupby('year').mean().reset_index()

plt.figure(figsize=(14, 7))
plt.plot(yearly['year'], yearly['price'], label='Actual Price', color='black', marker='o', linewidth=4)
plt.plot(yearly['year'], yearly['7A_pred'], label='7A (Control)', color='gray', linestyle='--', marker='x')
plt.plot(yearly['year'], yearly['7B_pred'], label='7B (OSM)', color='blue', linestyle='--', marker='x')
plt.plot(yearly['year'], yearly['7C_pred'], label='7C (News SBERT)', color='green', linestyle='--', marker='x')
plt.plot(yearly['year'], yearly['7D_pred'], label='7D (Trends)', color='orange', linestyle='--', marker='x')
plt.plot(yearly['year'], yearly['7E_pred'], label='7E (All Combined)', color='purple', linestyle='-', marker='s', linewidth=2)

plt.title('Phase 07: Forecast Validation (Yearly)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Price (£)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.savefig('07_forecast_validation_yearly.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. MONTHLY AGGREGATION
monthly = df_master.groupby('month').mean().reset_index()

plt.figure(figsize=(14, 7))
plt.plot(monthly['month'], monthly['price'], label='Actual Price', color='black', marker='o', linewidth=4)
plt.plot(monthly['month'], monthly['7A_pred'], label='7A (Control)', color='gray', linestyle='--', marker='x')
plt.plot(monthly['month'], monthly['7B_pred'], label='7B (OSM)', color='blue', linestyle='--', marker='x')
plt.plot(monthly['month'], monthly['7C_pred'], label='7C (News SBERT)', color='green', linestyle='--', marker='x')
plt.plot(monthly['month'], monthly['7D_pred'], label='7D (Trends)', color='orange', linestyle='--', marker='x')
plt.plot(monthly['month'], monthly['7E_pred'], label='7E (All Combined)', color='purple', linestyle='-', marker='s', linewidth=2)

plt.title('Phase 07: Forecast Validation (Monthly Seasonality)', fontsize=16)
plt.xlabel('Month (1-12)', fontsize=12)
plt.ylabel('Average Price (£)', fontsize=12)
plt.xticks(range(1, 13))
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.savefig('07_forecast_validation_monthly.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n-> [SAVED] 07_forecast_validation_yearly.png")
print("-> [SAVED] 07_forecast_validation_monthly.png")
print("======================================================")
