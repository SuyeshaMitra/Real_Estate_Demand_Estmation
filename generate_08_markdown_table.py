import pandas as pd

df = pd.read_csv('08_Master_Results.csv')
table = "### The Ultimate Phase 08 Combinatorial Inference Table\n\n"
table += "| Phase | Feature Combination | Features Count | LightGBM MAE | Random Forest MAE | XGBoost MAE | Best Model |\n"
table += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"

df_sorted = df.sort_values(by='LGBM_MAE', ascending=True)

for index, row in df_sorted.iterrows():
    combo = row['Combo']
    features = row['Features']
    lgb_mae = row['LGBM_MAE']
    rf_mae = row['RF_MAE']
    xgb_mae = row['XGB_MAE']
    maes = {'LightGBM': lgb_mae, 'Random Forest': rf_mae, 'XGBoost': xgb_mae}
    best_model = min(maes, key=maes.get)
    name = combo.split('_', 1)[1].replace('_', ' + ')
    table += f"| **{combo.split('_')[0]}** | {name} | {features} | £{lgb_mae:,.0f} | £{rf_mae:,.0f} | £{xgb_mae:,.0f} | **{best_model}** |\n"

with open('08_Master_Table.md', 'w', encoding='utf-8') as f:
    f.write(table)
