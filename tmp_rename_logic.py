import os

files = ['07_Features_Random_Forest_modeling.py','07_Features_XGBoost_modeling.py','07_Features_LightGBM_modeling.py','07_04_vs_07_model_comparison.py', 'Walkthrough.md']

for f in files:
    with open(f, 'r', encoding='utf-8') as r:
        c = r.read()
        
    c = c.replace('06_Features_', '07_Features_')
    c = c.replace('prediction_validation_06_', 'prediction_validation_07_')
    c = c.replace('prediction_validation_xgb.csv', 'prediction_validation_07_xgboost.csv')
    c = c.replace('06A_chart_feature_impact_comparison.png', '07_chart_feature_impact_comparison.png')
    c = c.replace('06D_04_vs_06_model_comparison.py', '07_04_vs_07_model_comparison.py')
    
    with open(f, 'w', encoding='utf-8') as w:
        w.write(c)

print("Internal references updated to 07 successfully.")
