import pandas as pd
import glob

print("=========================")
print("  MAE EVALUATION AGENT   ")
print("=========================")

files = glob.glob('prediction_validation_07*.csv')
files.sort()

for f in files:
    try:
        df = pd.read_csv(f)
        mae = (abs(df['predicted_price'] - df['actual_price'])).mean()
        print(f"{f}: £{mae:,.0f}")
    except Exception as e:
        print(f"Failed {f}: {e}")
