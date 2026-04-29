import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

print("Executing Phase 08XD: Neural Network Isolation")

# 1. Load the Enriched Dataset
print("Loading dataset...")
df = pd.read_csv('london_geospatial_enriched_dataset.csv')

# 2. Strict Chronological Time-Series Split (2008-2017 Train, 2018-2022 Test)
train_df = df[(df['year'] >= 2008) & (df['year'] <= 2017)]
test_df = df[(df['year'] >= 2018) & (df['year'] <= 2022)]

y_train = train_df['price']
y_test = test_df['price']

# 3. Apply the ultimate configuration (All Features)
features = ['latitude', 'longitude', 'distance_to_nearest_hospital_km', 'distance_to_nearest_bank_km', 'distance_to_nearest_school_km', 'distance_to_nearest_station_km', 'sbert_sentiment_index', 'google_trends_volume', 'boe_interest_rate']
train_cols = ['year', 'month'] + features
X_train = train_df[train_cols]
X_test = test_df[train_cols]

print(f"Training on columns: {train_cols}")

# Neural Networks mathematically require StandardScaler to prevent gradient explosion
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Neural Network (Multi-Layer Perceptron)
print("Training Neural Network (This will be extremely slow on tabular geospatial data...)")
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=20, random_state=42)
nn_model.fit(X_train_scaled, y_train)

nn_preds = nn_model.predict(X_test_scaled)
nn_mae = mean_absolute_error(y_test, nn_preds)
nn_acc = np.median(np.maximum(0, 100 - (np.abs(y_test - nn_preds) / y_test) * 100))

print(f"Neural Network MAE: £{nn_mae:,.0f} | Accuracy: {nn_acc:.2f}%")
print("Model execution completed!")
