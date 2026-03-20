# Real Estate Demand Estimation Architecture

This document describes the design behind the data forecasting pipeline executed in the codebase. Our goal was to ingest a massive UK Land Registry dataset, explore its features, and predict property values 5 years into the future.

## Data Flow Pipeline

```mermaid
flowchart TD
    A[Raw UK Property Data 3.2GB] -->|01_data_exploration.py| B(Identify Schema & Missing Values)
    B -->|02_data_preparation.py| C{Filter & Chunk Engine}
    C -->|Extract Target Region| D[london_data.csv ~3.9M rows]
    D -->|03_trend_analysis_and_modeling.py| E[Feature Engineering]
    E -->|Extract Temporal Features| F[Categorical Encoding]
    F -->|Time-aligned Split| G[Train 2008-2017]
    F -->|Time-aligned Split| H[Test Holdout 2018-2022]
    G --> I[Random Forest Regressor]
    G --> J[Multi-Layer Perceptron NN]
    I --> K[Evaluate MAE/RMSE on Test]
    J --> K
    K --> L[Generate Artifact Charts]
```

## Technical Decisions
* **Chunking**: Using standard pandas to read a 3.2GB unindexed CSV can consume 15GB+ RAM. The `chunksize` approach in the preparation script processes 1,000,000 rows at a time, streaming matching outputs directly to disk.
* **Temporal Cross-Validation vs. K-Fold**: We trained on a strictly historical window (10 years) to explicitly predict a future unknown window (5 years). This avoids data leakage where future economic states influence past pricing.
* **Logarithmic Target Transformation**: Due to extremely long-tailed distribution (mansions vs flats), we log-transformed the target variable before pushing it to the scikit-learn regressors.

## Model Performance
1. **Random Forest Baseline**: Evaluated well on recognizing categorical differences between London Boroughs.
2. **Neural Network Baseline**: Required heavier temporal feature injection to compete with spatial tree structures.

*Note: The raw prediction outputs contain high variability because property pricing heavily depends on unobserved features not in the registry (e.g., square footage, number of bedrooms).*
