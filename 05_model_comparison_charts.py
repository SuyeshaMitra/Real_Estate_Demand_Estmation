import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Apply clean visual styles
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="muted")

models = ['Random Forest', 'XGBoost', 'LightGBM']

# Sourced directly from our established validation splits
mae_scores = [424476, 410339, 401075]
training_times = [1.22, 3.53, 0.55]
median_test_accuracy = [85.5, 87.1, 91.3] # Aggregate Median Accuracy % on holdouts

# 1. Plot MAE Comparison
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=models, y=mae_scores, hue=models, palette=['#FF9999', '#66B2FF', '#99FF99'], dodge=False)
plt.title('Geospatial Model Error (MAE) Comparison\n(Lower Error = Better)', fontsize=14, pad=15)
plt.ylabel('Mean Absolute Error (£)', fontsize=12)
# plt.ylim(380000, 430000) # Zoom in for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(mae_scores):
    ax.text(i, v + 2000, f'£{v:,.0f}', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('chart_model_mae_comparison.png', dpi=200)
plt.close()

# 2. Plot Speed Comparison
plt.figure(figsize=(10, 6))
ax2 = sns.barplot(x=models, y=training_times, hue=models, palette=['#FF9999', '#66B2FF', '#99FF99'], dodge=False)
plt.title('Execution Processing Speed (100k records)\n(Lower Time = Better)', fontsize=14, pad=15)
plt.ylabel('Training Time (Seconds)', fontsize=12)

for i, v in enumerate(training_times):
    ax2.text(i, v + 0.05, f'{v:.2f}s', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('chart_model_speed_comparison.png', dpi=200)
plt.close()

# 3. Plot Accuracy Score
plt.figure(figsize=(10, 6))
ax3 = sns.barplot(x=models, y=median_test_accuracy, hue=models, palette=['#FF9999', '#66B2FF', '#99FF99'], dodge=False)
plt.title('Median Validation Accuracy %\n(Higher Accuracy = Better)', fontsize=14, pad=15)
plt.ylabel('Spatial Target Accuracy (%)', fontsize=12)
plt.ylim(80, 100) # Focus domain

for i, v in enumerate(median_test_accuracy):
    ax3.text(i, v + 0.5, f'{v}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('chart_model_accuracy_comparison.png', dpi=200)
plt.close()

print("Successfully generated comparison chart PNGs: MAE, Speed, and Accuracy.")
