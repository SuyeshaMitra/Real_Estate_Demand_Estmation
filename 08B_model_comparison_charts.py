# Import matplotlib for rendering static graph image files natively
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Apply clean visual styles natively
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="muted")

# Setup the x-axis visual labels defining the AI models tracked
models = ['Random Forest', 'XGBoost', 'LightGBM']

# ==============================================================================
# --- 08B ANALYTICS (OSM PURE GEOGRAPHY TRACK) ---
# Here we aggressively track the models stripped of Google Macro Noise!
# ==============================================================================

mae_scores_08b = [421050, 406120, 398540]
training_times_08b = [1.25, 1.45, 0.58]
median_test_accuracy_08b = [86.2, 88.5, 92.1] 

# STEP 1: PLOTTING MAE 08B
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=models, y=mae_scores_08b, hue=models, palette=['#FF9999', '#66B2FF', '#99FF99'], dodge=False)
plt.title('OSM-Isolated Model Error (MAE)\n(Lower Error = Better)', fontsize=14, pad=15)
plt.ylabel('Mean Absolute Error (£)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(mae_scores_08b):
    ax.text(i, v + 2000, f'£{v:,.0f}', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('08B_chart_model_mae_comparison.png', dpi=200)
plt.close()

# STEP 2: SPEED COMPARISON
plt.figure(figsize=(10, 6))
ax2 = sns.barplot(x=models, y=training_times_08b, hue=models, palette=['#FF9999', '#66B2FF', '#99FF99'], dodge=False)
plt.title('Execution Processing Speed (OSM Track)\n(Lower Time = Better)', fontsize=14, pad=15)
plt.ylabel('Training Time (Seconds)', fontsize=12)

for i, v in enumerate(training_times_08b):
    ax2.text(i, v + 0.05, f'{v:.2f}s', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('08B_chart_model_speed_comparison.png', dpi=200)
plt.close()

# STEP 3: ACCURACY
plt.figure(figsize=(10, 6))
ax3 = sns.barplot(x=models, y=median_test_accuracy_08b, hue=models, palette=['#FF9999', '#66B2FF', '#99FF99'], dodge=False)
plt.title('Median Validation Accuracy % (OSM Track)\n(Higher Accuracy = Better)', fontsize=14, pad=15)
plt.ylabel('Spatial Target Accuracy (%)', fontsize=12)
plt.ylim(80, 100) 

for i, v in enumerate(median_test_accuracy_08b):
    ax3.text(i, v + 0.5, f'{v}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('08B_chart_model_accuracy_comparison.png', dpi=200)
plt.close()

print("Successfully generated 08B (OSM Track) comparison chart series!")
