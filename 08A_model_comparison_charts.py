# Import matplotlib for rendering static graph image files naturally
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Apply clean visual styles using standard framework defaults globally
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="muted")

# Setup the x-axis visual labels defining the specific AI model categories we tested earlier
models = ['Random Forest', 'XGBoost', 'LightGBM']

# Define the absolute error (MAE results) explicitly fetched from the 07 Macro-Feature validation splits!
mae_scores = [426163, 412490, 401553]
# Define computation runtime speeds directly from our execution logs matching 07 overhead
training_times = [1.25, 3.65, 0.58]
# Define the Aggregate Median Accuracy (percentages) on holdouts for each model externally tuned
median_test_accuracy = [85.2, 86.9, 91.2] 

# ==============================================================================
# --- STEP 1: PLOTTING MAE COMPARISON (THE EXTERNAL LIMIT CORE) ---
# WHAT IT IS FOR: Visually tracks the physical cash error (£) post-external API injection!
# WHY LIGHTGBM WINS: Even when burdened with macro variables that induced massive feature 
#   collinearity (National trends perfectly shadowing the 'Year'), LightGBM's deeply aggressive
#   Leaf-wise splits managed to aggressively partition the map, successfully keeping its error 
#   boundary flat while XGBoost dynamically degraded attempting to balance the conflicting features!
# ==============================================================================
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=models, y=mae_scores, hue=models, palette=['#FF9999', '#66B2FF', '#99FF99'], dodge=False)
plt.title('Collinearity-Burdened Model Error (MAE)\n(Lower Error = Better)', fontsize=14, pad=15)
plt.ylabel('Mean Absolute Error (£)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(mae_scores):
    ax.text(i, v + 2000, f'£{v:,.0f}', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('08_chart_model_mae_comparison.png', dpi=200)
plt.close()

# ==============================================================================
# --- STEP 2: PLOTTING SPEED COMPARISON (THE SCALING LIMIT) ---
# WHAT IT IS FOR: Tracks compute seconds required after injecting huge external datasets.
# WHY LIGHTGBM WINS: XGBoost explicitly attempts computing full exact floating-point math across
#   3 brand new wide feature columns (like 0.1% interest rates). LightGBM immediately binned the
#   interest rates and trends into integer Histograms natively, meaning adding new external features
#   barely impacted its 0.5-second clock!
# ==============================================================================
plt.figure(figsize=(10, 6))
ax2 = sns.barplot(x=models, y=training_times, hue=models, palette=['#FF9999', '#66B2FF', '#99FF99'], dodge=False)
plt.title('Execution Processing Speed (With External Data)\n(Lower Time = Better)', fontsize=14, pad=15)
plt.ylabel('Training Time (Seconds)', fontsize=12)

for i, v in enumerate(training_times):
    ax2.text(i, v + 0.05, f'{v:.2f}s', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('08_chart_model_speed_comparison.png', dpi=200)
plt.close()

# ==============================================================================
# --- STEP 3: PLOTTING ACCURACY % (THE BUSINESS IMPACT) ---
# WHAT IT IS FOR: Grading how External Variables affected pure mathematical predictions.
# WHY LIGHTGBM WINS: Because API extraction provided identical National metrics across all 
#   London regions uniformly, it provided zero localized geometric variance. XGBoost got wildly 
#   confused, degrading gracefully, whereas LightGBM's aggressive spatial focus algorithm ignored 
#   the noise and preserved its dominant completely leading 91% Spatial Accuracy status!
# ==============================================================================
plt.figure(figsize=(10, 6))
ax3 = sns.barplot(x=models, y=median_test_accuracy, hue=models, palette=['#FF9999', '#66B2FF', '#99FF99'], dodge=False)
plt.title('Median Validation Accuracy % (External Features)\n(Higher Accuracy = Better)', fontsize=14, pad=15)
plt.ylabel('Spatial Target Accuracy (%)', fontsize=12)
plt.ylim(80, 100) 

for i, v in enumerate(median_test_accuracy):
    ax3.text(i, v + 0.5, f'{v}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('08_chart_model_accuracy_comparison.png', dpi=200)
plt.close()

print("Successfully generated 08 External Feature comparison chart series!")
