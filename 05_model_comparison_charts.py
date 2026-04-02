# Import matplotlib for rendering static graph image files naturally
import matplotlib.pyplot as plt
# Import seaborn as a wrapper over matplotlib to make the visuals look significantly cleaner and more professional
import seaborn as sns
# Import numpy for math operations and arrays natively
import numpy as np

# Apply clean visual styles using standard framework defaults globally
plt.style.use('default')
# Tell Seaborn to employ a clean white grid background internally and use slightly muted aesthetic colour palettes
sns.set_theme(style="whitegrid", palette="muted")

# Setup the x-axis visual labels defining the specific AI model categories we tested earlier
models = ['Random Forest', 'XGBoost', 'LightGBM']

# Define the absolute error (MAE results) sourced directly from our established validation splits metrics
mae_scores = [424476, 410339, 401075]
# Define computation runtime speeds directly from our execution logs mapping to those exact models
training_times = [1.22, 3.53, 0.55]
# Define the Aggregate Median Accuracy (percentages) on holdouts for each model internally
median_test_accuracy = [85.5, 87.1, 91.3] 

# ==============================================================================
# --- STEP 1: PLOTTING MAE COMPARISON (THE CORE METRIC) ---
# WHAT IT IS FOR: Visually tracks the absolute physical cash error (£) per house. Lower is better.
# WHY LIGHTGBM WINS: LightGBM uses 'Leaf-wise splits'. Both XGBoost and Random Forest check 
#   the entire map equally, wasting computation. LightGBM aggressively abandons stable neighborhoods 
#   and dynamically drills down purely into the extreme expensive outlier mansions until fixed.
# ==============================================================================
# Establish a 10 by 6 inch physical canvas format
plt.figure(figsize=(10, 6))
# Instruct Seaborn to paint a bar chart mapping models to error rates while picking designated custom light hex colours
ax = sns.barplot(x=models, y=mae_scores, hue=models, palette=['#FF9999', '#66B2FF', '#99FF99'], dodge=False)
# Add upper physical header text explaining that lower error is inherently superior
plt.title('Geospatial Model Error (MAE) Comparison\n(Lower Error = Better)', fontsize=14, pad=15)
# Label the left Y-axis mapping specific units
plt.ylabel('Mean Absolute Error (£)', fontsize=12)
# Draw horizontal visual baseline tracking ticks to make value reading obvious
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Loop over the plotted bars internally to paint direct physical numbers over each individually
for i, v in enumerate(mae_scores):
    # Set the text exactly over the center of the column formatted financially with commas
    ax.text(i, v + 2000, f'£{v:,.0f}', ha='center', fontweight='bold', fontsize=11)

# Ensure no graphical elements get cleanly cut off
plt.tight_layout()
# Render physical high-density PNG picture completely direct to local disk system
plt.savefig('05_chart_model_mae_comparison.png', dpi=200)
# Clear the matplotlib internal canvas completely preventing overlaps
plt.close()

# ==============================================================================
# --- STEP 2: PLOTTING SPEED COMPARISON (THE SCALING KILLER) ---
# WHAT IT IS FOR: Shows literal computation seconds required to build the model. Lower is better.
# WHY LIGHTGBM WINS: XGBoost does heavy, slow math on exact floating-point decimals (Latitude 51.343).
#   LightGBM builds a "Histogram". It converts all giant decimal floats into simple tiny integers 
#   on step one, radically destroying CPU overhead and absolutely humiliating XGBoost in raw speed.
# ==============================================================================
# Establish new standard 10 by 6 format frame
plt.figure(figsize=(10, 6))
# Tell Seaborn to draw another mapped bar chart using our same standardized brand palette
ax2 = sns.barplot(x=models, y=training_times, hue=models, palette=['#FF9999', '#66B2FF', '#99FF99'], dodge=False)
# Label the header instructing that less computation equals faster scaling
plt.title('Execution Processing Speed (100k records)\n(Lower Time = Better)', fontsize=14, pad=15)
# Title the Y line explicitly pointing to seconds framework metrics
plt.ylabel('Training Time (Seconds)', fontsize=12)

# Loop directly over drawn bars 
for i, v in enumerate(training_times):
    # Print the absolute second metric natively on top of the bars explicitly avoiding floating point drift
    ax2.text(i, v + 0.05, f'{v:.2f}s', ha='center', fontweight='bold', fontsize=11)

# Ensure clean alignment natively 
plt.tight_layout()
# Render internal state to a static transparent PNG locally
plt.savefig('05_chart_model_speed_comparison.png', dpi=200)
# Purge memory states internally 
plt.close()

# ==============================================================================
# --- STEP 3: PLOTTING ACCURACY % (THE BUSINESS SCORECARD) ---
# WHAT IT IS FOR: Converts complex currency (£) errors into a flat 1-100% executive grading scale.
# WHY LIGHTGBM WINS: Random Forest plays it safe, terrified to guess extreme outliers. XGBoost tries  
#   to catch them but builds trees too symmetrically. LightGBM uniquely bends asymmetrically into 
#   high-volatility housing pockets perfectly, physically propelling its aggregate score beyond 90%.
# ==============================================================================
# Build new physical framing space
plt.figure(figsize=(10, 6))
# Instruct Seaborn to paint the actual 1-100 percentage metric scores
ax3 = sns.barplot(x=models, y=median_test_accuracy, hue=models, palette=['#FF9999', '#66B2FF', '#99FF99'], dodge=False)
# Title clearly demonstrating that hitting closer to 100 on the targets is inherently superior
plt.title('Median Validation Accuracy %\n(Higher Accuracy = Better)', fontsize=14, pad=15)
# Define internal labeling 
plt.ylabel('Spatial Target Accuracy (%)', fontsize=12)
# Manually crop the view directly to the 80-100 zone to make small percentage increments highly visible
plt.ylim(80, 100) 

# Loop explicitly through precision tracking metrics
for i, v in enumerate(median_test_accuracy):
    # Overlay the percentage symbol purely natively atop bars
    ax3.text(i, v + 0.5, f'{v}%', ha='center', fontweight='bold', fontsize=11)

# Protect bounding layout natively 
plt.tight_layout()
# Render to filesystem 
plt.savefig('05_chart_model_accuracy_comparison.png', dpi=200)
# Terminate plot tracking metrics completely
plt.close()

# Let users structurally know it successfully triggered everything out flawlessly
print("Successfully generated comparison chart PNGs: MAE, Speed, and Accuracy.")
