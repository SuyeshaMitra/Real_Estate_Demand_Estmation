import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_theme(style="whitegrid", palette="muted")

# Setup the x-axis visual labels defining the specific AI model categories
models = ['Random Forest', 'XGBoost', 'LightGBM']

# Step 04 Baseline Errors
mae_04 = [424476, 410339, 401075]

# Step 06 Errors (After injecting Trends/Interest Rates mapping locally)
mae_06 = [426163, 412490, 401553] 

# Establish format
fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(models))
width = 0.35

# Plot side by side performance geometries
bars1 = ax.bar([i - width/2 for i in x], mae_04, width, label='04 Baseline (Geo Only)', color='#FF9999')
bars2 = ax.bar([i + width/2 for i in x], mae_06, width, label='06 Enhanced (Macro + Geo)', color='#66B2FF')

# Add descriptive structural tracking texts
ax.set_title('Spatial Extrapolation Limit Test\n(Did External Variables Break the Ceiling?)', fontsize=14, pad=15)
ax.set_ylabel('Mean Absolute Error (£)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ensure absolute math labels print explicitly
for i, v in enumerate(mae_04):
    ax.text(i - width/2, v + 2000, f'£{v:,.0f}', ha='center', fontweight='bold', fontsize=9, color='red')
    
for i, v in enumerate(mae_06):
    ax.text(i + width/2, v + 2000, f'£{v:,.0f}', ha='center', fontweight='bold', fontsize=9, color='blue')

# Fix arbitrary cutoff edge collisions
plt.ylim(350000, 450000)
plt.tight_layout()
# Render internal metric mapping explicitly locally
plt.savefig('07_chart_feature_impact_comparison.png', dpi=200)
plt.close()

print("Successfully generated 06 Extrapolation Break visualization.")
