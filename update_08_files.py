import os
import glob

combinations_map = {
    "08A_LatLon_OSM_News_Trends.py": "(Latitude and Longitude) + OSM + Google News + Google Trends",
    "08B_LatLon_OSM_News_Rates.py": "(Latitude and Longitude) + OSM + Google News + Rates World Bank",
    "08C_LatLon_OSM_Trends_Rates.py": "(Latitude and Longitude) + OSM + Google Trends + Rates World Bank",
    "08D_LatLon_News_Trends_Rates.py": "(Latitude and Longitude) + Google News + Google Trends + Rates World Bank",
    "08E_LatLon_OSM_News.py": "(Latitude and Longitude) + OSM + Google News",
    "08F_LatLon_OSM_Trends.py": "(Latitude and Longitude) + OSM + Google Trends",
    "08G_LatLon_OSM_Rates.py": "(Latitude and Longitude) + OSM + Rates World Bank",
    "08H_LatLon_News_Trends.py": "(Latitude and Longitude) + Google News + Google Trends",
    "08I_LatLon_News_Rates.py": "(Latitude and Longitude) + Google News + Rates World Bank",
    "08J_LatLon_Trends_Rates.py": "(Latitude and Longitude) + Google Trends + Rates World Bank",
    "08K_LatLon_OSM.py": "(Latitude and Longitude) + OSM",
    "08L_LatLon_News.py": "(Latitude and Longitude) + Google News",
    "08M_LatLon_Trends.py": "(Latitude and Longitude) + Google Trends",
    "08N_LatLon_Rates.py": "(Latitude and Longitude) + Rates World Bank",
    "08O_OSM_News_Trends_Rates.py": "OSM + Google News + Google Trends + Rates World Bank",
    "08P_OSM_News_Trends.py": "OSM + Google News + Google Trends",
    "08Q_OSM_News_Rates.py": "OSM + Google News + Rates World Bank",
    "08R_OSM_Trends_Rates.py": "OSM + Google Trends + Rates World Bank",
    "08S_OSM_News.py": "OSM + Google News",
    "08T_OSM_Trends.py": "OSM + Google Trends",
    "08U_OSM_Rates.py": "OSM + Rates World Bank",
    "08V_News_Trends_Rates.py": "Google News + Google Trends + Rates World Bank",
    "08W_News_Trends.py": "Google News + Google Trends",
    "08X_News_Rates.py": "Google News + Rates World Bank",
    "08Y_Trends_Rates.py": "Google Trends + Rates World Bank",
    "08Z_OSM_Only.py": "OSM",
    "08AA_News_Only.py": "Google News",
    "08AB_Trends_Only.py": "Google Trends",
    "08AC_Rates_Only.py": "Rates World Bank",
    "08AD_LatLon_Only.py": "(Latitude and Longitude)"
}

print("Injecting Docstrings into Python Files...")
walkthrough_list = "#### C) Explicit Python File Combinatorial Breakdown\nHere is an exact novice-level explanation of every single execution file generated in Phase 08. Every single script below mathematically races Random Forest, XGBoost, and LightGBM against each other.\n\n"

for filename, proper_name in combinations_map.items():
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        docstring = f'''"""
Step 8: Combinatorial ML Execution Track
========================================
Target Combination: {proper_name}
Models Executed: Random Forest, XGBoost, LightGBM

Explanation:
This script isolates the dataset to purely train on the specific features listed above. 
It loads the data (Step 1), temporally splits it (Step 2), filters the target columns (Step 3),
and then mathematically races Random Forest, XGBoost, and LightGBM models against each other (Step 4) 
to determine which algorithm is the most stable under this specific data context.
"""
'''
        if "Step 8: Combinatorial ML Execution Track" not in content:
            new_content = docstring + "\n" + content
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
    # Build the walkthrough list
    walkthrough_list += f"*   **`{filename}`**: Runs the 3 Machine Learning models exclusively on `{proper_name}`. The goal of this isolated script is to determine exactly how accurate the algorithms are when they are artificially forced to rely ONLY on these specific variables, helping us detect if any of these features cause noise or overfitting.\n"

print("Docstrings injected.")

# Now inject the walkthrough list into Walkthrough.md
print("Updating Walkthrough.md...")
with open('Walkthrough.md', 'r', encoding='utf-8', errors='ignore') as f:
    wt = f.read()

# Replace the old summary block with the explicit list
old_block = """*   `08A` through `08N` test specific variations explicitly utilizing the physical Geographic coordinates (`Latitude`/`Longitude`).
*   `08O` through `08AB` test combinations where the baseline pure geography is completely erased, and the model attempts to survive entirely on OSM Proximity and Macroeconomic vectors.
*   `08AC` and `08AD` physically isolate a single feature (e.g. ONLY World Bank Rates, ONLY Lat/Lon).
*   `08XD_geospatial_Neural_Network.py` is the specific Neural Network baseline file, executing Multi-Layer Perceptrons on the spatial matrix to conclusively prove its failure threshold."""

if old_block in wt:
    wt = wt.replace(old_block, walkthrough_list)
    with open('Walkthrough.md', 'w', encoding='utf-8') as f:
        f.write(wt)
    print("Walkthrough.md successfully updated!")
else:
    print("Could not find the old block to replace in Walkthrough.md")
