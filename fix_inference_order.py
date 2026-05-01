import os

old_block = """> [!WARNING]
> ### 🚨 **FINAL INFERENCE: WHAT STANDS OUT? WHICH MODEL WORKS BEST AND WHY?** 🚨
> 
> When looking at the 30-combination matrix below, **LightGBM** wins 23 out of the 30 combinations. Random Forest occasionally beats it when Macro features (like Trends or Rates) flood the system with noise, because LightGBM gets confused trying to bin the macro-data, whereas Random Forest just forcefully averages it out. 
>
> **The Ultimate Takeaway**: The absolutely best model across all 90 runs is **Random Forest on Track 08P (OSM + News + Trends)** achieving an error of only £537,786. By removing the Lat/Lon coordinates (which causes severe spatial overfitting) and removing National Interest Rates (which causes complete dataset collinearity), the Random Forest beautifully balanced local infrastructure distance with global sentiment and demand!"""

new_block = """> [!WARNING]
> ### **FINAL INFERENCE: WHAT STANDS OUT? WHICH MODEL WORKS BEST AND WHY?**
> 
> When looking at the 30-combination matrix above, **LightGBM** wins 23 out of the 30 combinations. Random Forest occasionally beats it when Macro features (like Trends or Rates) flood the system with noise, because LightGBM gets confused trying to bin the macro-data, whereas Random Forest just forcefully averages it out. 
>
> **The Ultimate Takeaway**: The absolutely best model across all 90 runs is **Random Forest on Track 08P (OSM + News + Trends)** achieving an error of only £537,786. By removing the Lat/Lon coordinates (which causes severe spatial overfitting) and removing National Interest Rates (which causes complete dataset collinearity), the Random Forest beautifully balanced local infrastructure distance with global sentiment and demand!"""

# Process README.md
with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

# We need to split around the table
if old_block in text:
    # Find the table end
    # The table ends before "## Cloud Deployment"
    table_start_idx = text.find('### The Ultimate Phase 08 Combinatorial Inference Table')
    if table_start_idx != -1:
        table_end_idx = text.find('\n## Cloud', table_start_idx)
        if table_end_idx != -1:
            table_content = text[table_start_idx:table_end_idx]
            
            # The structure was: old_block \n\n table_content
            # We want: table_content \n\n new_block
            
            # Remove old block completely
            text = text.replace(old_block + "\n\n", "")
            
            # Now insert the new block after the table content
            text = text.replace(table_content, table_content + "\n\n" + new_block)
            
            with open('README.md', 'w', encoding='utf-8') as f:
                f.write(text)
            print("Successfully updated README.md")

# Process Walkthrough.md
with open('Walkthrough.md', 'r', encoding='utf-8') as f:
    text = f.read()

if old_block in text:
    table_start_idx = text.find('### The Ultimate Phase 08 Combinatorial Inference Table')
    if table_start_idx != -1:
        table_end_idx = text.find('\n---', table_start_idx)
        if table_end_idx != -1:
            table_content = text[table_start_idx:table_end_idx]
            
            text = text.replace(old_block + "\n\n", "")
            text = text.replace(table_content, table_content + "\n\n" + new_block)
            
            with open('Walkthrough.md', 'w', encoding='utf-8') as f:
                f.write(text)
            print("Successfully updated Walkthrough.md")
