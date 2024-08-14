import pandas as pd
from neuroCombat import neuroCombat

# Step 1: Read the CSV Files
ppmi = pd.read_csv("PPMI.csv")
ukbb = pd.read_csv("UKBB.csv")

# Step 2: Identify Imaging Data Columns
# Assuming imaging data starts from the 5th column (index 4). Adjust if needed.
imaging_start_col = 5
imaging_columns = ppmi.columns[imaging_start_col:]

# Step 3: Subset the DataFrames for Harmonization
ppmi_subset = ppmi[["subjectkey", "age", "sex"] + list(imaging_columns)]
ukbb_subset = ukbb[["subjectkey", "age", "sex"] + list(imaging_columns)]

# Add a 'batch' column to distinguish between datasets
ppmi_subset['batch'] = 'PPMI'
ukbb_subset['batch'] = 'UKBB'

# Combine the subsets
combined = pd.concat([ppmi_subset, ukbb_subset], ignore_index=True)

# Step 4: Prepare Covariates
covars = combined[["sex", "age", "batch"]]
covars['sex'] = covars['sex'].replace({'M': 1, 'F': 0})  # Encode Sex as numerical

# Extract and transpose imaging data
imaging_data = combined[imaging_columns]
data_T = imaging_data.T
data_T_np = data_T.to_numpy()

# Step 5: Apply neuroCombat
batch_col = "batch"
categorical_cols = ["sex"]

data_combat = neuroCombat(dat=data_T_np, 
                          covars=covars, 
                          batch_col=batch_col, 
                          categorical_cols=categorical_cols)["data"]

# Convert back to DataFrame
harmonized_data = pd.DataFrame(data_combat, index=imaging_columns, columns=combined.index).T

# Step 6: Split the Harmonized Data Back into PPMI and UKBB
ppmi_harmonized = harmonized_data[combined['batch'] == 'PPMI']
ukbb_harmonized = harmonized_data[combined['batch'] == 'UKBB']

# Reset the index to match the original DataFrames
ppmi_harmonized.index = ppmi.index
ukbb_harmonized.index = ukbb.index

# Step 7: Replace Original Imaging Data with Harmonized Data
ppmi.loc[:, imaging_columns] = ppmi_harmonized.values
ukbb.loc[:, imaging_columns] = ukbb_harmonized.values

# Save the updated DataFrames back to CSV if needed
ppmi.to_csv("PPMI_harmonized.csv", index=False)
ukbb.to_csv("UKBB_harmonized.csv", index=False)
