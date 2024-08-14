#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
from pathlib import Path
import pandas as pd

project_path = Path(os.getcwd())
sys.path.append(str(project_path) +'/')

data_path =  project_path/'PPMI_Data'/'freesurfer_v6'/'aseg_lh_rh_vol_ct_freesurferFlipped_v6_PPMI_noDups_age_sex_gene_snps_prs_dropUnnecessaryColumns_normalizedVols_reorderUKBB_normalized01ALL.csv'#aseg_vol_prs_nonRegress_genesRecoded_PPMI_droppedUnnecessaryColumns_normalized_imputed.csv'#aseg_vol_rh_lh_aparc_prs_genesRecoded_PPMI_droppedUnnecessaryColumns_imputed.csv' #'PPMI_data'/'aseg_vol_prs_genesRecoded_PPMI_droppedUnnecessaryColumns.csv'# prs-pd-counterfactual/PPMI_data/aseg_vol_rh_lh_aparc_prs_genesRecoded_PPMI_droppedUnnecessaryColumns_imputed.csv #'aseg_vol_rh_lh_aparc_prs_PPMI_droppedUnnecessaryColumns_imputed.csv' #project_path/'PPMI_data'/'aseg_vol_prs_PPMI_droppedUnnecessaryColumns.csv' # #project_path/'UKBB_data_extraction'/'ukbb_CMR_PD.csv'


# In[2]:


from utils.datasets import PRSDataframe

dataset = PRSDataframe(data_path)


# In[ ]:


import torch
from  torch.utils.data import DataLoader,random_split
import math
import numpy as np
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pickle
from automacaw_mmnist import automacaw_mmnist


# In[4]:


features = 172 #174 #119 #48 #189 # NUM OF FEATURES #ukbb = 140 #PPMI = 106 PPMI only vol = 45 #PPMI freesurfer vols = 49, vols_ct_sa = 188


# In[5]:


batch_size = 64
m =len(dataset)
print(m)
train_data, val_data = random_split(dataset, [math.floor(m-m*0.2), math.ceil(m*0.2)], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)


# In[6]:


test = next(iter(train_data))
print(test)
print(test.shape)


# In[7]:


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ## Train Flow

# In[8]:


from automacaw_prs_pd_snps import automacaw_prs_pd

automacaw_model = automacaw_prs_pd(encoded_dim=features)
losses = {'nll_train_loss':[],'nll_val_loss':[]}

automacaw_model.macaw = automacaw_model._init_macaw()


# In[9]:


early_stopper = EarlyStopper(patience=10, min_delta=0.1)

for epoch in (pbar := tqdm(range(100))):
    nll_train_loss = automacaw_model.train_macaw(train_loader, lr=0.001)
    nll_val_loss = automacaw_model.test_likelihood(val_loader)
    
    losses['nll_train_loss'].append(nll_train_loss)
    losses['nll_val_loss'].append(nll_val_loss)
    
#     clear_output()
#     fig = cf_test(automacaw_model,val_loader)
    
#     print(f"nll_train: {nll_train_loss:.3f}, nll_val: {losses['nll_val_loss'][-1]:.3f}")
#     display(fig)
    
    pbar.set_description(f"nll_train: {nll_train_loss:.3f}, nll_val: {losses['nll_val_loss'][-1]:.3f}")
    
    
    if early_stopper.early_stop(nll_val_loss):             
        break    
        
#     clear_output()
#     fig = automacaw_model.plot_ae_outputs(val_loader)
    


# In[10]:


plt.plot(losses['nll_train_loss'],color='b')
plt.plot(losses['nll_val_loss'],color='y')


# In[11]:


torch.save(automacaw_model,"/media/gdagasso/TOSHIBA EXT/prs-pd-counterfactual/PPMI_data/automacaw_model_age_sex_snps_freesurfer_v6_ukbb_vols_aparc_ct_vols.pt")


# In[42]:


# automacaw_model1 = torch.load("/media/gdagasso/TOSHIBA EXT/prs-pd-counterfactual/PPMI_data/automacaw_model_age_sex_cohort_freesurferVols.pt")


# ## Counterfactuals

# In[158]:


val_set = []

# Iterate over the validation loader
for x_obs in val_loader:
   val_set.append(x_obs)


val_set = torch.cat(val_set, dim=0)

print(f'Validation set size: {val_set.size()}')


# In[164]:


snp1 = 16# 28
snp2 = 4


# In[165]:


true1 = 1
true2 = 1

cf1 = 0
cf2 = 0


# In[166]:



print(val_set[:,snp1])

#filtered_rows_for_cf


# In[152]:


x_obs = val_set
m = automacaw_model.feature_mean.detach().cpu().numpy()
s = automacaw_model.feature_std.detach().cpu().numpy()

filtered_rows_for_cf = x_obs[ (x_obs[:,snp1] == true1 ) ]#& (x_obs[:,snp2] == true2)] # > 1 (x_obs[:, 1] == 1)] &
x_cf = automacaw_model.counterfactual(filtered_rows_for_cf, {snp1:cf1})#,snp2:cf2}) # -2 # min age == 31 so have to subtract from cf
#x_cf = automacaw_model.counterfactual(x_obs, {1:0})
features = x_cf[:,47:] * s + m
x_obs = x_obs.detach().cpu().numpy()


# In[135]:


print(x_obs.shape)
print(features.shape)


# In[65]:


# average analyses


# In[136]:


# filtered_rows = x_obs[x_obs[:, snp] == 0]#& (x_obs[:,1] == 1)] # > 1
filtered_rows = x_obs[(x_obs[:,snp1] == true1 ) ]#& (x_obs[:,snp2] == true2)]
print(filtered_rows)


# In[137]:


averages = np.mean(filtered_rows[:, 47:], axis=0) #-1


# In[138]:


averages.shape


# In[139]:



print(x_cf[:,snp1])
print(x_obs[:,snp1])


# In[140]:


#x_obs are true vals and features are cfs
plt.bar(np.arange(172), averages, color='b', alpha=0.5) # 139 # all xx people averaged
plt.bar(np.arange(172), (np.mean(features[:, 0:], axis=0)), color='r', alpha=0.5) #[:, 1:] # changed all xx people to yy
plt.savefig("LRRK2_2_1.svg")
# so would expect sizes to increase when making healthy, right??


# In[141]:


plt.plot((averages) - (np.mean(features[:, 0:], axis=0))) # [:, 1:] # averages should be smaller then features so should be negative values


# In[142]:


diff = (averages) - (np.mean(features[:, 0:], axis=0)) # [:, 1:]
vals = np.argwhere(abs(diff) > 0.00)
#vals = np.argwhere(diff > 500)


# In[143]:


dataset.df.columns


# In[144]:


print(dataset.df.columns[124]) #124


# In[145]:


for val in vals:
    print(val, diff[val])
    print(dataset.df.columns[val+124]) #4 UKBB, 25 PPMI


# In[ ]:


# T-Tests


# In[76]:


import pandas as pd
from scipy.stats import ttest_ind


# In[146]:


column_names = dataset.df.columns[124:].values.tolist()
print(column_names)


# In[147]:


print(filtered_rows[:, 47:].shape)
print(features.shape)


# In[148]:


true = pd.DataFrame(filtered_rows[:, 47:], columns=column_names)
cfs = pd.DataFrame(features[:,0:], columns=column_names)
print(cfs.shape)
print(true.shape)


# In[80]:


# true.to_csv("/media/gdagasso/TOSHIBA EXT/prs-pd-counterfactual/CF_dataframes/trueValues_g2019s_0_to_1_exp_flipped_aseg_ct_vol.csv", sep=',')
# cfs.to_csv("/media/gdagasso/TOSHIBA EXT/prs-pd-counterfactual/CF_dataframes/cfValues_g2019s_0_to_1_exp_flipped_aseg_ct_vol.csv", sep=',')


# In[149]:




true_file_path = f"/media/gdagasso/TOSHIBA EXT/prs-pd-counterfactual/CF_dataframes/ppmi_testset/snps/trueValues_E365K_rs2230288_{true1}_{cf1}_freesurfer_v6_ukbbTestSet_vols_aseg_ct_vol.csv"
cf_file_path = f"/media/gdagasso/TOSHIBA EXT/prs-pd-counterfactual/CF_dataframes/ppmi_testset/snps/cfValues_E365K_rs2230288_{true1}_{cf1}_freesurfer_v6_ukbbTestSet_vols_aseg_ct_vol.csv"

# Save the DataFrames to the new file paths
true.to_csv(true_file_path, sep=",")
cfs.to_csv(cf_file_path, sep=",")


# In[113]:




true_file_path = f"/media/gdagasso/TOSHIBA EXT/prs-pd-counterfactual/CF_dataframes/ppmi_testset/snps/trueValues_E365K_rs2230288_{true1}_{cf1}_rs356181_{true2}_{cf2}_freesurfer_v6_ukbbTestSet_vols_aseg_ct_vol.csv"
cf_file_path = f"/media/gdagasso/TOSHIBA EXT/prs-pd-counterfactual/CF_dataframes/ppmi_testset/snps/cfValues_E365K_rs2230288_{true1}_{cf1}_rs356181_{true2}_{cf2}_freesurfer_v6_ukbbTestSet_vols_aseg_ct_vol.csv"

# Save the DataFrames to the new file paths
true.to_csv(true_file_path, sep=",")
cfs.to_csv(cf_file_path, sep=",")


# In[ ]:





# In[173]:


snp1 = 16
snp2 = 4

true1 = 0
true2 = 0

cf1 = 2
cf2 = 1

age_index = 0


# In[179]:


print(val_set[:,0])


# In[180]:


import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

num_bins = 8  # For example, divide into 8 bins
age_min = df['age'].min()
age_max = df['age'].max()

# Using pandas.cut to create age bins automatically
df['age_bin'] = pd.cut(df['age'], bins=num_bins, labels=False)  # bins will be created automatically

# Alternatively, define custom bins explicitly with floating-point precision
bin_edges = np.linspace(age_min, age_max, num_bins + 1)  # Creates bins with equal intervals
bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(num_bins)]

# Use these custom bins
df['age_bin'] = pd.cut(df['age'], bins=bin_edges, labels=bin_labels, right=False)


# Initialize a list to store results
all_counterfactuals = []

for start_age in age_range:
    end_age = start_age + 4  # Define the end of the bin

    # Filter observations for the current age bin
    age_bin_rows = x_obs[(x_obs[:, age_index] >= start_age) & (x_obs[:, age_index] <= end_age) & (x_obs[:, snp1] == true1)]
    
    print(f"Age bin {start_age}-{end_age}: {age_bin_rows.shape[0]} rows")

    if age_bin_rows.shape[0] == 0:
        print("hi")
        continue  # Skip if no observations for this age bin
    
    # Generate counterfactuals for the filtered rows
    counterfactuals = automacaw_model.counterfactual(age_bin_rows, {snp1: cf1}) # , snp2: cf2})
    
    # De-normalize the counterfactuals
    features = counterfactuals[:, 47:] * s + m
    
    # Store the results
    all_counterfactuals.append(features)

# Combine all counterfactuals to form the joint distribution
joint_distribution = np.vstack(all_counterfactuals)

# Calculate and print summary statistics
mean_values = np.mean(joint_distribution, axis=0)
std_values = np.std(joint_distribution, axis=0)

print("Mean of joint distribution across age bins:", mean_values)
print("Standard deviation of joint distribution across age bins:", std_values)

# Visualization
# Example: Plotting the distribution of a specific feature (e.g., Brain IDP 1)
feature_index = 0  # Adjust the index based on the feature of interest

sns.histplot(joint_distribution[:, feature_index], kde=True)
plt.title("Distribution of Brain IDP 1 across age bins with SNP risk alleles")
plt.xlabel("Brain IDP 1")
plt.ylabel("Frequency")
plt.show()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


columns_true = range(1, len(true.columns))
columns_cfs = range(1, len(cfs.columns))


# In[37]:


print(columns_true)


# In[38]:


# Perform t-tests for each column
t_test_results = {}
for col_true, col_cfs in zip(columns_true, columns_cfs):
    t_stat, p_value = ttest_ind(true.iloc[:, col_true], cfs.iloc[:, col_cfs])
    t_test_results[(col_true, col_cfs)] = {'t_statistic': t_stat, 'p_value': p_value}


# In[39]:


# Print the results with column names for p-values < 0.05
for cols, result in t_test_results.items():
    col_name_true = true.columns[ cols[0]]
    col_name_cfs = cfs.columns[ cols[1]]
    if result['p_value'] < 0.1:
        print(f"T-test for columns '{col_name_true}' and '{col_name_cfs}': p-value = {result['p_value']}")


# 
# ## MANOVA

# In[73]:


import pandas as pd
import numpy as np
import statsmodels.api as sm


# In[74]:


true['Group'] = 'Original'
cfs['Group'] = 'Counterfactual'
combined_df = pd.concat([true, cfs], ignore_index=True)


# In[75]:


X = combined_df.drop(columns=['Group'])


# In[76]:


X.head()
X.shape


# In[77]:


group_labels = np.where(combined_df['Group'] == 'Original', 0, 1)


# In[78]:


manova_model = sm.multivariate.MANOVA(X, group_labels)
manova_results = manova_model.mv_test()

print(manova_results.summary())


# ## MANCOVA

# In[47]:


true = pd.DataFrame(filtered_rows[:, 47:], columns=column_names)
cfs = pd.DataFrame(features[:,0:], columns=column_names)
true['Group'] = 'Original'
cfs['Group'] = 'Counterfactual'

covs = pd.DataFrame(filtered_rows[:,0:2])

covs.columns = ['Age','Sex']

true_covs = pd.concat([true, covs], axis=1)
print(true_covs.head)
cfs_covs = pd.concat([cfs, covs], axis=1)
print(cfs_covs.head)

combined_df = pd.concat([true_covs, cfs_covs], ignore_index=True)

print(cfs.shape)
print(true.shape)

#combined_df = pd.concat([combined_df, covs], axis=1)


# In[48]:


X = combined_df.drop(columns=['Group'])


# In[ ]:





# In[49]:


group_labels = np.where(combined_df['Group'] == 'Original', 0, 1)

# Perform MANCOVA
mancova_model = sm.multivariate.MANOVA.from_formula('X ~ Age + Sex + Group', data=combined_df)
mancova_results = mancova_model.mv_test()


# In[50]:


print(mancova_results.summary())


# ## PAIRED ANOVA

# In[70]:


from scipy.stats import f_oneway

# Assuming you have two DataFrames: original_df and counterfactual_df
# Each DataFrame contains the same participants with original and counterfactual values for the same features.
significant_features = []

# Iterate over each feature
for feature in true.columns:
    # Extract original and counterfactual values for the current feature
    original_values = true[feature]
    counterfactual_values = cfs[feature]
    
    # Perform paired ANOVA for the current feature
    f_statistic, p_value = f_oneway(original_values, counterfactual_values)
    if p_value < 0.05:
        significant_features.append(feature)
    
    # Print the results
    print(f"Paired ANOVA results for feature '{feature}':")
    print(f"F-statistic: {f_statistic}")
    print(f"P-value: {p_value}\n")

if significant_features:
    print("Significant features with p-value < 0.05:")
    for feature in significant_features:
        print(feature)
else:
    print("No significant features with p-value < 0.05 found.")


# In[ ]:





# In[ ]:





# # Single person analyses

# In[25]:


print(x_obs[:,0])
print(x_obs[:,1])
print(x_obs[:,2])
print(x_obs[:,3])


# In[48]:


print(filtered_rows_for_cf[:,0])
print(filtered_rows_for_cf[:,1])
print(filtered_rows_for_cf[:,2])
print(filtered_rows_for_cf[:,3])


# In[52]:


partipant = 3

print(filtered_rows_for_cf[partipant,2])
print(x_cf[partipant,2])
#print(x_obs.shape)
#features.shape


# In[53]:


#x_obs are true vals and features are cfs
plt.bar(np.arange(48), x_obs[partipant,4:-1], color='b', alpha=0.5,label='x_obs')
plt.bar(np.arange(48), features[0,0:-1], color='r', alpha=0.5,label='cf') #features[:, 0:-1]
plt.legend()


# In[54]:


plt.plot(x_obs[partipant,4:-1] -features[partipant,0:-1])


# In[55]:


np.argmax(x_obs[partipant,3:] -features[partipant,:])


# In[56]:


dataset.df.columns[126]


# In[57]:


diff = x_obs[partipant,4:-1] -features[partipant,0:-1] # [:, 1:]
vals = np.argwhere(abs(diff) > 5)
#vals = np.argwhere(diff > 0)


# In[58]:


for val in vals:
    print(val, diff[val])
    print(dataset.df.columns[val+126]) #4 UKBB, 25 PPMI

