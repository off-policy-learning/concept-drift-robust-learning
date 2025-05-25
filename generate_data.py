"""
Run this file to generate training and testing datasets for LN.
Original dataset published by Behaghel et al., 2014.
The two-action example, behaghel.csv, utlized here is by Kallus, 2023.
"""
from utils import DataInput, set_global_seeds, subarray_datacls
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

# =======================================================================================================================
# # Process the unemployment dataset
# =======================================================================================================================

df = pd.read_csv(f'unemployment_data/behaghel.csv') # read csv

Xbin_df = df[['College_education',
  'nivetude2',
  'Vocational',
  'High_school_dropout',
  'Manager',
  'Technician',
  'Skilled_clerical_worker',
  'Unskilled_clerical_worker',
  'Skilled_blue_colar',
  'Unskilled_blue_colar',
  'Woman',
  'Married',
  'French',
  'African',
  'Other_Nationality',
  'Paris_region',
  'North',
  'Other_regions',
  'Employment_component_level_1',
  'Employment_component_level_2',
  'Employment_component_missing',
  'Economic_Layoff',
  'Personnal_Layoff',
  'End_of_Fixed_Term_Contract',
  'End_of_Temporary_Work',
  'Other_reasons_of_unemployment',
  'Statistical_risk_level_2',
  'Statistical_risk_level_3',
  'Other_Statistical_risk',
  'Search_for_a_full_time_position',
  'Sensitive_suburban_area',
  'Insertion',
  'Interim',
  'Conseil']]


Xnum_df = df[['age',  
  'Number_of_children', 
  'exper',
  'salaire.num',
  'mois_saisie_occ',
  'ndem'
  ]]

Xall_df = pd.concat([Xbin_df, Xnum_df], axis=1)
Xall = Xall_df.to_numpy()

## PCA on Xall
X_normalized = StandardScaler().fit_transform(Xall) # standardize the data
pca = PCA(n_components=5) # apply PCA to reduce to 5 components
Xpca = pca.fit_transform(X_normalized)
Xpca_df = pd.DataFrame(Xpca)

## regression on outome Y(a)|x and propensity score
Y_df = df['Y'] # retrieve outcome data
pi0_df, pi1_df = df['A_public'], df['A_private'] # retrieve action data

pi0, pi1 = np.mean(pi0_df.to_numpy()), np.mean(pi1_df.to_numpy()) # compute constant propensity score

Y0X_df = Xpca_df[df['A_public'] == 1] # regression on Y(0)|x
Y0_df = Y_df[df['A_public'] == 1]
reg_Y0 = RandomForestRegressor().fit(Y0X_df.to_numpy(), Y0_df.to_numpy())

Y1X_df = Xpca_df[df['A_private'] == 1] # regression on Y(1)|x
Y1_df = Y_df[df['A_private'] == 1]
reg_Y1 = RandomForestRegressor().fit(Y1X_df.to_numpy(), Y1_df.to_numpy())
Y1_all = reg_Y1.predict(Xpca_df.to_numpy())

# =======================================================================================================================
# # Training and testing dataset generation functions
# =======================================================================================================================

n = Xpca_df.shape[0]

def generate_data():
    """
    Shuffle the processed dataset according to the set global seed,
    and randomize the action A and the outcome Y0, Y1, according to the fitted regression models.
    """
    order = np.arange(n)
    np.random.shuffle(order) # shuffle processed dataset

    s = Xpca_df.iloc[order].reset_index(drop=True).to_numpy()
    a = np.random.binomial(1, pi1, size=(n,))

    Y0 = reg_Y0.predict(s).reshape((n,1)) # fit expected outcome E[Y(0)]
    Y1 = reg_Y1.predict(s).reshape((n,1)) # fit expected outcome E[Y(1)]
    exp_reward_mat = np.concatenate((Y0,Y1), axis=1)
    reward_mat = np.random.binomial(1, exp_reward_mat, size=(n,2)) # generate randomized outcomes Y(0), Y(1)

    r = reward_mat[range(n), a]

    a_prob = np.concatenate(((np.array([pi0]*n)).reshape((n,1)), (np.array([pi1]*n)).reshape((n,1))), axis=1)

    return DataInput(
        s=s,
        a=a,
        r=r,
        a_prob=a_prob,
        reward_mat=reward_mat
    )

def save_dataset(
    data: DataInput,
    seed: int,
    datasize_list: list,
    test_datasize: int,
    num_actions: int,
):
    """
    Truncate dataset according to the training and testing dataset sizes.
    Save dataset to designated file path.
    """
    num_actions = data.a.max() + 1
    unique_a = np.unique(data.a)
    assert unique_a.shape[0] == num_actions
    for i in range(num_actions):
        assert unique_a[i] == i

    data_dir = f'data/{seed}/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    assert datasize_list[-1] + test_datasize <= data.s.shape[0]
    for datasize in datasize_list:
        data_file_path = f'data/{seed}/{datasize}.pkl'
        with open(data_file_path, 'wb') as f:
            sub_data = subarray_datacls(data, datasize)
            pickle.dump(sub_data, f)

    test_data_path = f'data/{seed}/test.pkl'
    with open(test_data_path, 'wb') as f:
        test_data = subarray_datacls(data, test_datasize, from_back=True)
        pickle.dump(test_data, f)

# =======================================================================================================================
# # Generate training and testing dataset over given number of seeds
# =======================================================================================================================

if __name__ == '__main__':

    num_seeds = int(sys.argv[1]) # randomize dataset over num_seeds seeds
    total_datapoints = 33000
    datasize_list = [
        10000,
        12500,
        15000,
        17500,
        20000
    ]
    test_datasize = 13000

    num_actions = 2

    for seed in range(num_seeds):
        set_global_seeds(seed)
        data = generate_data()
        save_dataset(
            data=data,
            seed=seed,
            datasize_list=datasize_list,
            test_datasize=test_datasize,
            num_actions=num_actions,
        )