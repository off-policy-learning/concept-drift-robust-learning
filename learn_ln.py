"""
Run this file to test the learning algorithm on the unemployment dataset.
"""
import numpy as np
import jax.numpy as jnp
import pandas as pd
import random
from patsy import dmatrix
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import pickle
import sys
import os

from utils import DataInput

# =======================================================================================================================
# # Learning algorithm LN parameters
# =======================================================================================================================

numpy2ri.activate() # initialize policy tree search policy class by Sverdrup et al., 2020
pt = importr('policytree')
L = 1 # tree depth

fold_number = 3 # cross-fitting fold number K

knts_num = 2 # spline degree

eps = 0.01 # nuisance parameter alpha truncation threshold

# =======================================================================================================================
# # Learning algorithm LN functions
# =======================================================================================================================

def f_con(x):
    """
    Compute KL divergence.
    """
    return np.exp(x-1)

def fc(a_x, eta_x, y):
    """
    Compute the conjugate function.
    """
    x = - (y + eta_x)/a_x
    return f_con(x)

def loss(theta, Xdat, Y, delta, eps):
    """
    Compute the empirical loss.
    """
    p = int(len(theta)/2)
    alpha = theta[0:p]
    eta = theta[p:2*p]
    a_x =  np.maximum(eps, Xdat @ alpha)
    eta_x = Xdat @ eta
    loss = np.mean(a_x * fc(a_x, eta_x, Y) + eta_x + a_x * delta)
    return(loss)

def split_data(data: dict):
    """
    Randomly split the data for cross-fitting.
    """
    full_n = len(data['action'])

    a1_idx = np.array(range(full_n))[data['action']==1]
    a2_idx = np.array(range(full_n))[data['action']==2]
    a1_n = len(a1_idx)
    a2_n = len(a2_idx)

    a1_sample_idx = random.sample(range(a1_n),a1_n)
    a2_sample_idx = random.sample(range(a2_n),a2_n)

    a1_sample_idx_list = []
    a2_sample_idx_list = []
    for k in range(fold_number): # randomly split data into K folds
        a1_start = int(np.floor(k*a1_n/fold_number))
        a2_start = int(np.floor(k*a2_n/fold_number))
        if k+1 == fold_number:
            a1_sample_idx_list.append(a1_idx[a1_sample_idx[a1_start:a1_n]])
            a2_sample_idx_list.append(a2_idx[a2_sample_idx[a2_start:a2_n]])
        else:
            a1_end = int(np.floor((k+1)*a1_n/fold_number))
            a2_end = int(np.floor((k+1)*a2_n/fold_number))
            a1_sample_idx_list.append(a1_idx[a1_sample_idx[a1_start:a1_end]])
            a2_sample_idx_list.append(a2_idx[a2_sample_idx[a2_start:a2_end]])


    return a1_sample_idx_list, a2_sample_idx_list

def r_est(input_data: dict, target_indx: np.ndarray, action: int):
    """
    Estimate constant propensity score.
    """
    dim = input_data['action'][target_indx].shape[0]
    y = np.zeros((dim,))
    A_act_indx = np.array(range(dim))[input_data['action'][target_indx]==action]
    y[A_act_indx] = 1
    return np.mean(y)

def turn_spline(X: jnp.ndarray):
    """
    Turn the covariate data into splines, according to the knts_num degrees.
    """
    Xdat = pd.DataFrame()
    for jcol in range(X.shape[1]):
        knts_quantile = [i/(knts_num+1) for i in range(1, knts_num+1)]
        knots = np.quantile(X[:,jcol], knts_quantile)
        if jcol == 0:
            Xdat = pd.concat((Xdat, dmatrix('bs(train, knots=knts, degree=deg, include_intercept=True)',
                                            {'train': X[:,jcol], 'knts': knots, 'deg': knts_num},
                                            return_type='dataframe')),
                             axis=1)
        else:
            Xdat = pd.concat((Xdat, dmatrix('bs(train, knots=knts, degree=deg, include_intercept=False)',
                                            {'train': X[:,jcol], 'knts': knots, 'deg': knts_num},
                                            return_type='dataframe')),
                             axis=1)
    return(Xdat)

def opt_ERM(X: jnp.ndarray, Y: jnp.ndarray, delta: float):
    """
    Solve for ERM: find the optimal nuisance parameters alpha and eta.
    """
    Xdat = turn_spline(X) # turn x into spline
    alpha = np.random.uniform(0.02, 0.1, size = Xdat.shape[1])
    eta = np.random.uniform(-2, 2, size = Xdat.shape[1])
    theta = np.concatenate((alpha,eta))

    opt_prob = minimize(loss, theta, method='Nelder-Mead',
                        args=(Xdat, Y, delta, eps),
                        options={'xtol':1e-4, 'maxfev':200000, 'disp':False}) # optimize for nuisance parameters

    opt_theta = opt_prob.x
    opt_alpha = opt_theta[0:len(alpha)]
    opt_eta = opt_theta[len(alpha):2*len(alpha)]

    return {'val':loss(opt_theta, Xdat, Y, delta, eps),'theta': opt_theta,'Xdat': Xdat,'alpha': opt_alpha,'eta': opt_eta}

def produce_Gamma(data: dict, delta: float):
    """
    Compute concept drift doubly robust policy value estimates.

    Output a matrix Gamma (of size n*a) of policy value estimates,
    where each entry Gamma(x,a) is the policy value estimate if the action is a, given covariate x.

    policytree later uses this policy value estimate matrix Gamma to learn an optimal policy.
    """
    a1_indx_lists, a2_indx_lists = split_data(data)
    indx_lists = [a1_indx_lists, a2_indx_lists]

    Gamma_matrix = np.zeros((n,2))

    for k in range(fold_number):

        reg_g_list = []
        for j in range(2):
            indx = np.concatenate((a1_indx_lists[(k+1)%fold_number], a2_indx_lists[(k+1)%fold_number]))
            reg_r = r_est(data, indx, j+1) # estimate propensity score 

            all_probs = []
            all_losses = []
            for rrr in range(3): # randomly initialize ERM for 3 times
                this_run = opt_ERM(data['covariate'][indx_lists[j][(k+1)%fold_number]],
                                   data['outcome'][indx_lists[j][(k+1)%fold_number]],
                                   delta)
                all_probs.append(this_run)
                all_losses.append(this_run['val'])
            opt_prob = all_probs[np.argmin(all_losses)]

            Xdat = turn_spline(data['covariate'][indx_lists[j][(k+2)%fold_number]])
            ax = np.maximum(eps, Xdat @ opt_prob['alpha'])
            eta = Xdat @ opt_prob['eta']
            Gval = ax * fc(ax, eta, data['outcome'][indx_lists[j][(k+2)%fold_number]]) + eta + ax * delta

            reg_g = RandomForestRegressor().fit(data['covariate'][indx_lists[j][(k+2)%fold_number]],
                                                Gval) # fit g function on off-folds
            reg_g_list.append(reg_g)

            adj_Xdat = turn_spline(data['covariate'][indx_lists[j][k]])
            adj_ax = np.maximum(eps, adj_Xdat @ opt_prob['alpha'])
            adj_eta = adj_Xdat @ opt_prob['eta']

            adj_Gval = adj_ax * fc(adj_ax, adj_eta, data['outcome'][indx_lists[j][k]]) + adj_eta + adj_ax * delta
            adj_e = reg_r * np.ones((data['covariate'][indx_lists[j][k]].shape[0],))
            adj_p0 = 1 / adj_e
            adj_g = reg_g.predict(data['covariate'][indx_lists[j][k]])
            gamma_j = adj_p0 * (adj_Gval - adj_g) + adj_g # double robust policy value estimate

            Gamma_matrix[[indx_lists[j][k]],j] = - gamma_j

        for j in range(2):
            adj_gj = reg_g_list[(j+1)%2].predict(data['covariate'][indx_lists[j][k]])
            Gamma_matrix[[indx_lists[j][k]],(j+1)%2] = - adj_gj

    return Gamma_matrix

def policy_predict(pi, test_data: jnp.ndarray):
    """
    Reteieve policy action according to the testing dataset covariates.
    """
    opt_pi_T = pt.predict_policy_tree(pi, test_data)
    temp_data_dic = {'covariate': test_data, 'action': opt_pi_T}
    return temp_data_dic

def ground_true_V_optpi(pi, data: DataInput, delta: float):
    """
    Compute the true policy value using the testing dataset.
    """
    test_X = data.s
    pi_data = policy_predict(pi, test_X)
    A = np.array([int(x) for x in pi_data['action']])
    fitted_A = A-1
    Y_pi = data.reward_mat[range(data.a.shape[0]), fitted_A]
    pi_data.update({'outcome': Y_pi})

    all_probs = []
    all_losses = []

    for i in range(3):
        this_run = opt_ERM(pi_data['covariate'], pi_data['outcome'], delta)
        all_probs.append(this_run)
        all_losses.append(this_run["val"])

    opt_prob = all_probs[np.argmin(all_losses)]


    Xdat = turn_spline(pi_data['covariate'])
    ax = np.maximum(eps, Xdat @ opt_prob['alpha'])
    eta = Xdat @ opt_prob["eta"]
    Gval = ax * fc(ax, eta, pi_data['outcome']) + eta + ax * delta

    V_pi = - np.mean(Gval)
    return V_pi

# =======================================================================================================================
# # Run experiment
# =======================================================================================================================

if __name__ == '__main__':
    seed = int(sys.argv[1])
    deltas = [0.05, 0.1]
    n_list = [10000, 12500, 15000, 17500, 20000]

    test_data_path = f'data/{seed}/test.pkl' # load testing dataset of this seed
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    results ={}
    for delta in deltas:
        results[delta] = []

    for delta in deltas:
        for n in n_list:
            data_path = f'data/{seed}/{n}.pkl'
            with open(data_path, 'rb') as f: # load training dataset
                train_data = pickle.load(f)
            data = {'covariate': train_data.s, 'action': train_data.a+1, 'outcome': train_data.r}

            Matrix_Gamma = produce_Gamma(data, delta)
            opt_pi = pt.policy_tree(data['covariate'], Matrix_Gamma, L) # learn policy via policy tree search

            opt_pi_V = ground_true_V_optpi(opt_pi, test_data, delta) # evaluate the learnt policy LN
            results[delta].append(opt_pi_V)

    to_save = pd.DataFrame.from_dict(results, orient='index') # save result of this seed
    to_save.columns = n_list

    result_dir = f'ln_results/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    to_save.to_csv(f'ln_results/{seed}.csv')
