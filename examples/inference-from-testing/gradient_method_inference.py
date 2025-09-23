SAVE_INFERENCE_RESULTS = True
# Imports
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, savefig
from matplotlib.cm import get_cmap
from pickle import dump, load
import scipy as sp
from scipy.integrate import solve_ivp
import time
from tqdm import tqdm
from numpy import linalg as LA
import pickle
from numpy.linalg import eig
from collections import Counter


import numpy as np
from numpy.random import choice
from scipy.optimize import minimize
import scipy as sp
import os
from copy import deepcopy
from datetime import datetime
from numpy import arange, array, atleast_2d, log, sum, where, zeros
import pandas
from pandas import read_csv
from scipy.sparse.linalg import expm
from scipy.stats import lognorm
from scipy.sparse import csr_matrix

from model.preprocessing import (
    estimate_beta_ext, estimate_growth_rate,
    SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector
)
from model.specs import SINGLE_AGE_SEIR_SPEC_FOR_FITTING, SINGLE_AGE_UK_SPEC, SINGLE_TYPE_INFERENCE_SPEC
from model.common import UnloopedSEIRRateEquations, UnloopedSEPIRRateEquations
from model.imports import FixedImportModel, NoImportModel


if 'inference-from-testing' in os.getcwd():
    os.chdir("../..")
os.getcwd()

# Load synthetic data generated in script Synthetic_data_generator.py
pickle_path = "synthetic_data_simulation_fixed_import_results.pkl"

with open(pickle_path, "rb") as f:
    results = pickle.load(f)

multi_hh_data = results["multi_hh_data"]
#rhs = results["base_rhs"]

# Model setup
comp_dist = array([1])
composition_list = array([[3]])

SPEC = SINGLE_TYPE_INFERENCE_SPEC
DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

n_sims = 10
lambda_0 = 3.0
tau_0 = 0.25
n_hh = 100
pop_prev = 1e-3
test_times = np.array([14, 28])

# Single Household likelihood calculation
# Step 1: Create mappings
def create_result_mappings(max_hh_size):
    result_to_index = {}  # List: (N_HH, y) → index
    index_to_result = []  # List: index → (N_HH, y)

    for N in range(1, max_hh_size + 1):     # Household sizes
        for y in range(0, N + 1):           # Number of positives in that household
            k = len(index_to_result)
            result_to_index[(N, y)] = k
            index_to_result.append((N, y))

    return result_to_index, index_to_result

max_hh_size = composition_list.max()  # e.g., 3
result_to_index, index_to_result = create_result_mappings(max_hh_size)

# Check step 1:
print("Mapping (N_HH, y) → index:")
for k, v in result_to_index.items():
    print(f"{k} → {v}")

# Step 2: Build HouseholdPopulation + SEIRRateEquations for each N_HH
hh_models = {}  # key = N_HH, value = (household_population, rate_equation)
x0 = array([pop_prev]) # Initial external prevalence
for N in range(1, max_hh_size + 1):
    comp_list = np.array([[N]])
    comp_dist = np.array([1.0])  # dummy value; only one composition
    model_input = SEIRInput(SPEC, comp_list, comp_dist)
    hh_pop = HouseholdPopulation(comp_list, comp_dist, model_input)
    no_imports = NoImportModel(4, 1)
    rhs = UnloopedSEIRRateEquations(model_input, hh_pop, no_imports)
    fixed_imports = FixedImportModel(4,
                                     1,
                                     rhs,
                                     x0)
    rhs = UnloopedSEIRRateEquations(model_input,
                                    hh_pop,
                                    fixed_imports,
                                    sources="IMPORT")
    P0 = zeros(rhs.total_size, )
    P0[where(abs(rhs.states_sus_only - N) < 1e-1)[0]] = 1.
    hh_models[N] = (hh_pop, rhs, P0)

# Step 3: Build Chi list
from scipy.sparse import csr_matrix
Chi = []
for (N_HH, y) in index_to_result:
    hh_pop, rhs, P0 = hh_models[N_HH]
    states_inf_only = rhs.states_inf_only
    # Step 3a: Find valid states for given y
    possible_states = np.where(abs(states_inf_only - y) < 1e-1)[0]

    # Step 3b: Build sparse Chi matrix
    size = len(states_inf_only)
    data = np.ones(len(possible_states))
    rowcol = possible_states
    Chi_k = csr_matrix((data, (rowcol, rowcol)), shape=(size, size))
    Chi.append(Chi_k)

#Step 4: Extract Chi for the inference
for hh_data in multi_hh_data:  # hh_data is array like [y1, y2]?
    N_HH = int(composition_list[0, 0])  # Assuming uniform household size for now

    obs1 = (N_HH, int(hh_data[0]))
    obs2 = (N_HH, int(hh_data[1]))

    k1 = result_to_index[obs1]
    k2 = result_to_index[obs2]

    Chi_1 = Chi[k1]
    Chi_2 = Chi[k2]

    #print(Chi_1.toarray())
    #print(Chi_2.toarray())

# Step 5a: Set the ground work

# Prep/work
N_HH = int(composition_list[0, 0])  # household size
y1 = multi_hh_data[0][0]
y2 = multi_hh_data[0][1]

# Map to Chi matrices
k1 = result_to_index[(N_HH, y1)]
k2 = result_to_index[(N_HH, y2)]
Chi_1 = Chi[k1]
Chi_2 = Chi[k2]

# Initial distribution for this household type
P0 = hh_models[N_HH][2]

# Rate equation object for this household type
rhs = hh_models[N_HH][1]
Q1 = rhs.Q_int_inf
Q2 = rhs.Q_import
Q0 = rhs.Q_int_fixed

# Times
t0 = 0.0
t1 = test_times[0]
t2 = test_times[1]

# Ingridients
Q_theta = tau_0 * Q1.T + lambda_0 * Q2.T + Q0.T
A = expm((t1 - t0) * Q_theta)
B = expm((t2 - t1) * Q_theta)

def likelihood_tau_lambda(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2):

    u = Chi_1.dot(A.dot(P0))
    v = Chi_2.dot(B.dot(u))

    # Likelihood
    lh = np.sum(v) / np.sum(u)
    llh = np.log(sum(v)) - np.log(sum(u))

    return lh, llh

# Compute likelihood
single_llh_val = likelihood_tau_lambda(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2)
print(f"Likelihood for a single household: {single_llh_val[0]}")
print(f"Log-Likelihood for a single household: {single_llh_val[1]}")

# Step 6: Compute the gradient of the likelihood

def gradients(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2):

    u = Chi_1.dot(A.dot(P0))
    v = Chi_2.dot(B.dot(u))

    # Norms
    u_norm = u.sum()
    v_norm = v.sum()

    # Derivatives of u wrt tau and lambda
    du_dtau = Chi_1.dot(Q1.dot(A.dot(P0)))
    du_dlam = Chi_2.dot(Q2.dot(B.dot(u)))

    du_dtau_norm = du_dtau.sum()
    du_dlam_norm = du_dlam.sum()

    # Derivatives of v wrt tau and lambda
    dv_dtau = Chi_2.dot(Q1.T.dot(B.dot(Chi_1.dot(A.dot(P0)))) +
                        B.dot(Chi_1.dot(Q1.T.dot(A.dot(P0)))))
    dv_dlam = Chi_2.dot(Q2.T.dot(B.dot(Chi_1.dot(A.dot(P0)))) +
                        B.dot(Chi_1.dot(Q2.T.dot(A.dot(P0)))))

    dv_dtau_norm = dv_dtau.sum()
    dv_dlam_norm = dv_dlam.sum()

    # Gradients (quotient rule)
    dL_dtau = (dv_dtau_norm * u_norm - v_norm * du_dtau_norm) / (u_norm ** 2)
    dL_dlam = (dv_dlam_norm * u_norm - v_norm * du_dlam_norm) / (u_norm ** 2)

    return dL_dtau, dL_dlam

single_grad_val = gradients(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2)
print(f"Gradient wrt tau for a single household: {single_grad_val[0]}")
print(f"Gradient wrt lam for a single household: {single_grad_val[1]}")

# All households likelihood calculation

def all_households_loglike_and_grads(A, B, hh_models, result_to_index, Chi, multi_hh_data):
    total_lh = 1.0        # start from 1.0 since we multiply likelihoods
    total_llh = 0.0
    grad_tau = 0.0
    grad_lam = 0.0


    llh_list = []
    for hh_data in multi_hh_data:
        # Observations (positives at test times)
        obs1 = (N_HH, int(hh_data[0]))
        obs2 = (N_HH, int(hh_data[1]))

        # Map to Chi matrices
        k1 = result_to_index[obs1]
        k2 = result_to_index[obs2]
        Chi_1 = Chi[k1]
        Chi_2 = Chi[k2]

        # Household likelihood & gradients
        lh, llh = likelihood_tau_lambda(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2)
        dL_dtau, dL_dlam = gradients(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2)

        # Total likelihood / log-likelihood
        total_lh *= lh
        total_llh += llh

        # Add gradients for log-likelihood
        grad_tau += dL_dtau / (lh + 1e-12)
        grad_lam += dL_dlam / (lh + 1e-12)

    return total_lh, total_llh, grad_tau, grad_lam

# Compute totals for all households
total_lh, total_llh, grad_tau, grad_lam = all_households_loglike_and_grads(
    A, B, hh_models, result_to_index, Chi, multi_hh_data
)

# Print results
print(f"Total likelihood for all households: {total_lh}")
print(f"Total log-likelihood for all households: {total_llh}")
print(f"Gradient wrt tau for all households: {grad_tau}")
print(f"Gradient wrt lambda for all households: {grad_lam}")

# Compute the conditional log-likelihood, collapsed over unique datapoints.

def loglike_with_counts(A, B, P0, Q1, Q2, Q0, Chi, result_to_index, multi_hh_data):

    obs_counts = Counter(tuple(map(int, hh)) for hh in multi_hh_data)

    total_llh = 0.0
    total_lh = 1.0

    for (y1, y2), count in obs_counts.items():
        # Map to Chi matrices
        k1 = result_to_index[(N_HH, y1)]
        k2 = result_to_index[(N_HH, y2)]
        Chi_1 = Chi[k1]
        Chi_2 = Chi[k2]

        u = Chi_1 @ (A @ P0)
        v = Chi_2 @ (B @ u)

        lh = np.sum(v) / np.sum(u)

        total_lh = total_lh * (lh ** count)
        total_llh = total_llh + count * np.log(lh)


        # Print each unique observation
        print(f"Obs (y1={y1}, y2={y2}), count={count}, lh={lh:.6e}, log-like contrib={count*np.log(lh):.6f}")

    return total_lh, total_llh


total_lh, total_llh = loglike_with_counts(
    A, B, P0, Q1, Q2, Q0, Chi, result_to_index, multi_hh_data)

# Print results
print(f"Total likelihood for all households: {total_lh:.6e}")
print(f"Total log-likelihood for all households: {total_llh:.6f}")