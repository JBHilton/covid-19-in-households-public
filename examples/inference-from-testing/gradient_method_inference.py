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
from model.specs import SINGLE_AGE_SEIR_SPEC_FOR_FITTING, SINGLE_AGE_UK_SPEC
from model.common import UnloopedSEIRRateEquations, UnloopedSEPIRRateEquations
from model.imports import FixedImportModel, NoImportModel


if 'inference-from-testing' in os.getcwd():
    os.chdir("../..")
os.getcwd()

# Load synthetic data generated in script Synthetic_data_generator.py
pickle_path = r"C:\Users\igmsb\Documents\Github\covid-19-in-households-public\examples\inference-from-testing\synthetic_data_simulation_result.pkl"

with open(pickle_path, "rb") as f:
    results = pickle.load(f)

multi_hh_data = results["multi_hh_data"]
#base_rhs = results["base_rhs"]

print(f"Loaded synthetic data: {len(multi_hh_data)} households")

# Model setup
comp_dist = array([1])
composition_list = array([[3]])

SPEC = {**SINGLE_AGE_SEIR_SPEC_FOR_FITTING, **SINGLE_AGE_UK_SPEC}
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
x0 = array([1e-2]) # Initial external prevalence
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

# Step 5: Compute the likelihood

# Step 5a: Set the ground work

# Prep/work
N_HH = int(composition_list[0, 0])  # household size
y1 = int(1)
y2 = int(1)

# Map to Chi matrices
k1 = result_to_index[(N_HH, y1)]
k2 = result_to_index[(N_HH, y2)]
Chi_1 = Chi[k1]
Chi_2 = Chi[k2]

# Initial distribution for this household type
P0 = hh_models[N_HH][2]

# Rate equation object for this household type
rhs = hh_models[N_HH][1]
Q1 = rhs.Q_int
Q2 = pop_prev * rhs.Q_ext
Q0 = rhs.Q_int_fixed

# Times
t0 = 0.0
t1 = test_times[0]
t2 = test_times[1]

def likelihood_tau_lambda(tau, lam, t0, t1, t2, P0, Q1, Q2, Q0, Chi_1, Chi_2):


    # Ingridients
    Q_theta = tau * Q1.T + lam * Q2.T + Q0.T
    A = expm((t1 - t0) * Q_theta)
    B = expm((t2 - t1) * Q_theta)
    u = Chi_1.dot(A.dot(P0))
    v = Chi_2.dot(B.dot(u))

    # print(A.todense())
    # print(B.todense())
    # print(A.dot(P0))
    # print(u)
    # print(v)

    # Likelihood
    return np.sum(v) / np.sum(u)

# Compute likelihood
like_val = likelihood_tau_lambda(tau_0, lambda_0, t0, t1, t2, P0, Q1, Q2, Q0, Chi_1, Chi_2)
print(f"Likelihood for household 0: {like_val}")

# Step 6: Compute the gradient of the likelihood

def gradients(tau, lam, t0, t1, t2, P0, Q1, Q2, Q0, Chi_1, Chi_2):
    # Ingridiens
    Q_theta = tau * Q1 + lam * Q2 + Q0
    A = expm((t1 - t0) * Q_theta)
    B = expm((t2 - t1) * Q_theta)
    u = Chi_1 @ (A @ P0)
    v = Chi_2 @ (B @ u)

    # Norms
    u_norm = np.sum(u)
    v_norm = np.sum(v)

    # du/dtau, du/dlambda
    du_dtau = Chi_1 @ (Q1 @ (A @ P0))
    du_dlam = Chi_1 @ (Q2 @ (A @ P0))

    du_dtau_norm = one_vec @ du_dtau
    du_dlam_norm = one_vec @ du_dlam

    # dv/dtau, dv/dlambda
    dv_dtau = Chi_2 @ (Q1 @ (B @ (Chi_1 @ (A @ P0))) + B @ (Chi_1 @ (Q1 @ (A @ P0))))
    dv_dlam = Chi_2 @ (Q2 @ (B @ (Chi_1 @ (A @ P0))) + B @ (Chi_1 @ (Q2 @ (A @ P0))))

    dv_dtau_norm = one_vec @ dv_dtau
    dv_dlam_norm = one_vec @ dv_dlam

    # Likelihood
    L = v_norm / u_norm

    # Gradients (quotient rule)
    dL_dtau = (dv_dtau_norm * u_norm - v_norm * du_dtau_norm) / (u_norm ** 2)
    dL_dlam = (dv_dlam_norm * u_norm - v_norm * du_dlam_norm) / (u_norm ** 2)

    return L, dL_dtau, dL_dlam

def gradients(tau, lam, t0, t1, t2, P0, Q1, Q2, Q0, Chi_1, Chi_2):
    one_vec = np.ones((P0.shape[0], 1))

    # Generator
    Q_theta = tau * Q1.T + lam * Q2.T + Q0.T

    # Evolution matrices
    A = expm((t1 - t0) * Q_theta)
    B = expm((t2 - t1) * Q_theta)

    # Forward propagation
    u = Chi_1 @ (A @ P0)
    v = Chi_2 @ (B @ u)

    # Norms
    u_norm = float(one_vec.T @ u)
    v_norm = float(one_vec.T @ v)

    # Derivatives of u wrt tau and lambda
    du_dtau = Chi_1 @ (Q1.T @ (A @ P0))
    du_dlam = Chi_1 @ (Q2.T @ (A @ P0))

    du_dtau_norm = float(one_vec.T @ du_dtau)
    du_dlam_norm = float(one_vec.T @ du_dlam)

    # Derivatives of v wrt tau and lambda
    dv_dtau = Chi_2 @ (Q1.T @ (B @ (Chi_1 @ (A @ P0))) +
                       B @ (Chi_1 @ (Q1.T @ (A @ P0))))
    dv_dlam = Chi_2 @ (Q2.T @ (B @ (Chi_1 @ (A @ P0))) +
                       B @ (Chi_1 @ (Q2.T @ (A @ P0))))

    dv_dtau_norm = float(one_vec.T @ dv_dtau)
    dv_dlam_norm = float(one_vec.T @ dv_dlam)

    # Likelihood
    L = v_norm / u_norm

    # Gradients (quotient rule)
    dL_dtau = (dv_dtau_norm * u_norm - v_norm * du_dtau_norm) / (u_norm ** 2)
    dL_dlam = (dv_dlam_norm * u_norm - v_norm * du_dlam_norm) / (u_norm ** 2)

    return L, dL_dtau, dL_dlam

L, dL_dtau, dL_dlam = gradients(tau_0, lambda_0, t0, t1, t2, P0, Q1, Q2, Q0, Chi_1, Chi_2)
print(f"Likelihood: {L}")
print(f"dL/dtau: {dL_dtau}")
print(f"dL/dlambda: {dL_dlam}")

# All households likelihood calculation

def all_households_loglike_and_grads(tau, lam, t0, t1, t2, hh_models, result_to_index, Chi, multi_hh_data):
    loglike = 0.0
    grad_tau = 0.0
    grad_lam = 0.0

    for hh_data in multi_hh_data:
        # Household size (assuming fixed for now)
        N_HH = int(next(iter(hh_models.keys())))
        P0 = hh_models[N_HH][2]
        rhs = hh_models[N_HH][1]
        Q1, Q2, Q0 = rhs.Q_int, rhs.Q_ext, rhs.Q_int_fixed

        # Observations (positives at test times)
        obs1 = (N_HH, int(hh_data[0]))
        obs2 = (N_HH, int(hh_data[1]))

        # Map to Chi matrices
        k1 = result_to_index[obs1]
        k2 = result_to_index[obs2]
        Chi_1 = Chi[k1]
        Chi_2 = Chi[k2]

        # Households likelihood & gradients
        L, dL_dtau, dL_dlam = gradients(
            tau, lam, t0, t1, t2, P0, Q1, Q2, Q0, Chi_1, Chi_2
        )

        # Add to log-likelihood
        loglike += np.log(L + 1e-12)  # safeguard against log(0)

        # Add gradients (chain rule for log-likelihood)
        grad_tau += dL_dtau / (L + 1e-12)
        grad_lam += dL_dlam / (L + 1e-12)

    return loglike, grad_tau, grad_lam

logL, dLdtau, dLdlam = all_households_loglike_and_grads(
    tau_0, lambda_0, t0, t1, t2, hh_models, result_to_index, Chi, multi_hh_data)

print("Dataset log-likelihood:", logL)
print("Gradient wrt tau:", dLdtau)
print("Gradient wrt lambda:", dLdlam)
