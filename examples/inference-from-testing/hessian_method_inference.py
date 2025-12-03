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
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import expm
from scipy.stats import lognorm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import warnings

from model.preprocessing import (
    estimate_beta_ext, estimate_growth_rate,
    SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector
)
from model.specs import SINGLE_AGE_SEIR_SPEC_FOR_FITTING, SINGLE_AGE_UK_SPEC, SINGLE_TYPE_INFERENCE_SPEC
from model.common import UnloopedSEIRRateEquations, UnloopedSEPIRRateEquations
from model.imports import FixedImportModel, NoImportModel

#########USE IF NEED SYNTHETIC DATA GENERATOR FILE
#if 'inference-from-testing' in os.getcwd():
#    os.chdir("../..")
#os.getcwd()

# Load synthetic data generated in script Synthetic_data_generator.py
#pickle_path = "synthetic_data_simulation_fixed_import_results.pkl"

#with open(pickle_path, "rb") as f:
#    results = pickle.load(f)

#multi_hh_data = results["multi_hh_data"]
#rhs = results["base_rhs"]
###############

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

##### DATA GENERATOR
# Initialize model input based on specifications
model_input_to_fit = SEIRInput(SPEC, composition_list, comp_dist)
'''Quick fix to make sure initial states are in form S>0, I>0 rather than S>0, E>0'''
model_input_to_fit.new_case_compartment = 2
#true_density_expo = .5 # Todo: bring this in from model input rather than defining directly
test_times = np.array([14, 28])

def initialise_at_first_test_date(rhs,
                                  H0):
    base_sol = solve_ivp(rhs,
                         (0, test_times[0]),
                         H0,
                         first_step=0.001,
                         atol=1e-16)
    H_t0 = base_sol.y[:, -1]
    return H_t0


# Simulation function to generate data
def run_simulation(lambda_val, tau_val):
    model_input = deepcopy(model_input_to_fit)
    model_input.k_home *= tau_val  / model_input_to_fit.beta_int
    model_input.beta_int = tau_val
    model_input.k_ext *= lambda_val

    true_density_expo = model_input.density_expo

    # With the parameters chosen, we calculate Q_int:
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input)

    no_imports = NoImportModel(4, 1)

    base_rhs = UnloopedSEIRRateEquations(model_input, household_population, no_imports)
    r_est = estimate_growth_rate(household_population, base_rhs, [0.001, 5], 1e-9)
    base_H0 = make_initial_condition_by_eigenvector(r_est,
                                                    model_input,
                                                    household_population,
                                                    base_rhs,
                                                    pop_prev,
                                                    0.0,
                                                    False,
                                                    3)
    x0 = base_H0.T.dot(household_population.states[:, 2::4])

    fixed_imports = FixedImportModel(4,
                                     1,
                                     base_rhs,
                                     x0)

    rhs = UnloopedSEIRRateEquations(model_input, household_population, fixed_imports, sources="IMPORT")
    H_t0 = initialise_at_first_test_date(rhs,
                                  base_H0)

    solve_times = np.array([0, test_times[0], test_times[1]])
    def generate_single_hh_test_data(test_times):
        Ht = deepcopy(H_t0)
        test_data = np.zeros((len(test_times),))
        for i in range(len(test_times)):
            tspan = (solve_times[i], solve_times[i+1])
            solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-16)
            T = solution.t
            H = solution.y
            state = choice(range(len(H[:, -1])), 1, p=H[:, -1] / sum(H[:, -1]))
            test_data[i] = rhs.states_inf_only[state]
            Ht *= 0
            Ht[state] = 1
        return (test_data)

    ## Now do multiple households

    # Generate test data:

    multi_hh_data = [generate_single_hh_test_data(test_times) for i in range(n_hh)]
    return multi_hh_data, rhs

# Make sure to run this first:
multi_hh_data, rhs = run_simulation(lambda_0, tau_0)

######################################


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

# Ingredients
Q_theta = tau_0 * Q1.T + lambda_0 * Q2.T + Q0.T
A = expm((t1 - t0) * Q_theta)
B = expm((t2 - t1) * Q_theta)

# If we could get fractional matrix exponentials to work effectively, this stuff could be precalculated:
A1 = expm((t1 - t0) * Q1.T)
B1 = expm((t2 - t1) * Q1.T)
A2 = expm((t1 - t0) * Q2.T)
B2 = expm((t2 - t1) * Q2.T)
A0 = expm((t1 - t0) * Q0.T)
B0 = expm((t2 - t1) * Q0.T)

def likelihood_tau_lambda(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2):

    u = Chi_1.dot(A.dot(P0))
    v = Chi_2.dot(B.dot(u))

    # Likelihood
    lh = np.sum(v) / np.sum(u)
    llh = np.log(sum(v)) - np.log(sum(u))

    return lh, llh, u, v

# Compute likelihood
single_llh_val = likelihood_tau_lambda(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2)
print(f"Likelihood for a single household: {single_llh_val[0]}")
print(f"Log-Likelihood for a single household: {single_llh_val[1]}")

# Step 6: Compute the gradient of the likelihood

Chi_1_Q1 = Chi_1.dot(Q1.T)
Chi_1_Q2 = Chi_1.dot(Q2.T)
Chi_2_Q1 = Chi_2.dot(Q1.T)
Chi_2_Q2 = Chi_2.dot(Q2.T)

def gradients(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2, u, v):

    # Norms
    u_norm = u.sum()
    v_norm = v.sum()

    # Derivatives of u wrt tau and lambda
    du_dtau = (t1 - t0) * Chi_1.dot(Q1.T).dot(A.dot(P0))
    du_dlam = (t1 - t0) * Chi_1.dot(Q2.T).dot(A.dot(P0))

    du_dtau_norm = du_dtau.sum()
    du_dlam_norm = du_dlam.sum()

    # Derivatives of v wrt tau and lambda
    dv_dtau = Chi_2.dot((t2 - t1) * Q1.T.dot(B.dot(u)) +
                        B.dot(du_dtau))
    dv_dlam = Chi_2.dot((t2 - t1) * Q2.T.dot(B.dot(u)) +
                        B.dot(du_dlam))

    dv_dtau_norm = dv_dtau.sum()
    dv_dlam_norm = dv_dlam.sum()

    # Gradients (quotient rule)
    dL_dtau = (dv_dtau_norm * u_norm - v_norm * du_dtau_norm) / (u_norm ** 2)
    dL_dlam = (dv_dlam_norm * u_norm - v_norm * du_dlam_norm) / (u_norm ** 2)

    return dL_dtau, dL_dlam

u = Chi_1.dot(A.dot(P0))
v = Chi_2.dot(B.dot(u))
single_grad_val = gradients(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2, u, v)
print(f"Gradient wrt tau for a single household: {single_grad_val[0]}")
print(f"Gradient wrt lam for a single household: {single_grad_val[1]}")

# All households likelihood calculation

def all_households_loglike_and_grads(tau, lam, hh_models, result_to_index, Chi, multi_hh_data):
    Q_theta = tau * Q1.T + lam * Q2.T + Q0.T
    A = expm((t1 - t0) * Q_theta)
    B = expm((t2 - t1) * Q_theta)

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
        lh, llh, u, v = likelihood_tau_lambda(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2)
        dL_dtau, dL_dlam = gradients(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2, u, v)

        # Total likelihood / log-likelihood
        total_lh *= lh
        total_llh += llh

        # Add gradients for log-likelihood
        grad_tau += dL_dtau / (lh + 1e-12)
        grad_lam += dL_dlam / (lh + 1e-12)

    return total_lh, total_llh, grad_tau, grad_lam

# Compute totals for all households
total_lh, total_llh, grad_tau, grad_lam = all_households_loglike_and_grads(
    tau_0, lambda_0, hh_models, result_to_index, Chi, multi_hh_data
)

# Print results
print(f"Total likelihood for all households: {total_lh}")
print(f"Total log-likelihood for all households: {total_llh}")
print(f"Gradient wrt tau for all households: {grad_tau}")
print(f"Gradient wrt lambda for all households: {grad_lam}")

# Compute the conditional log-likelihood, collapsed over unique datapoints.

def loglike_with_counts(tau, lam, P0, Q1, Q2, Q0, Chi, result_to_index, multi_hh_data):
    Q_theta = tau * Q1.T + lam * Q2.T + Q0.T
    A = expm((t1 - t0) * Q_theta)
    B = expm((t2 - t1) * Q_theta)

    obs_counts = Counter(tuple(map(int, hh)) for hh in multi_hh_data)

    total_llh = 0.0
    total_lh = 1.0
    grad_tau = .0
    grad_lam = .0

    for (y1, y2), count in obs_counts.items():
        # Map to Chi matrices
        k1 = result_to_index[(N_HH, y1)]
        k2 = result_to_index[(N_HH, y2)]
        Chi_1 = Chi[k1]
        Chi_2 = Chi[k2]

        # Household likelihood & gradients
        lh, llh, u, v = likelihood_tau_lambda(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2)
        dL_dtau, dL_dlam = gradients(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2, u, v)

        # Total likelihood / log-likelihood
        total_lh *= (lh ** count)
        total_llh += count * llh

        # Add gradients
        grad_tau += count * dL_dtau / lh
        grad_lam += count * dL_dlam / lh


        # Print each unique observation
        # print(f"Obs (y1={y1}, y2={y2}), count={count}, lh={lh:.6e}, log-like contrib={count*np.log(lh):.6f}")

    return total_lh, total_llh, grad_tau, grad_lam


total_lh, total_llh, grad_tau, grad_lam = loglike_with_counts(
    tau_0, lambda_0, P0, Q1, Q2, Q0, Chi, result_to_index, multi_hh_data)

# Print results
print(f"Total likelihood for all households: {total_lh:.6e}")
print(f"Total log-likelihood for all households: {total_llh:.6f}")

def all_households_loglike(tau, lam, result_to_index, Chi, multi_hh_data):
    Q_theta = tau * Q1.T + lam * Q2.T + Q0.T
    A = expm((t1 - t0) * Q_theta)
    B = expm((t2 - t1) * Q_theta)

    total_lh = 1.0        # start from 1.0 since we multiply likelihoods
    total_llh = 0.0


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
        lh, llh, u, v = likelihood_tau_lambda(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2)

        # Total likelihood / log-likelihood
        total_lh *= lh
        total_llh += llh

    return total_lh, total_llh

# Negative log-likelihood wrapper using the collapsed-with-counts function you wrote
def neg_loglike(params):
    """Return negative total log-likelihood for params = [tau, lam]."""
    tau, lam = params[0], params[1]
    # loglike_with_counts returns: total_lh, total_llh, grad_tau, grad_lam
    _, total_llh, grad_tau, grad_lam = loglike_with_counts(
        tau, lam, P0, Q1, Q2, Q0, Chi, result_to_index, multi_hh_data
    )
    return -total_llh

def neg_loglike_and_grad(params):
    """Return (neg_loglike, neg_grad) for optimizer that accepts jac."""
    tau, lam = params[0], params[1]
    _, total_llh, grad_tau, grad_lam = loglike_with_counts(
        tau, lam, P0, Q1, Q2, Q0, Chi, result_to_index, multi_hh_data
    )
    neg_ll = -total_llh
    # grad_tau and grad_lam are derivatives of the log-likelihood; we need derivative of -log-likelihood
    neg_grad = -np.array([grad_tau, grad_lam], dtype=float)
    return neg_ll, neg_grad

# Callback to watch progress
def callback_print(xk):
    print("trying", xk)

# Initial guess and bounds (ensure tau, lambda remain positive)
initial_guess = np.array([tau_0, lambda_0])  # you already set tau_0, lambda_0 above
bounds = [(1e-8, None), (1e-8, None)]  # enforce positivity

# Choose optimizer: L-BFGS-B (use jac), fallback to Nelder-Mead if you prefer derivative-free
#I think L-BFGS-B is a bit better and faster. (opinion)
#res = minimize(
#    neg_loglike,
#    initial_guess,
#    method="Nelder-Mead",
#    callback=lambda x: print("trying", x),
#    options={"maxiter": 1000, "disp": True}
#)

res = minimize(
    fun=lambda p: neg_loglike_and_grad(p)[0],
    x0=initial_guess,
    method="L-BFGS-B",
    jac=lambda p: neg_loglike_and_grad(p)[1],
    bounds=bounds,
    callback=callback_print,
    options={"disp": True, "maxiter": 1000}
)

# If L-BFGS-B fails or you explicitly want Nelder-Mead:
# res = minimize(lambda p: neg_loglike(p), initial_guess, method='Nelder-Mead', callback=callback_print)

print("\nOptimization result:\n", res)

# Extract MLE
xhat = res.x
tau_hat, lam_hat = xhat[0], xhat[1]
print(f"\nEstimated parameters: tau = {tau_hat:.6f}, lambda = {lam_hat:.6f}")

# Hessian and covariance estimation
def compute_hessian_at(f, x0):
    """Try numdifftools Hessian, fallback to finite-diff (approx)."""
    if _have_numdifftools:
        Hfun = nd.Hessian(f)
        H = Hfun(x0)
    else:
        # Finite-diff approximation using 2-sided differences (small step)
        eps = np.sqrt(np.finfo(float).eps)
        n = len(x0)
        H = np.zeros((n, n), float)
        f0 = f(x0)
        for i in range(n):
            ei = np.zeros(n); ei[i] = 1.0
            for j in range(i, n):
                ej = np.zeros(n); ej[j] = 1.0
                h = eps * max(1.0, abs(x0[i]), abs(x0[j]))
                x_pp = x0 + 0.5*h*(ei+ej)
                x_pm = x0 + 0.5*h*(ei-ej)
                x_mp = x0 + 0.5*h*(-ei+ej)
                x_mm = x0 - 0.5*h*(ei+ej)
                # mixed second derivative approx
                H_ij = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (h*h)
                H[i, j] = H_ij
                H[j, i] = H_ij
    return H

# Compute Hessian of negative log-likelihood at the MLE
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    H = compute_hessian_at(neg_loglike, xhat)

# Optional: try to use numdifftools if installed for Hessian
try:
    import numdifftools as nd
    _have_numdifftools = True
except Exception:
    _have_numdifftools = False
    from scipy.linalg import pinv

# Try to invert Hessian to get covariance; handle non-PD case gracefully
try:
    # If Hessian is positive definite, np.linalg.inv works
    cov = np.linalg.inv(H)
except Exception:
    # fallback to pseudo-inverse
    cov = pinv(H)

# Extract standard errors and 95% CIs
stds = np.sqrt(np.abs(np.diag(cov)))  # abs to avoid tiny negative numerical noise
ci_lower = xhat - 1.96 * stds
ci_upper = xhat + 1.96 * stds

print("\nStandard errors:", stds)
print("95% CI for tau: [{:.6f}, {:.6f}]".format(ci_lower[0], ci_upper[0]))
print("95% CI for lambda: [{:.6f}, {:.6f}]".format(ci_lower[1], ci_upper[1]))
