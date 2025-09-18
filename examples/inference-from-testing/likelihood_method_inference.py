from numpy.matlib import zeros

SAVE_INFERENCE_RESULTS = True
# Import required libraries
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, savefig
from matplotlib.cm import get_cmap
from pickle import dump, load
import scipy as sp
from scipy.integrate import solve_ivp
import time
from tqdm import tqdm
from scipy.integrate import solve_ivp
from numpy import linalg as LA
import pickle
from numpy.linalg import eig

import numpy as np
from numpy.random import choice
from scipy.optimize import minimize
import scipy as sp

# Adjust for Windows file path if necessary
import os

if 'inference-from-testing' in os.getcwd():
    os.chdir("../..")
os.getcwd()

# Import required libraries
import numpy as np
from scipy.sparse.linalg import expm
from numpy.random import choice
from scipy.optimize import minimize
import scipy as sp
import time
from copy import deepcopy
from datetime import datetime
from numpy import arange, array, atleast_2d, log, sum, where
import pandas
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import SINGLE_AGE_SEIR_SPEC_FOR_FITTING, SINGLE_AGE_UK_SPEC, SINGLE_TYPE_INFERENCE_SPEC
from model.common import UnloopedSEIRRateEquations, UnloopedSEPIRRateEquations
from model.imports import FixedImportModel
from model.imports import NoImportModel

# Load synthetic data generated in script Synthetic_data_generator.py
pickle_path = "synthetic_data_simulation_fixed_import_results.pkl"

with open(pickle_path, "rb") as f:
    results = pickle.load(f)

multi_hh_data = results["multi_hh_data"]
rhs = results["rhs"]

DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

lambda_0 = 3.0
tau_0 = 0.25
pop_prev = 1e-3
test_times = np.array([14, 28])

# Set up
SPEC = SINGLE_TYPE_INFERENCE_SPEC
comp_dist = np.array([1.])
comp_list = np.array([[3]])
model_input = SEIRInput(SPEC, comp_list, comp_dist)
household_population = HouseholdPopulation(comp_list, comp_dist, model_input)
no_imports = NoImportModel(
    no_inf_compartments=model_input.no_inf_compartments,
    no_age_classes=model_input.no_age_classes
)
base_rhs = UnloopedSEIRRateEquations(model_input, household_population, no_imports)
r_est = estimate_growth_rate(household_population, base_rhs, [0.001, 5], 1e-9)
base_H0 = make_initial_condition_by_eigenvector(
    r_est,
    model_input,
    household_population,
    base_rhs,
    pop_prev,
    0.0,
    False,
    3
)
x0 = base_H0.T.dot(household_population.states[:, 2::4])
fixed_imports = FixedImportModel(
    no_inf_compartments=model_input.no_inf_compartments,
    no_age_classes=model_input.no_age_classes,
    rhs=base_rhs,
    x0=x0
)
rhs = UnloopedSEIRRateEquations(model_input, household_population, fixed_imports, sources="IMPORT")


def initialise_at_first_test_date(rhs, H0):
    base_sol = solve_ivp(rhs, (0, test_times[0]), H0, first_step=0.001, atol=1e-16)
    return base_sol.y[:, -1]


def one_step_household_llh(hh_data, test_times, rhs, H_t0):
    Ht = 0 * H_t0
    possible_states = np.where(np.abs(np.sum(rhs.states_inf_only, 1) - hh_data[0]) < 1e-1)[0]
    norm_factor = np.sum(H_t0[possible_states])

    if norm_factor == 0:
        # Impossible household, return large negative log-likelihood
        return -1e6

    Ht[possible_states,] = H_t0[possible_states,] / norm_factor

    tspan = (test_times[0], test_times[1])
    solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-6)

    I = hh_data[1]
    possible_states = np.where(np.abs(np.sum(rhs.states_inf_only, 1) - I) < 1e-1)[0]

    llh_value = np.sum(solution.y[possible_states, -1])

    # If sum is zero (another impossible state), penalize
    if llh_value <= 0:
        return -1e6

    return np.log(llh_value)


def one_step_population_likelihood(test_data, test_times, tau, lam, rhs, growth_rate, init_prev, R_comp):
    rhs = deepcopy(rhs)
    rhs.update_int_rate(tau)
    rhs.update_int_rate(lam)

    base_H0 = make_initial_condition_by_eigenvector(
        growth_rate, rhs.model_input, rhs.household_population, rhs, init_prev, 0.0, False, R_comp
    )
    H_t0 = initialise_at_first_test_date(rhs, base_H0)

    return sum(array([one_step_household_llh(data_i, test_times, rhs, H_t0) for data_i in test_data]))

def run_inference(multi_hh_data, rhs):
    def f(params):
        tau, lam = params
        return -one_step_population_likelihood(multi_hh_data, test_times, tau, lam, rhs, growth_rate, init_prev=pop_prev, R_comp=3)

    mle = sp.optimize.minimize(f, [tau_0, lambda_0], bounds=((0.0, 1.), (0., 10.)))
    tau_est, lambda_est = mle.x
    loglike = -mle.fun
    return tau_est, lambda_est, loglike
llh = one_step_population_likelihood(multi_hh_data,
                                     test_times,
                                     tau_0,
                                     lambda_0,
                                     rhs,
                                     growth_rate,
                                     init_prev=pop_prev,
                                     R_comp=3)
print(llh)
# tau_est, lambda_est, loglike_est = run_inference(multi_hh_data, rhs)
# print(f"🔎 Inference results: tau = {tau_est:.4f}, lambda  = {lambda_est:.4f}, loglike = {loglike_est:.4f}")