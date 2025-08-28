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
from model.specs import SINGLE_AGE_SEIR_SPEC_FOR_FITTING, SINGLE_AGE_UK_SPEC
from model.common import UnloopedSEIRRateEquations, UnloopedSEPIRRateEquations
from model.imports import FixedImportModel
from model.imports import NoImportModel

DATA_FILE = "synthetic_data_simulation_result.pkl"
DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

lambda_0 = 3.0
tau_0 = 0.25
pop_prev = 1e-3
test_times = np.array([14, 28])

def initialise_at_first_test_date(rhs, H0):
    base_sol = solve_ivp(rhs, (0, test_times[0]), H0, first_step=0.001, atol=1e-16)
    return base_sol.y[:, -1]

def one_step_household_llh(hh_data, test_times, rhs, H_t0):
    Ht = 0 * H_t0
    possible_states = where(abs(sum(rhs.states_inf_only, 1) - hh_data[0]) < 1e-1)[0]
    norm_factor = sum(H_t0[possible_states])
    if norm_factor == 0:
        raise ValueError("Initial condition normalization failed: sum zero")
    Ht[possible_states, ] = H_t0[possible_states, ] / norm_factor

    tspan = (test_times[0], test_times[1])
    solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-6)
    I = hh_data[1]
    possible_states = where(abs(sum(rhs.states_inf_only, 1) - I) < 1e-1)[0]
    return log(sum(solution.y[possible_states, -1]))

def one_step_population_likelihood(test_data, test_times, tau, lam, base_rhs, growth_rate, init_prev, R_comp):
    rhs = deepcopy(base_rhs)
    rhs.update_int_rate(tau)
    rhs.update_int_rate(lam)

    base_H0 = make_initial_condition_by_eigenvector(
        growth_rate, rhs.model_input, rhs.household_population, rhs, init_prev, 0.0, False, R_comp
    )
    H_t0 = initialise_at_first_test_date(rhs, base_H0)

    return sum(array([one_step_household_llh(data_i, test_times, rhs, H_t0) for data_i in test_data]))

def run_inference(multi_hh_data, base_rhs):
    def f(params):
        tau, lam = params
        return -one_step_population_likelihood(multi_hh_data, test_times, tau, lam, base_rhs, growth_rate, init_prev=pop_prev, R_comp=3)

    mle = sp.optimize.minimize(f, [tau_0, lambda_0], bounds=((0.0, 1.), (0., 10.)))
    tau_est, lambda_est = mle.x
    loglike = -mle.fun
    return tau_est, lambda_est, loglike

with open(DATA_FILE, "rb") as f:
    multi_hh_data, base_rhs = pickle.load(f)

tau_est, lambda_est, loglike_est = run_inference(multi_hh_data, base_rhs)
print(f"🔎 Inference results: tau = {tau_est:.4f}, lambda  = {lambda_est:.4f}, loglike = {loglike_est:.4f}")