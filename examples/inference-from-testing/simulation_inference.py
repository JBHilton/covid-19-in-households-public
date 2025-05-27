# Import required libraries
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, savefig
from matplotlib.cm import get_cmap
from pickle import dump, load
import scipy as sp
from scipy.integrate import solve_ivp
import time
from tqdm import tqdm
from scipy.optimize import minimize
import scipy as sp
from scipy.integrate import solve_ivp
from numpy import linalg as LA
import pickle
from numpy.linalg import eig

# Adjust for Windows file path if necessary
import os

if 'inference-from-testing' in os.getcwd():
    os.chdir("../..")
os.getcwd()

# Import functions from the project code
from copy import deepcopy
from numpy import arange, array, atleast_2d, log, where
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import SINGLE_AGE_SEIR_SPEC_FOR_FITTING, SINGLE_AGE_UK_SPEC
from model.common import UnloopedSEIRRateEquations, UnloopedSEPIRRateEquations
from model.imports import FixedImportModel
from model.imports import NoImportModel
from scipy.stats import lognorm

# Load and process demographic data
comp_dist = read_csv(
    'inputs/england_hh_size_dist.csv',
    header=0).to_numpy().squeeze()
comp_dist = comp_dist[:3]
comp_dist[:2] *= 0
comp_dist = comp_dist/sum(comp_dist)
max_hh_size = len(comp_dist)
composition_list = np.atleast_2d(arange(1, max_hh_size+1)).T

# Specify model parameters
SPEC = {**SINGLE_AGE_SEIR_SPEC_FOR_FITTING, **SINGLE_AGE_UK_SPEC}
DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

# Simulation constants
n_sims = 10
lambda_0 = 3.0
tau_0 = 0.09
n_hh = 100  # Number of households for synthetic data

# Initialize model input based on specifications
model_input_to_fit = SEIRInput(SPEC, composition_list, comp_dist)
#true_density_expo = .5 # Todo: bring this in from model input rather than defining directly
test_times = np.array([7, 14, 21])


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

    pop_prev = 1e-2

    no_imports = NoImportModel(4, 1)

    base_rhs = UnloopedSEIRRateEquations(model_input, household_population, no_imports)
    base_H0 = make_initial_condition_by_eigenvector(growth_rate,
                                                    model_input,
                                                    household_population,
                                                    base_rhs,
                                                    1e-1,
                                                    0.0,
                                                    False,
                                                    3)
    x0 = base_H0.T.dot(household_population.states[:, 1::4])

    fixed_imports = FixedImportModel(4,
                                     1,
                                     base_rhs,
                                     x0)

    rhs = UnloopedSEIRRateEquations(model_input, household_population, fixed_imports, sources="IMPORT")

    H0 = np.zeros((household_population.total_size), )
    all_sus = np.where(np.sum(rhs.states_exp_only + rhs.states_inf_only + rhs.states_rec_only, 1) < 1e-1)[0]
    one_inf = np.where((np.abs(np.sum(rhs.states_inf_only, 1) - 1) < 1e-1) & (
                np.sum(rhs.states_exp_only + rhs.states_rec_only, 1) < 1e-1))[0]
    H0[all_sus] = comp_dist

    ## Now set up evaluation time points and solve system:

    # New time at which we evaluate the infection
    trange = np.arange(0, 7 * 5, 7)  # Evaluate for 12 weeks

    H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, 1e-1, 0.0, False, 3)



    def generate_single_hh_test_data(test_times):
        Ht = deepcopy(H0)
        test_data = np.zeros((len(test_times),))
        for i in range(len(test_times) - 1):
            tspan = (test_times[i], test_times[i + 1])
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
    return multi_hh_data, base_rhs

def one_step_household_llh(hh_data,
                  test_times,
                  tau,
                  lam,
                  base_rhs,
                 growth_rate,
                 init_prev,
                 R_comp):
    '''This function calculates the log likelihood of parameters tau and lam given some data with test results at two
    time points for a single household. This assumes data comes as number of positive tests at start and end time point,
    and that each positive test corresponds to exactly one individual in the I compartment of the model.'''
    rhs = deepcopy(base_rhs)
    rhs.int_rate *= tau
    rhs.ext_rate *= lam

    H0 = make_initial_condition_by_eigenvector(growth_rate,
                                               rhs.model_input,
                                               rhs.household_population,
                                               rhs,
                                               init_prev,
                                               0.0,
                                               False,
                                               R_comp)
    Ht = 0 * H0 # Set up initial condition vector
    possible_states = where(abs(sum(rhs.states_inf_only, 1) - hh_data[0]) < 1e-1)[0]
    # Set Ht equal to eigenvector initial condition, conditioned on test results
    Ht[possible_states, ] = H0[possible_states, ] / sum(H0[possible_states, ])
    llh = 0

    tspan = (test_times[0], test_times[1])
    solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-16)
    T = solution.t
    H = solution.y
    I = hh_data[1]
    possible_states = where(abs(sum(rhs.states_inf_only,1)-I)<1e-1)[0]
    llh += log(sum(H[possible_states, -1]))
    return(llh)

def one_step_population_likelihood(test_data,
                  test_times,
                  tau,
                  lam,
                  base_rhs,
                 growth_rate,
                 init_prev,
                 R_comp):
    '''This is a wrapper for one_step_household_llh function, which allows for the calculation of the joint likelihood
    of an entire population's worth of independent one-step testing samples in a single function call.'''
    return (sum(array([one_step_household_llh(data_i,
                  test_times,
                  tau,
                  lam,
                  base_rhs,
                 growth_rate,
                 init_prev,
                 R_comp) for data_i in test_data])))

# Argument to type in console to run run_inference
#multi_hh_data = run_simulation(3.0, .09)
multi_hh_data, base_rhs = run_simulation(3.0, .09)


# Run inference to estimate tau and lambda
def run_inference(multi_hh_data, base_rhs):

    def f(params):
        tau = params[0]
        lam = params[1]
        return -one_step_population_likelihood(multi_hh_data, test_times, tau, lam, base_rhs, growth_rate, init_prev=1e-2, R_comp=3)

    mle = sp.optimize.minimize(f, [tau_0, lambda_0], bounds=((0.0, 0.15), (2., 5.)))
    tau_est, lambda_est = mle.x[0], mle.x[1]
    return tau_est, lambda_est

# Run run_inference
#run_inference(multi_hh_data)
run_inference(multi_hh_data, base_rhs)

