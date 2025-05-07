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
from numpy import arange, array, atleast_2d, log
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


comp_dist = read_csv(
    'inputs/england_hh_size_dist.csv',
    header=0).to_numpy().squeeze()
comp_dist = comp_dist[:3]
comp_dist[:2] *= 0
comp_dist = comp_dist/sum(comp_dist)
max_hh_size = len(comp_dist)
composition_list = np.atleast_2d(arange(1, max_hh_size+1)).T

SPEC = {**SINGLE_AGE_SEIR_SPEC_FOR_FITTING, **SINGLE_AGE_UK_SPEC}


DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

# Constants

n_sims = 10
lambda_0 = 1.0
tau_0 = 0.03

n_hh = 100 # Number of households for synth. data

# Arrays to store true and estimated values
lambda_true = np.zeros((n_sims, 1))
tau_true = np.zeros((n_sims, 1))
lambda_est = np.zeros((n_sims, 1))
tau_est = np.zeros((n_sims, 1))

#Array to initiliaze lambda and tau.
lambda_true = lognorm.rvs(s = 0.5*lambda_0, # Sets standard deviation to half of mean (or possible exp(std) to half of exp(mean))
                          loc = lambda_0, # Should set mean to lambda_0 (or possibly log(lambda_0))
                          size = len(lambda_true))
tau_true = lognorm.rvs(s = 0.5*tau_0, # Sets standard deviation to half of mean (or possible exp(std) to half of exp(mean))
                          loc = tau_0, # Should set mean to tau_0 (or possibly log(tau_0))
                          size = len(tau_true))

model_input_to_fit = SEIRInput(SPEC, composition_list, comp_dist)
#true_density_expo = .5 # Todo: bring this in from model input rather than defining directly
test_times = np.arange(7, 7 * 5, 7)


# Simulation and inference functions
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
    return multi_hh_data

test_times = np.arange(7, 7 * 5, 7)

def llh_from_test_data(test_data, test_ts, rhs, H0):
    total_sol_time = 0
    Ht = deepcopy(H0)
    llh = 0
    for i in range(len(test_times)-1):
        if i==0:
            start_time = 0
        else:
            start_time = test_times[i-1]
        tspan = (start_time, test_times[i])
        pre_sol = time.time()
        solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-16)
        total_sol_time += time.time()-pre_sol
        T = solution.t
        H = solution.y
        I = test_data[i]
        possible_states = np.where(np.abs(np.sum(rhs.states_inf_only,1)-I)<1e-1)[0]
        llh += np.log(np.sum(H[possible_states, -1]))
        Ht *= 0
        Ht[possible_states] = H[possible_states, -1]
    #print(total_sol_time)
    return(llh)

def llh_from_pars(data, test_times, tau, lam):
    pre_hh_time = time.time()
    model_input = SEIRInput(SPEC, composition_list, comp_dist, print_ests=False)
    model_input.k_home *= tau / model_input_to_fit.beta_int
    model_input.beta_int = tau
    model_input.k_ext *= lam

    true_density_expo = model_input.density_expo
    # With the parameters chosen, we calculate Q_int:
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input)
    # lam goes in as arg for rate equations

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
    rhs = UnloopedSEIRRateEquations(model_input, household_population, fixed_imports)
    H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, 1e-2, 0.0,
                                               False, 3)
    tspan = (0.0, 365)
    # print(time.time() - pre_hh_time)
    return (sum(array([llh_from_test_data(data_i, test_times, rhs, H0) for data_i in data])))

#argument to run run_inference
mhd = run_simulation(3.0, .09)

def run_inference(multi_hh_data):

    def f(params):
        tau = params[0]
        lam = params[1]
        return -llh_from_pars(multi_hh_data, test_times, tau, lam)

    mle = sp.optimize.minimize(f, [tau_0, lambda_0], bounds=((0.005, 0.15), (2., 5.)))


    ## Get MlE
    tau_est, lambda_est = mle.x[0], mle.x[1]


    return tau_est, lambda_est

# Main loop for simulations and inference
for sim in range(n_sims):
    # Generate noisy true values
    lambda_true[sim] = lambda_0 * (0.95 + 0.1 * np.random.uniform())
    tau_true[sim] = tau_0 * (0.95 + 0.1 * np.random.uniform())

    # Simulate synthetic data based on current true values
    synth_data = run_simulation(lambda_true[sim], tau_true[sim])

    # Perform inference on the synthetic data
    lambda_est[sim], tau_est[sim] = run_inference(synth_data)

# Plot results with y=x line for comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Plot lambda results
ax1.plot(lambda_true, lambda_est, "rx", label="Estimated vs True")
ax1.plot([0, max(lambda_true)], [0, max(lambda_true)], "k--", label="y=x line")
ax1.set_xlabel("True λ")
ax1.set_ylabel("Estimated λ")
ax1.legend()
ax1.set_title("True vs. Estimated λ")

# Plot tau results
ax2.plot(tau_true, tau_est, "rx", label="Estimated vs True")
ax2.plot([0, max(tau_true)], [0, max(tau_true)], "k--", label="y=x line")
ax2.set_xlabel("True τ")
ax2.set_ylabel("Estimated τ")
ax2.legend()
ax2.set_title("True vs. Estimated τ")

plt.tight_layout()
plt.show()
