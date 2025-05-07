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
test_times = np.arange(7, 7 * 5, 7)


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
    return multi_hh_data


# Function to compute log-likelihood from test data
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


# Function to calculate log-likelihood from parameter values
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


# Argument to type in console to run run_inference
mhd = run_simulation(3.0, .09)


# Run inference to estimate tau and lambda
def run_inference(multi_hh_data):

    def f(params):
        tau = params[0]
        lam = params[1]
        return -llh_from_pars(multi_hh_data, test_times, tau, lam)

    mle = sp.optimize.minimize(f, [tau_0, lambda_0], bounds=((0.0, 0.15), (2., 5.)))
    tau_est, lambda_est = mle.x[0], mle.x[1]
    return tau_est, lambda_est

# Run run_inference
run_inference(mhd)

###############################
#Sensitivity analysis
###############################

# Initialize arrays to store true and estimated values
lambda_true = np.zeros(n_sims)
tau_true = np.zeros(n_sims)
lambda_est = np.zeros(n_sims)
tau_est = np.zeros(n_sims)

# Perform sensitivity analysis
for sim in range(n_sims):
    # Generate true values by introducing noise around lambda_0 and tau_0
    lambda_true[sim] = lambda_0 * (1 + 0.1 * np.random.uniform(-1, 1))  # ±10% noise
    tau_true[sim] = tau_0 * (1 + 0.1 * np.random.uniform(-1, 1))        # ±10% noise

    # Simulate synthetic data using true parameters
    synthetic_data = run_simulation(lambda_true[sim], tau_true[sim])

    # Estimate parameters from synthetic data
    tau_est[sim], lambda_est[sim] = run_inference(synthetic_data)


# Save results to a pickle file
fname = "outputs/inference-from-testing/sensitivity_results_hh_size" + str(max_hh_size) + "_tau_" + str(tau_0) + ".pkl"
with open(fname, "wb") as f:
    pickle.dump({"lambda_true": lambda_true, "tau_true": tau_true,
                 "lambda_est": lambda_est, "tau_est": tau_est}, f)

# Scatter plot for λ and τ
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# λ: True vs. Estimated
ax1.scatter(lambda_true, lambda_est, color="red", alpha=0.7, label="Estimated vs True")
ax1.plot([min(lambda_true), max(lambda_true)], [min(lambda_true), max(lambda_true)],
         "k--", label="y=x line")
ax1.set_xlabel("True λ", fontsize=12)
ax1.set_ylabel("Estimated λ", fontsize=12)
ax1.legend()
ax1.set_title("True vs. Estimated λ", fontsize=14)

# τ: True vs. Estimated
ax2.scatter(tau_true, tau_est, color="blue", alpha=0.7, label="Estimated vs True")
ax2.plot([min(tau_true), max(tau_true)], [min(tau_true), max(tau_true)],
         "k--", label="y=x line")
ax2.set_xlabel("True τ", fontsize=12)
ax2.set_ylabel("Estimated τ", fontsize=12)
ax2.legend()
ax2.set_title("True vs. Estimated τ", fontsize=14)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("outputs/sensitivity_results_hh_size" + str(max_hh_size) + "_tau_" + str(tau_0) + ".png")

###############################
#multiple_lambda_tau_testing_results
###############################

# Define ranges for lambda and tau
lambda_values = np.linspace(0.8 * lambda_0, 1.2 * lambda_0, 5)  # 5 values around λ_0
tau_values = np.linspace(0.8 * tau_0, 1.2 * tau_0, 5)           # 5 values around τ_0

# Arrays to store results
lambda_true = []
tau_true = []
lambda_est = []
tau_est = []

# Loop over all combinations of lambda and tau
for lambda_val in lambda_values:
    for tau_val in tau_values:
        # Store true values
        lambda_true.append(lambda_val)
        tau_true.append(tau_val)

        # Run simulation for current lambda and tau
        synthetic_data = run_simulation(lambda_val, tau_val)

        # Perform inference
        tau_hat, lambda_hat = run_inference(synthetic_data)

        # Store estimated values
        lambda_est.append(lambda_hat)
        tau_est.append(tau_hat)

# Convert results to numpy arrays for plotting
lambda_true = np.array(lambda_true)
tau_true = np.array(tau_true)
lambda_est = np.array(lambda_est)
tau_est = np.array(tau_est)

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# λ: True vs. Estimated
ax1.scatter(lambda_true, lambda_est, color="red", alpha=0.7, label="Estimated vs True")
ax1.plot([min(lambda_true), max(lambda_true)], [min(lambda_true), max(lambda_true)],
         "k--", label="y=x line")
ax1.set_xlabel("True λ", fontsize=12)
ax1.set_ylabel("Estimated λ", fontsize=12)
ax1.legend()
ax1.set_title("True vs. Estimated λ for Multiple Values", fontsize=14)

# τ: True vs. Estimated
ax2.scatter(tau_true, tau_est, color="blue", alpha=0.7, label="Estimated vs True")
ax2.plot([min(tau_true), max(tau_true)], [min(tau_true), max(tau_true)],
         "k--", label="y=x line")
ax2.set_xlabel("True τ", fontsize=12)
ax2.set_ylabel("Estimated τ", fontsize=12)
ax2.legend()
ax2.set_title("True vs. Estimated τ for Multiple Values", fontsize=14)

# Final adjustments and save the figure
plt.tight_layout()
plt.savefig("outputs/multiple_lambda_tau_testing_results.png")

###############################
#nhh_sensitivity_analysis
###############################

# Define household counts to test
n_hh_values = [10, 100, 200, 500, 1000]

# True values of lambda and tau
lambda_fixed = lambda_0
tau_fixed = tau_0

# Arrays to store results
n_hh_tested = []
lambda_est = []
tau_est = []

# Loop over different values of n_hh
for n_hh in n_hh_values:
    # Store the current n_hh
    n_hh_tested.append(n_hh)

    # Update global n_hh for this run
    synthetic_data = []
    for _ in range(n_hh):
        synthetic_data.extend(run_simulation(lambda_fixed, tau_fixed))

    # Perform inference
    tau_hat, lambda_hat = run_inference(synthetic_data)

    # Store estimated values
    lambda_est.append(lambda_hat)
    tau_est.append(tau_hat)

# Convert results to numpy arrays for plotting
n_hh_tested = np.array(n_hh_tested)
lambda_est = np.array(lambda_est)
tau_est = np.array(tau_est)

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# λ: Estimated vs. Number of Households
ax1.plot(n_hh_tested, lambda_est, "o-", label="Estimated λ", color="red")
ax1.axhline(lambda_fixed, color="k", linestyle="--", label="True λ")
ax1.set_xlabel("Number of Households (n_hh)", fontsize=12)
ax1.set_ylabel("Estimated λ", fontsize=12)
ax1.legend()
ax1.set_title("Effect of n_hh on Estimated λ", fontsize=14)

# τ: Estimated vs. Number of Households
ax2.plot(n_hh_tested, tau_est, "o-", label="Estimated τ", color="blue")
ax2.axhline(tau_fixed, color="k", linestyle="--", label="True τ")
ax2.set_xlabel("Number of Households (n_hh)", fontsize=12)
ax2.set_ylabel("Estimated τ", fontsize=12)
ax2.legend()
ax2.set_title("Effect of n_hh on Estimated τ", fontsize=14)

# Final adjustments and save the figure
plt.tight_layout()
plt.savefig("outputs/nhh_sensitivity_analysis.png")
plt.show()
