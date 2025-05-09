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
#Plots
###############################
def plot_results(solution, household_population, model_input):
    T = solution.t
    H = solution.y
    S = H.T.dot(household_population.states[:, 0])
    E = H.T.dot(household_population.states[:, 1])
    I = H.T.dot(household_population.states[:, 2])
    R = H.T.dot(household_population.states[:, 3])

    data_list = [S / model_input.ave_hh_by_class,
                 E / model_input.ave_hh_by_class,
                 I / model_input.ave_hh_by_class,
                 R / model_input.ave_hh_by_class]

    lgd = ['S', 'E', 'I', 'R']
    fig, axis = subplots(1, 1, sharex=True)
    cmap = plt.get_cmap('tab20')
    alpha = 0.5
    for i in range(len(data_list)):
        axis.plot(T, data_list[i], label=lgd[i], color=cmap(i / len(data_list)), alpha=alpha)
    axis.set_ylabel('Proportion of population')
    axis.legend(ncol=1, bbox_to_anchor=(1, 0.50))
    plt.savefig('outputs/Trajectories')

def plot_results(solution, household_population, model_input):
    S = H_all.T.dot(household_population.states[:, ::4])
    E = H_all.T.dot(household_population.states[:, 1::4])
    I = H_all.T.dot(household_population.states[:, 2::4])
    R = H_all.T.dot(household_population.states[:, 3::4])

    data_list = [S / model_input.ave_hh_by_class,
                 E / model_input.ave_hh_by_class,
                 I / model_input.ave_hh_by_class,
                 R / model_input.ave_hh_by_class]

    lgd = ['S', 'E', 'I', 'R']
    fig, axis = subplots(1, 1, sharex=True)
    cmap = plt.get_cmap('tab20')
    alpha = 0.5
    for i in range(len(data_list)):
        axis.plot(T, data_list[i], label=lgd[i], color=cmap(i / len(data_list)), alpha=alpha)
    axis.set_ylabel('Proportion of population')
    axis.legend(ncol=1, bbox_to_anchor=(1, 0.50))
    plt.savefig('outputs/Trajectories')
    plt.close(fig)


llh, H_all, t_all = llh_with_traj(sample_data, test_times, rhs, H0)


#Plot
def plot_population_trajectories(H_all, household_population, model_input, t_all, test_times, sample_data):
    """Plot population trajectories for S, E, I, R and overlay test data."""
    S = H_all.T.dot(household_population.states[:, ::4])
    E = H_all.T.dot(household_population.states[:, 1::4])
    I = H_all.T.dot(household_population.states[:, 2::4])
    R = H_all.T.dot(household_population.states[:, 3::4])

    data_list = [
        S / model_input.ave_hh_by_class,
        E / model_input.ave_hh_by_class,
        I / model_input.ave_hh_by_class,
        R / model_input.ave_hh_by_class
    ]

    lgd = ['S', 'E', 'I', 'R', "Test data"]

    fig, axis = plt.subplots(1, 1, sharex=True)
    cmap = get_cmap('tab20')
    alpha = 0.5

    for i in range(len(data_list)):
        axis.plot(t_all, data_list[i], label=lgd[i], color=cmap(i / len(data_list)), alpha=alpha)

    axis.plot(test_times, sample_data / model_input.ave_hh_by_class, marker=".", ls="", ms=20, label=lgd[-1])
    axis.set_ylabel('Proportion of population')
    axis.legend(ncol=1, bbox_to_anchor=(1, 0.50))

    plt.savefig('outputs/LLh with Trajectories')
    plt.close(fig)

def plot_1D_llh(tau_vals, lam_vals, llh_fixed_lam, llh_fixed_tau, tau_hat, lam_hat):
    """Plot 1D log-likelihoods for tau and lambda."""
    tau_vals = np.linspace(0.05, 0.15, 100)
    lam_vals = np.linspace(2.5, 3.5, 100)

    llh_fixed_lam = np.zeros((len(tau_vals),))
    for i in range(len(tau_vals)):
        # print("tau=",tau_vals[i])
        llh_fixed_lam[i] = llh_from_pars(sampled_households, test_times, tau_vals[i], true_lam)

    llh_fixed_tau = np.zeros((len(lam_vals),))
    for i in range(len(lam_vals)):
        # print("lam=",lam_vals[i])
        llh_fixed_tau[i] = llh_from_pars(sampled_households, test_times, model_input.beta_int, lam_vals[i])

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.plot(tau_vals, llh_fixed_lam, label='Log likelihood vs tau')
    ax1.set_xlabel("tau")
    ax1.set_ylabel("log likelihood")
    ax1.axvline(model_input.beta_int, color='k', linestyle='--', label='true value for lambda')
    ax1.axvline(tau_hat, color='r', linestyle='--', label='estimated value for lambda')
    ax1.legend(loc='lower right')

    ax2.plot(lam_vals, llh_fixed_tau, label='Log likelihood vs lambda')
    ax2.set_xlabel("lambda")
    ax2.set_ylabel("log likelihood")
    ax2.axvline(3., color='k', linestyle='--', label='true value for tau')
    ax2.axvline(lam_hat, color='r', linestyle='--', label='estimated value for tau')
    ax2.legend(loc='lower right')

    plt.savefig('outputs/1D llh')
    plt.close(fig)

def plot_2D_llh(tau_vals, lam_vals, llh_vals):
    """Plot 2D log-likelihood surface."""
    tau_vals = arange(0.060, 0.105, 0.01)
    lam_vals = arange(2., 4.0, 0.25)
    llh_vals = np.zeros((len(tau_vals), len(lam_vals)))
    for i in range(len(tau_vals)):
        for j in range(len(lam_vals)):
            llh_vals[i, j] = llh_from_pars(sampled_households, test_times, tau_vals[i], lam_vals[j])
            print("tau=", tau_vals[i], ", lam=", lam_vals[j], ", llh[tau, lam]=", llh_vals[i, j])
    with open('outputs/inference-from-testing/synth-data-gridded-par-ests.pkl', 'wb') as f:
        dump((tau_vals, lam_vals, llh_vals), f)


    fig, ax = plt.subplots(1, 1)
    lam_inc = lam_vals[1] - lam_vals[0]
    lam_max = lam_vals[-1] + lam_inc
    tau_inc = tau_vals[1] - tau_vals[0]
    tau_max = tau_vals[-1] + tau_inc

    ax.imshow(llh_vals, origin='lower',
              extent=(lam_vals[0] - 0.5 * lam_inc, lam_max - 0.5 * lam_inc,
                      tau_vals[0] - 0.5 * tau_inc, tau_max - 0.5 * tau_inc),
              aspect=(lam_max - lam_vals[0]) / (tau_max - tau_vals[0]))

    ax.set_xlabel("tau")
    ax.set_ylabel("lambda")
    ax.plot([true_lam], [model_input.beta_int], marker=".", ms=20)

    plt.savefig('outputs/2D llh')
    plt.close(fig)