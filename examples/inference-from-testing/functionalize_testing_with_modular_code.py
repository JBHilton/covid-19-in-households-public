import numpy as np
import matplotlib.pyplot as plt
import os
import time
from numpy import arange, array, atleast_2d, log
from numpy.random import choice
from numpy.linalg import eig
from scipy.integrate import solve_ivp
from scipy import optimize
from pandas import read_csv
from matplotlib.pyplot import subplots, savefig
from copy import deepcopy
from pickle import dump, load

# Import necessary components from the model
from model.preprocessing import (
    estimate_beta_ext, estimate_growth_rate,
    SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector
)
from model.specs import SINGLE_AGE_SEIR_SPEC_FOR_FITTING, SINGLE_AGE_UK_SPEC
from model.common import UnloopedSEIRRateEquations, UnloopedSEPIRRateEquations
from model.imports import FixedImportModel, NoImportModel

# Set a random seed for reproducibility
np.random.seed(637)

# Constants
DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME
TRUE_LAM = 3.0

# Create output directories if they do not exist
def create_output_directories():
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    if not os.path.isdir('outputs/inference-from-testing'):
        os.mkdir('outputs/inference-from-testing')

# Load and prepare the composition distribution
def load_composition_distribution():
    comp_dist = read_csv('inputs/england_hh_size_dist.csv', header=0).to_numpy().squeeze()
    comp_dist = comp_dist[:3]
    comp_dist[:2] *= 0  # Set the first two entries to 0
    return comp_dist / np.sum(comp_dist)

# Initialize the model input and household population
def initialize_model(comp_dist):
    max_hh_size = len(comp_dist)
    composition_list = np.atleast_2d(arange(1, max_hh_size + 1)).T
    SPEC = {**SINGLE_AGE_SEIR_SPEC_FOR_FITTING, **SINGLE_AGE_UK_SPEC}
    base_sitp = SPEC["SITP"]
    SPEC["SITP"] = 1 - (1 - base_sitp) ** 3
    model_input_to_fit = SEIRInput(SPEC, composition_list, comp_dist)
    household_population_to_fit = HouseholdPopulation(
        composition_list, comp_dist, model_input_to_fit)
    model_input = deepcopy(model_input_to_fit)
    model_input.k_ext *= true_lam

    true_density_expo = model_input.density_expo

    # With the parameters chosen, we calculate Q_int:
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input)

    pop_prev = 1e-2


    return model_input, household_population

# Set up the initial conditions
def setup_initial_conditions(model_input, household_population):
    # Create the no imports model and rate equations
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
    # H0[one_inf] = 0.01 * comp_dist
    S0 = H0.T.dot(household_population.states[:, ::4])
    E0 = H0.T.dot(household_population.states[:, 1::4])
    I0 = H0.T.dot(household_population.states[:, 2::4])
    R0 = H0.T.dot(household_population.states[:, 3::4])
    start_state = (1 / model_input.ave_hh_size) * array([S0.sum(),
                                                         E0.sum(),
                                                         I0.sum(),
                                                         R0.sum()])
    return rhs, base_H0

# Solve the SEIR model
def solve_seir_model(rhs, H0, evaluation_times):
    tspan = [evaluation_times[0], evaluation_times[-1]]
    solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16, t_eval=evaluation_times)
    return solution

# Plot the results
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

# Generate synthetic test data for a single household

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, 1e-1, 0.0,False,3)
test_times = np.arange(7,7*5,7)

def generate_single_hh_test_data(H0, test_times, rhs):
    Ht = deepcopy(H0)
    test_data = np.zeros((len(test_times),))
    for i in range(len(test_times) - 1):
        tspan = (test_times[i], test_times[i + 1])
        solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-16)
        T = solution.t
        H = solution.y
        state = choice(range(len(solution.y[:, -1])), 1, p=solution.y[:, -1] / np.sum(solution.y[:, -1]))
        test_data[i] = rhs.states_inf_only[state]
        Ht *= 0
        Ht[state] = 1
    return test_data

sample_data = generate_single_hh_test_data(test_times)
print(sample_data)

# Calculate log likelihood from test data
def llh_from_test_data(test_data, test_times, rhs, H0):
    Ht = deepcopy(H0)
    llh = 0
    for i in range(len(test_times) - 1):
        start_time = 0 if i == 0 else test_times[i - 1]
        tspan = (start_time, test_times[i])
        solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-16)
        I = test_data[i]
        possible_states = np.where(np.abs(np.sum(rhs.states_inf_only, 1) - I) < 1e-1)[0]
        llh += np.log(np.sum(solution.y[possible_states, -1]))
        Ht *= 0
        Ht[possible_states] = solution.y[possible_states, -1]
    return llh

# Calculate log likelihood with trajectory return
def llh_with_traj(test_data, test_times, rhs, H0):
    Ht = deepcopy(H0)
    H_all = np.atleast_2d(deepcopy(H0)).T
    t_all = np.array(0)
    llh = 0
    for i in range(len(test_times)):
        start_time = 0 if i == 0 else test_times[i - 1]
        tspan = (start_time, test_times[i])
        solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-16)
        I = test_data[i]
        possible_states = np.where(np.abs(np.sum(rhs.states_inf_only, 1) - I) < 1e-1)[0]
        llh += np.log(np.sum(solution.y[possible_states, -1]))
        Ht *= 0
        Ht[possible_states] = solution.y[possible_states, -1]
        Ht /= np.sum(Ht)
        H_all = np.hstack((H_all, solution.y))
        t_all = np.hstack((t_all, solution.t))
    return llh, H_all, t_all

# Plot the results
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


# Generate test data:
def generate_or_load_data(test_times, file_path='outputs/inference-from-testing/synthetic-testing-data.pkl', n_hh=1000):
    """Generate or load synthetic household test data."""
    if isfile(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        multi_hh_data = [generate_single_hh_test_data(test_times) for _ in range(n_hh)]
        with open(file_path, 'wb') as f:
            pickle.dump(multi_hh_data, f)
        return multi_hh_data

#Make a partial sample of the population.
#def sample_households(multi_hh_data, m=100, seed=42):
#    """Randomly sample households from the generated data."""
#    np.random.seed(seed)  # For reproducibility
#    sample_idx = np.random.choice(range(len(multi_hh_data)), m)
#    return [multi_hh_data[i] for i in sample_idx]


def llh_from_pars(data, test_times, tau, lam):
    """Calculate the log-likelihood for given parameters."""
    model_input = SEIRInput(SPEC, composition_list, comp_dist, print_ests=False)
    model_input.k_home = (tau / model_input.beta_int) * model_input.k_home
    model_input.k_ext = 0 * model_input.k_ext
    model_input.density_expo = true_density_expo

    household_population = HouseholdPopulation(composition_list, comp_dist, model_input)
    rhs = SEIRRateEquations(model_input, household_population, fixed_imports)
    H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, 1e-2, 0.0, False, 3)

    return sum(array([llh_from_test_data(data, test_times, rhs, H0) for data in sampled_households]))

llh_from_pars(sampled_households, test_times, 0.015, 2.75)

## Use a root finder approach to get the MLEs.
def get_tau_lam_mles(data, test_times, tau_0, lam_0):
    """Find maximum likelihood estimates for tau and lambda."""

    def f(params):
        tau, lam = params
        return -llh_from_pars(data, test_times, tau, lam)

    return sp.optimize.minimize(f, [tau_0, lam_0], bounds=((0.005, 0.15), (2., 5.)))

start_time = time.time()
mle = get_tau_lam_mles(sampled_households, test_times, 0.02, 2.5)
end_time = time.time()
tau_hat, lam_hat = mle.x[0], mle.x[1]
print("Parameter estimation takes", (end_time - start_time)/60,"minutes.")
print("Optimised in", mle.nit, "iterations.")
print("MLE of tau=", tau_hat)
print("MLE of lam=", lam_hat)


def plot_1D_llh(tau_vals, lam_vals, llh_fixed_lam, llh_fixed_tau, tau_hat, lam_hat):
    """Plot 1D log-likelihoods for tau and lambda."""
    tau_vals = arange(0.050, 0.18, 0.01)
    lam_vals = arange(2., 4.0, 0.25)

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
