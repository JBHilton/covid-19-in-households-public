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
from numpy import arange, array, atleast_2d, log, sum, where, zeros
import pandas
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
# comp_dist = read_csv(
#     'inputs/england_hh_size_dist.csv',
#     header=0).to_numpy().squeeze()
# comp_dist = comp_dist[:3]
# comp_dist[:2] *= 0
# comp_dist = comp_dist/sum(comp_dist)
comp_dist = array([1])
# max_hh_size = len(comp_dist)
# composition_list = np.atleast_2d(arange(1, max_hh_size+1)).T
composition_list = array([[3]])

# Specify model parameters
SPEC = {**SINGLE_AGE_SEIR_SPEC_FOR_FITTING, **SINGLE_AGE_UK_SPEC}
DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

# Simulation constants
n_sims = 10
lambda_0 = 3.0
tau_0 = 0.25
n_hh = 100  # Number of households for synthetic data

pop_prev = 1e-3 # Initial prevalence in simulation

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

    # H0 = np.zeros((household_population.total_size), )
    # all_sus = np.where(np.sum(rhs.states_exp_only + rhs.states_inf_only + rhs.states_rec_only, 1) < 1e-1)[0]
    # one_inf = np.where((np.abs(np.sum(rhs.states_inf_only, 1) - 1) < 1e-1) & (
    #             np.sum(rhs.states_exp_only + rhs.states_rec_only, 1) < 1e-1))[0]
    # H0[all_sus] = comp_dist
    #
    # ## Now set up evaluation time points and solve system:
    #
    # # New time at which we evaluate the infection
    # trange = np.arange(0, 7 * 5, 7)  # Evaluate for 12 weeks
    #
    # H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, 1e-1, 0.0, False, 3)


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

# Save results if flag is enabled
if SAVE_INFERENCE_RESULTS:
    results = {
        "multi_hh_data": multi_hh_data,
        "rhs": rhs
    }

    # Save file relative to current working directory
    script_dir = os.getcwd()
    save_path = os.path.join(script_dir, "synthetic_data_simulation_fixed_import_results.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"✅ Simulation results saved to {save_path}")