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
    return multi_hh_data, base_rhs

#######################

def one_step_household_llh(hh_data,
                  test_times,
                  rhs,
                  H_t0):
    '''This function calculates the log likelihood of parameters tau and lam given some data with test results at two
    time points for a single household. This assumes data comes as number of positive tests at start and end time point,
    and that each positive test corresponds to exactly one individual in the I compartment of the model.'''

    Ht = 0 * H_t0 # Set up initial condition vector
    possible_states = where(abs(sum(rhs.states_inf_only, 1) - hh_data[0]) < 1e-1)[0]
    norm_factor = sum(H_t0[possible_states])
    if norm_factor == 0:
        raise ValueError("Initial condition normalization failed: sum zero")
    # Set Ht equal to eigenvector initial condition, conditioned on test results
    Ht[possible_states, ] = H_t0[possible_states, ] / sum(H_t0[possible_states, ])
    llh = 0

    tspan = (test_times[0], test_times[1])
    solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-6)
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
    rhs = deepcopy(base_rhs)
    rhs.update_int_rate(tau)
    rhs.update_int_rate(lam)

    base_H0 = make_initial_condition_by_eigenvector(growth_rate,
                                                    rhs.model_input,
                                                    rhs.household_population,
                                                    rhs,
                                                    init_prev,
                                                    0.0,
                                                    False,
                                                    R_comp)
    H_t0 = initialise_at_first_test_date(rhs,
                                         base_H0)
    return (sum(array([one_step_household_llh(data_i,
                  test_times,rhs,
                 H_t0) for data_i in test_data])))



# Argument to type in console to run run_inference
#multi_hh_data = run_simulation(3.0, .09)
multi_hh_data, base_rhs = run_simulation(3.0, .09)


# Run inference to estimate tau and lambda
def run_inference(multi_hh_data, base_rhs):

    def f(params):
        tau = params[0]
        lam = params[1]
        return -one_step_population_likelihood(multi_hh_data, test_times, tau, lam, base_rhs, growth_rate, init_prev=pop_prev, R_comp=3)

    mle = sp.optimize.minimize(f, [tau_0, lambda_0], bounds=((0.0, 1.), (0., 10.)))
    tau_est, lambda_est = mle.x[0], mle.x[1]
    return tau_est, lambda_est

#### Repeat fits written by JH
n_attempts = 2
tau_list = np.zeros(n_attempts)
lambda_list = np.zeros(n_attempts)
start_time = time.time()
for i in range(n_attempts):
    multi_hh_data_i, base_rhs_i = run_simulation(3.0, .09)
    tau_est_i, lambda_est_i = run_inference(multi_hh_data_i, base_rhs_i)
    tau_list[i] = tau_est_i
    lambda_list[i] = lambda_est_i
    elapsed_time = time.time() - start_time
    print("Elapsed time is",
          elapsed_time,
          "seconds ,",
          i+1,
          "iterations calculated, estimated",
          (n_attempts - (i+1)) * elapsed_time / (i+1),
          "seconds remaining.")

networked_rhs = UnloopedSEIRRateEquations(base_rhs.model_input,
                                          base_rhs.household_population,
                                          sources="BETWEEN",
                                          import_model=NoImportModel(4, 1))
# Estimate growth rates from fits:
r_list = np.zeros(n_attempts)
for i in range(n_attempts):
    rhs = deepcopy(networked_rhs)
    rhs.update_int_rate(tau_list[i])
    rhs.update_int_rate(lambda_list[i])
    r_list[i] = estimate_growth_rate(rhs.household_population,
                                     rhs,
                                     [0.001, 5],
                                     1e-9)

# Convert to doubling times for further checking:
dt_list = array([log(2)/r for r in r_list])

result_df = pandas.DataFrame({
    "sample_id" : ["sample_" + str(i) for i in range(1, len(r_list) + 1)],
    "tau_est" : tau_list,
    "lambda_est" : lambda_list,
    "r_est" : r_list,
    "dt_est" : dt_list
})

result_df = pandas.concat([result_df,
                          pandas.DataFrame({
                              "sample_id" : ["mean"],
                              "tau_est" : [tau_list.mean()],
                              "lambda_est" : [lambda_list.mean()],
                              "r_est" : [r_list.mean()],
                              "dt_est" : [dt_list.mean()]
})]).reset_index(drop=True)

true_growth_rate = estimate_growth_rate(base_rhs.household_population,
                                          networked_rhs,
                                          [0.001, 5],
                                          1e-9)
result_df = pandas.concat([result_df,
                          pandas.DataFrame({
                              "sample_id" : ["truth"],
                              "tau_est" : [0.09],
                              "lambda_est" : [3.0],
                              "r_est" : [true_growth_rate],
                              "dt_est" : [log(2) / true_growth_rate]
})]).reset_index(drop=True)

if SAVE_INFERENCE_RESULTS:
    result_df.to_csv("inference_results" + str(datetime.now()) + ".csv", index=False)


#### Runs over parameter space written by IGM:
# Check for NaNs
assert not np.isnan(H0).any(), "H0 contains NaNs"

#MCMC -sample from posterior

taus = np.linspace(0.01, 0.15, 30)
lambdas = np.linspace(2.0, 5.0, 30)
posterior = np.zeros((len(taus), len(lambdas)))

# Likelihood surface calculation

for i, tau in enumerate(tqdm(taus, desc="Outer loop")):
    for j, lam in enumerate(tqdm(lambdas, desc="Inner loop", leave=False)):
        llh = one_step_population_likelihood(multi_hh_data, test_times, tau, lam,
                                             base_rhs, growth_rate, pop_prev, R_comp=3)
        posterior[i, j] = np.exp(llh)

# Normalize posterior
posterior /= np.sum(posterior)

# Estimate marginals
tau_marginal = np.sum(posterior, axis=1)
lambda_marginal = np.sum(posterior, axis=0)

# Plot the heatmap of the posterior surface

# Create meshgrid
T, L = np.meshgrid(lambdas, taus)  # X = lambda, Y = tau

plt.figure(figsize=(8, 6))
plt.contourf(L, T, posterior, levels=50, cmap='viridis')
plt.xlabel('Lambda (External Transmission Rate)')
plt.ylabel('Tau (Internal Transmission Rate)')
plt.title('Posterior Surface (Likelihood exp(llh))')
plt.colorbar(label='Posterior Density')
plt.tight_layout()
plt.show()
plt.savefig('outputs/posterior llh')

# Number of repeated simulations/inferences
n_repeats = 5

# Store results
results = []

for i in range(n_repeats):
    print(f"Running simulation and inference {i + 1} / {n_repeats}")

    # Run simulation to generate synthetic data
    multi_hh_data, base_rhs = run_simulation(3.0, 0.09)

    # Run inference to estimate parameters
    tau_est, lambda_est = run_inference(multi_hh_data, base_rhs)

    # Store the estimates
    results.append((tau_est, lambda_est))

results_array = np.array(results)

# Print results
for i, (tau_est, lambda_est) in enumerate(results):
    print(f"Run {i + 1}: tau = {tau_est:.4f}, lambda = {lambda_est:.4f}")

###### Likelihood calculation by IGM

#Make sure to run this first:

multi_hh_data, base_rhs = run_simulation(lambda_0, tau_0)

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
for N in range(1, max_hh_size + 1):
    comp_list = np.array([[N]])
    comp_dist = np.array([1.0])  # dummy value; only one composition
    model_input = SEIRInput(SPEC, comp_list, comp_dist)
    hh_pop = HouseholdPopulation(comp_list, comp_dist, model_input)
    no_imports = NoImportModel(4, 1)
    rhs = UnloopedSEIRRateEquations(model_input, hh_pop, no_imports)
    hh_models[N] = (hh_pop, rhs)

# Step 3: Build Chi list
from scipy.sparse import csr_matrix
Chi = []
for (N_HH, y) in index_to_result:
    hh_pop, rhs = hh_models[N_HH]
    states_inf_only = rhs.states_inf_only
    # Step 3a: Find valid states for given y
    possible_states = np.where(abs(states_inf_only - y) < 1e-1)[0]

    # Step 3b: Build sparse Chi matrix
    size = len(states_inf_only)
    data = np.ones(len(possible_states))
    rowcol = possible_states
    Chi_k = csr_matrix((data, (rowcol, rowcol)), shape=(size, size))
    Chi.append(Chi_k)

# Step 3c: Print a Chi

k = 5  # or any k we want
Chi_k = Chi[k]
print(Chi_k.toarray())

#Step 4: Extract Chi for the inference
for hh_data in multi_hh_data:  # hh_data is array like [y1, y2]?
    N_HH = int(composition_list[0, 0])  # Assuming uniform household size for now

    obs1 = (N_HH, int(hh_data[0]))
    obs2 = (N_HH, int(hh_data[1]))

    k1 = result_to_index[obs1]
    k2 = result_to_index[obs2]

    Chi_1 = Chi[k1]
    Chi_2 = Chi[k2]

    print(Chi_1.toarray())
    print(Chi_2.toarray())

# Step 5: Compute the likelihood

# Step 5a: Set the ground work

# Prep/work
N_HH = int(composition_list[0, 0])  # household size
y1 = int(multi_hh_data[0][0])
y2 = int(multi_hh_data[0][1])

# Map to Chi matrices
k1 = result_to_index[(N_HH, y1)]
k2 = result_to_index[(N_HH, y2)]
Chi_1 = Chi[k1]
Chi_2 = Chi[k2]

# Initial distribution for this household type
P0 = hh_models[N_HH][0].initial_distribution

# Rate equation object for this household type
rhs = hh_models[N_HH][1]
Q1 = rhs.Q_int
Q2 = rhs.Q_ext
Q0 = rhs.Q_int_fixed  ### Q_int_fixed or Q_int_inf?

# Times
t0 = 0.0
t1 = test_times[0]
t2 = test_times[1]

def likelihood_tau_lambda(tau, lam, t0, t1, t2, P0, Q1, Q2, Q0, Chi_1, Chi_2):


    # Ingridients
    Q_theta = tau * Q1 + lam * Q2 + Q0
    A = expm((t1 - t0) * Q_theta)
    B = expm((t2 - t1) * Q_theta)
    u = Chi_1 @ (A @ P0)
    v = Chi_2 @ (B @ u)

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