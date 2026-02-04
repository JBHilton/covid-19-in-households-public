#Import libraries
import pickle

from copy import deepcopy
from collections import Counter
from matplotlib.pyplot import cm, subplots
from numpy import arange, array, ceil, delete, log, ones, percentile, sum, unravel_index, where, zeros
from numpy.random import choice
from os import chdir, mkdir, getcwd
from pandas import read_csv
from scipy import optimize
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm
from time import time
from tqdm import tqdm
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import SINGLE_TYPE_INFERENCE_SPEC, TWO_AGE_SEIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import SEIRRateEquations, UnloopedSEIRRateEquations
from model.imports import FixedImportModel, NoImportModel
from os.path import join
if 'inference-from-testing' in getcwd():
    chdir("../..")


# Model set up
DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

# List of observed household compositions
composition_list = read_csv(
    'inputs/eng_and_wales_adult_child_composition_list.csv',
    header=0).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/eng_and_wales_adult_child_composition_dist.csv',
    header=0).to_numpy().squeeze()


SPEC = {**TWO_AGE_SEIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
model_input_to_fit = SEIRInput(SPEC, composition_list, comp_dist)
household_population_to_fit = HouseholdPopulation(
    composition_list, comp_dist, model_input_to_fit)
rhs_to_fit = SEIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(4,2))
beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
model_input = deepcopy(model_input_to_fit)
model_input.k_ext *= beta_ext


SPEC = SINGLE_TYPE_INFERENCE_SPEC
comp_dist = array([1])
composition_list = array([[3]])

n_sims = 10
n_hh = 100
lambda_0 = 3.0
tau_0 = .25
pop_prev = 1e-3
test_times = array([14, 28])

##### DATA GENERATOR
# Initialize model input based on specifications
model_input_to_fit = SEIRInput(SPEC, composition_list, comp_dist)
'''Quick fix to make sure initial states are in form S>0, I>0 rather than S>0, E>0'''
model_input_to_fit.new_case_compartment = 2
#true_density_expo = .5 # Todo: bring this in from model input rather than defining directly

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

    # With the parameters chosen, we calculate Q_int:
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input)

    no_imports = NoImportModel(4, 1)

    base_rhs = UnloopedSEIRRateEquations(model_input, household_population, no_imports)

    x0 = array([pop_prev])

    fixed_imports = FixedImportModel(4,
                                     1,
                                     base_rhs,
                                     x0)

    rhs = UnloopedSEIRRateEquations(model_input, household_population, fixed_imports, sources="IMPORT")
    P0 = zeros(rhs.total_size, )
    P0[where(abs(rhs.states_sus_only - rhs.household_population.composition_by_state) < 1e-1)[0]] = 1.
    H_t0 = initialise_at_first_test_date(rhs,
                                  P0)

    solve_times = array([0, test_times[0], test_times[1]])
    def generate_single_hh_test_data(test_times):
        Ht = deepcopy(H_t0)
        test_data = zeros((len(test_times),))
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
multi_hh_data, sim_rhs = run_simulation(lambda_0, tau_0)

# IF YOU WANNA SAVE THE DATA
with open("simulated_test_outcomes_for_exp_llh.pkl", "wb") as f:
    pickle.dump(multi_hh_data, f)

    # Save dataset
    output_dir = "likelihood_simulation_results"
    os.makedirs(output_dir, exist_ok=True)
    pickle_path = join(output_dir, f"multi_hh_data_run_{run_idx+1}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(multi_hh_data, f)

#IF YOU WANNA USE THE SAVED DATA
with open("simulated_test_outcomes_for_exp_llh.pkl", "rb") as f:
    multi_hh_data = pickle.load(f)

#if 'inference-from-testing' in os.getcwd():
#    os.chdir("../..")
#os.getcwd()

# Load synthetic data generated in script Synthetic_data_generator.py
#pickle_path = "synthetic_data_simulation_fixed_import_results.pkl"

#with open(pickle_path, "rb") as f:
#    results = pickle.load(f)

#multi_hh_data = results["multi_hh_data"]
#rhs = results["base_rhs"]

#Likelihood Calculation

# Single Household likelihood calculation
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


# Construct models (should only be one type if we're doing uniform households)
hh_models = {}
x0 = array([pop_prev]) # Initial external prevalence
for N in range(1, max_hh_size + 1):
    comp_list = array([[N]])
    comp_dist = array([1.0])  # dummy value; only one composition
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

# Calculate mapping matrices
Chi = []
for (N_HH, y) in index_to_result:
    hh_pop, rhs, P0 = hh_models[N_HH]
    states_inf_only = rhs.states_inf_only
    # Step 3a: Find valid states for given y
    possible_states = where(abs(states_inf_only - y) < 1e-1)[0]

    # Step 3b: Build sparse Chi matrix
    size = len(states_inf_only)
    data = ones(len(possible_states))
    rowcol = possible_states
    Chi_k = csr_matrix((data, (rowcol, rowcol)), shape=(size, size))
    Chi.append(Chi_k)

# Set things up for this specific household size
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

# Convert data to frequencies
obs_counts = Counter(tuple(map(int, hh)) for hh in multi_hh_data)

def likelihood_tau_lambda(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2):
    u = Chi_1.dot(A.dot(P0))
    v = Chi_2.dot(B.dot(u))

    # Likelihood
    llh = log(sum(v)) - log(sum(u))

    return llh

# Compute the conditional log-likelihood, collapsed over unique datapoints.

hh_pop, rhs, P0 = hh_models[N_HH]
def loglike_with_counts(tau, lam, P0, Q1, Q2, Q0, Chi, result_to_index, obs_counts):
    tau = tau / rhs.model_input.beta_int # Rescale to match method from data generation
    Q_theta = (rhs.Q_int_fixed + tau * rhs.Q_int_inf + lam * rhs.Q_import)
    A = expm((t1 - t0) * Q_theta.T)
    B = expm((t2 - t1) * Q_theta.T)

    total_llh = 0.0

    for (y1, y2), count in obs_counts.items():
        # Map to Chi matrices
        k1 = result_to_index[(N_HH, y1)]
        k2 = result_to_index[(N_HH, y2)]
        Chi_1 = Chi[k1]
        Chi_2 = Chi[k2]

        # Household likelihood & gradients
        llh = likelihood_tau_lambda(A, B, P0, Q1, Q2, Q0, Chi_1, Chi_2)

        # Total likelihood / log-likelihood
        total_llh += count * llh

    return total_llh


#Likelihood surface

def f(pars):
    tau = pars[0]
    lam = pars[1]
    if (tau<=0)|(lam<=0):
        return 1e6
    else:
        return -loglike_with_counts(tau, lam, P0, Q1, Q2, Q0, Chi, result_to_index, obs_counts)

t_range = tau_0 * arange(.5, 1.55, .1)
t_start = t_range[0]
t_end = t_range[-1]
l_range = lambda_0 * arange(.5, 1.55, .1)
l_start = l_range[0]
l_end = l_range[-1]

me_start = time()
llh_array_matrix_exp = array([-f([t, l]) for t in t_range for l in l_range]).reshape(len(t_range), len(l_range))
print("LLH array with matrix exponential calculated in", time() - me_start)
mle_arg_matrix_exp = unravel_index(llh_array_matrix_exp.argmax(), llh_array_matrix_exp.shape)

fig, ax = subplots()
ax.imshow(llh_array_matrix_exp,
           cmap = cm.magma,
           origin='lower',
           extent=(l_start, l_end, t_start, t_end))
ax.scatter([lambda_0], [tau_0], color = 'k')
ax.set_aspect((l_end-l_start)/(t_end-t_start))
ax.set_title('Matrix exponential')
fig.show()