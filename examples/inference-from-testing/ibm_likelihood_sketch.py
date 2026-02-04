
from copy import deepcopy
from matplotlib.pyplot import cm, subplots
from numpy import (arange, array, ceil, delete, log, meshgrid, ones, percentile, sum, unique,
                   unravel_index, where, zeros)
from numpy.random import choice
from os import chdir, getcwd
from pandas import read_csv
from scipy import optimize
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm
from time import time
from tqdm import tqdm
from model.preprocessing import ( SEIRInput, HouseholdPopulation)
from model.specs import SINGLE_TYPE_INFERENCE_SPEC, LATENT_PERIOD, PRODROME_PERIOD, SYMPTOM_PERIOD
from model.common import UnloopedSEIRRateEquations
from model.imports import FixedImportModel, NoImportModel

# pylint: disable=invalid-name

if 'inference-from-testing' in getcwd():
    chdir("../..")

beta_int = SINGLE_TYPE_INFERENCE_SPEC["beta_int"]
beta_ext = 1.
x0 = [.1]
t_eval = arange(0, 361., 30.)


# Function for constructing individual-based model of a single household with homogeneous mixing:
def build_indiv_system(hh_size):
    composition_list = array([[1] * hh_size])
    # Proportion of households which are in each composition
    comp_dist = array([1.])

    SPEC = {
        'compartmental_structure': 'SEIR',  # This is which subsystem key to use
        'beta_int': beta_int,  # Internal infection rate
        'density_expo': 1.,
        'recovery_rate': 1 / (PRODROME_PERIOD +
                              SYMPTOM_PERIOD),  # Recovery rate
        'incubation_rate': 1 / LATENT_PERIOD,  # E->I incubation rate
        'sus': array([1] * hh_size),  # Relative susceptibility by
        # age/vulnerability class
        'fit_method': 'EL',
        'k_home': (1 / hh_size) * ones((hh_size, hh_size), dtype=float),
        'k_ext': ones((hh_size, hh_size), dtype=float),
        'skip_ext_scale': True
    }

    model_input = SEIRInput(SPEC, composition_list, comp_dist)
    model_input.k_ext *= beta_ext

    # With the parameters chosen, we calculate Q_int:
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input)

    # Set up import model
    no_imports = NoImportModel(4, hh_size)
    base_rhs = UnloopedSEIRRateEquations(model_input, household_population, no_imports)
    fixed_imports = FixedImportModel(4,
                                     hh_size,
                                     base_rhs,
                                     (1 / hh_size) * array(x0 * hh_size))

    rhs = UnloopedSEIRRateEquations(model_input,
                                    household_population,
                                    fixed_imports,
                                    sources="IMPORT")
    return (SPEC, model_input, household_population, rhs)


# Build systems for a range of household sizes:
hh_size_list = [3]
max_hh_size = hh_size_list[-1]
SPEC_LIST = []
model_input_list = []
household_population_list = []
rhs_list = []
P0_list = []
build_time_list = []
for hh_size in hh_size_list:
    build_start = time()
    (SPEC, model_input, household_population, rhs) = build_indiv_system(hh_size)
    SPEC_LIST.append(SPEC)
    model_input_list.append(model_input)
    household_population_list.append(household_population)
    rhs_list.append(rhs)
    P0 = zeros(rhs.total_size, )
    P0[where(sum(abs(rhs.states_sus_only - rhs.household_population.composition_by_state),1 ) < 1e-1)[0]] = 1.
    P0_list.append(P0)
    build_time = time() - build_start
    build_time_list.append(build_time)
    print("Construction for hh size",
          hh_size,
          "took",
          build_time,
          "seconds.")

# How do likelihood estimates differ between ODE solves and matrix exponentials?

# Model setup
comp_dist = read_csv('inputs/england_hh_size_dist.csv',header=0).to_numpy().squeeze()
comp_dist = array([comp_dist[h - 1] for h in hh_size_list])
comp_dist = comp_dist / sum(comp_dist)

SPEC = SINGLE_TYPE_INFERENCE_SPEC
DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

n_sims = 10
lambda_0 = 3.
tau_0 = .25
n_hh = 100
pop_prev = 1e-3
test_times = array([14, 28])

##### DATA GENERATOR

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
    H_t0_list = []
    temp_rhs_list = []
    for i in range(len(hh_size_list)):
        model_input = deepcopy(model_input_list[i])
        model_input.k_home *= tau_val / model_input.beta_int
        model_input.beta_int = tau_val
        model_input.k_ext *= lambda_val
        household_population = HouseholdPopulation(
            model_input.composition_list, model_input.composition_distribution, model_input)

        no_imports = NoImportModel(4, hh_size_list[i])

        base_rhs = UnloopedSEIRRateEquations(model_input, household_population, no_imports)

        x0 = array(hh_size_list[i] * [pop_prev])

        fixed_imports = FixedImportModel(4,
                                         hh_size_list[i],
                                         base_rhs,
                                         x0)

        rhs = UnloopedSEIRRateEquations(model_input, household_population, fixed_imports, sources="IMPORT")
        P0 = zeros(rhs.total_size, )
        P0[where(sum(abs(rhs.states_sus_only - rhs.household_population.composition_by_state), 1) < 1e-1)[0]] = 1.
        H_t0_list.append(initialise_at_first_test_date(rhs,
                                                     P0))
        temp_rhs_list.append(rhs)

    solve_times = array([0, test_times[0], test_times[1]])
    def generate_single_hh_test_data(test_times):
        hh_idx = choice(range(len(hh_size_list)), 1, p=comp_dist)[0]
        Ht = deepcopy(H_t0_list[hh_idx])
        rhs = temp_rhs_list[hh_idx]
        test_data = zeros((len(test_times), hh_size_list[hh_idx]))
        for i in range(len(test_times)):
            tspan = (solve_times[i], solve_times[i+1])
            solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-16)
            H = solution.y
            state = choice(range(len(H[:, -1])), 1, p=H[:, -1] / sum(H[:, -1]))[0]
            test_data[i, :] = rhs.states_inf_only[state]
            Ht *= 0
            Ht[state] = 1
        return (test_data.astype(int))

    ## Now do multiple households

    # Generate test data:

    multi_hh_data = [generate_single_hh_test_data(test_times) for i in range(n_hh)]
    return multi_hh_data

# Make simulation data:
multi_hh_data = run_simulation(lambda_0, tau_0)

# Get flattened version
multi_hh_data_flattened = [hhd.flatten() for hhd in multi_hh_data]

######################################

# Convert data to frequencies
obs_outcomes, obs_counts = unique(array(multi_hh_data), axis=0, return_counts = True)

def create_result_mappings(max_hh_size):
    result_to_index = {}  # List: (N_HH, y) → index
    index_to_result = []  # List: index → (N_HH, y)
    for N in range(1, max_hh_size + 1):     # Household sizes

        # The following line creates an array of all possible combinations of test results
        # for a given household size:
        possible_results = array(meshgrid(*[array([0, 1]) for i in range(N)])).reshape(N, 2**N)
        # Convert to a list
        ylist = [possible_results[:, i] for i in range(2**N)]

        for y in ylist:           # Number of positives in that household
            k = len(index_to_result)
            result_to_index[(N, tuple(y))] = k
            index_to_result.append((N, y))

    return result_to_index, index_to_result

result_to_index, index_to_result = create_result_mappings(max_hh_size)

# Construct models (should only be one type if we're doing uniform households)
hh_models = {}
for N in range(1, max_hh_size + 1):
    (SPEC_N, model_input_N, household_population_N, rhs_N) = build_indiv_system(N)
    P0_N = zeros(rhs_N.total_size, )
    P0_N[where(abs(sum(rhs_N.states_sus_only, 1) - N) < 1e-1)[0]] = 1.
    hh_models[N] = (household_population_N, rhs_N, P0_N)

# Calculate mapping matrices
Chi = []
for (N_HH, y) in index_to_result:
    hh_pop, rhs, P0 = hh_models[N_HH]
    states_inf_only = rhs.states_inf_only
    # Step 3a: Find valid states for given y
    possible_states = where(sum(abs(states_inf_only - y), 1) < 1e-1)[0]

    # Step 3b: Build sparse Chi matrix
    size = len(states_inf_only)
    data = ones(len(possible_states))
    rowcol = possible_states
    Chi_k = csr_matrix((data, (rowcol, rowcol)), shape=(size, size))
    Chi.append(Chi_k)

# Set things up for this specific household size
N_HH = 3  # household size
y1 = tuple(multi_hh_data[0][0])
y2 = tuple(multi_hh_data[0][1])

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

    for i in range(len(obs_counts)):
        y1 = tuple(obs_outcomes[i][0, :])
        y2 = tuple(obs_outcomes[i][1, :])
        count = obs_counts[i]

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

# Likelihood function with ODE solve:
def likelihood_tau_lambda_ode(rhs, H_t0, Chi_1, Chi_2):
    H_t1 = Chi_1.dot(H_t0)
    H_t1 = H_t1 / H_t1.sum()
    tspan = (t1, t2)
    solution = solve_ivp(rhs, tspan, H_t1, first_step=0.001, atol=1e-16)
    H = solution.y
    Ht_2 = Chi_2.dot(H[:, -1])

    # Likelihood
    llh = log(Ht_2.sum())

    return llh

def loglike_with_counts_odes(tau, lam, obs_counts):
    tau = tau / rhs.model_input.beta_int  # Rescale to match method from data generation

    # Set up model:
    rhs_tau_lam = deepcopy(rhs)
    rhs_tau_lam.update_int_rate(tau)
    rhs_tau_lam.update_ext_rate(lam)
    base_sol = solve_ivp(rhs_tau_lam,
                         (t0, t1),
                         P0,
                         first_step=0.001,
                         t_eval= (t0, t1),
                         atol=1e-16)
    H_t0 = base_sol.y[:, -1]
    total_llh = 0.0

    for i in range(len(obs_counts)):
        y1 = tuple(obs_outcomes[i][0, :])
        y2 = tuple(obs_outcomes[i][1, :])
        count = obs_counts[i]

        # Map to Chi matrices
        k1 = result_to_index[(N_HH, y1)]
        k2 = result_to_index[(N_HH, y2)]
        Chi_1 = Chi[k1]
        Chi_2 = Chi[k2]
        # Household likelihood & gradients
        llh = likelihood_tau_lambda_ode(rhs_tau_lam, H_t0, Chi_1, Chi_2)

        # log-likelihood
        total_llh += count * llh

    return total_llh


# Do some numerical investigation of likelihood functions


# Compare solve times for matrix exponential and ODEs at this scale:

rhs = hh_models[N_HH][1]

tspan = (0.0, 28.5)
t_eval = arange(0., 28.5, 14.)
def do_matrix_solve(t_eval, H0):
    linear_sol = [H0]
    for i in range(1, len(t_eval)):
        linear_sol += [expm((t_eval[i] - t_eval[i - 1]) * (rhs.Q_int_fixed.T + rhs.Q_int_inf.T + rhs.Q_import.T)).dot(linear_sol[i - 1])]
    return linear_sol

t_ode = 0
t_matrix = 0

n_attempt = 1
for i in range(n_attempt):
    t = time()
    solution = solve_ivp(rhs, tspan, P0, first_step=0.001, atol=1e-16, t_eval=t_eval)
    t_ode += time() - t
    t = time()
    linear_sol = do_matrix_solve(t_eval, P0)
    t_matrix += time() - t

print("Ave. execution time for ODE solves in likelihood:", t_ode / n_attempt)
print("Ave. execution time for matrix solves in likelihood:", t_matrix / n_attempt)

# Accuracy at this scale:
solution = solve_ivp(rhs, tspan, P0, first_step=0.001, atol=1e-16, t_eval=t_eval)
linear_sol = [P0]
for i in range(1, len(t_eval)):
    linear_sol += [expm((t_eval[i] -t_eval[i-1]) * (rhs.Q_int_fixed.T + rhs.Q_int_inf.T + rhs.Q_import.T)).dot(linear_sol[i-1])]

errs = [(abs(solution.y[where(solution.y[:, i]>1e-9)[0], i] - linear_sol[i][where(solution.y[:, i]>1e-9)[0]]) /
         solution.y[where(solution.y[:, i]>1e-9)[0], i]).sum() for i in range(len(t_eval))]

print("Errors in likelihood matrix calculations:", errs)

# Check we get to same P(t1) (or H(t1)):
H_t1_ode = initialise_at_first_test_date(rhs,
                                          P0)
H_t1_matrix = expm((t2 - t1) * (rhs.Q_int_fixed.T + rhs.Q_int_inf.T + rhs.Q_import.T)).dot(P0)

print("Error in time point 1 evaluation:",
      (abs(H_t1_ode - H_t1_matrix) / H_t1_ode).sum())

# Check solver used for simulations against solver used for likelihood:

sim_model_input = deepcopy(rhs.model_input)
sim_model_input.k_home *= tau_0 / sim_model_input.beta_int
sim_model_input.beta_int = tau_0
sim_model_input.k_ext *= lambda_0
sim_household_population = HouseholdPopulation(
    sim_model_input.composition_list, sim_model_input.composition_distribution, sim_model_input)

no_imports = NoImportModel(4, 3)

base_rhs = UnloopedSEIRRateEquations(sim_model_input, sim_household_population, no_imports)

x0 = array(3 * [pop_prev])

fixed_imports = FixedImportModel(4,
                                 3,
                                 base_rhs,
                                 x0)
sim_rhs = UnloopedSEIRRateEquations(sim_model_input,
                                    sim_household_population,
                                    fixed_imports,
                                    sources="IMPORT")

t_eval = array([t0, t1, t2])
sol_sim = solve_ivp(sim_rhs, (t0, t2), P0, first_step=0.001, t_eval=t_eval, atol=1e-16)
rhs_tau_lam = deepcopy(rhs)
rhs_tau_lam.update_int_rate(tau_0)
rhs_tau_lam.update_ext_rate(lambda_0)
sol_overwrite = solve_ivp(rhs_tau_lam, (t0, t2), P0, first_step=0.001, t_eval=t_eval, atol=1e-16)
H_sim = sol_sim.y
H_overwrite = sol_overwrite.y
H_err = abs((H_sim + 1e-9) - (H_overwrite + 1e-9)) / (H_sim + 1e-9)

print("Relative error in solutions =",
      H_err.sum())

# Make sure overwriting of parameters works correctly:
tau_check = tau_0 / rhs.model_input.beta_int # Rescale to match method from data generation
print("Max difference in internal infection matrix: ", abs(tau_check * rhs.Q_int_inf.todense() - rhs.Q_int_inf).max())
print("Max difference in import matrix: ", abs(lambda_0 * rhs.Q_import.todense() - rhs.Q_import).max())

# Now do likelihood surface with both methods:

def f(pars):
    tau = pars[0]
    lam = pars[1]
    if (tau<=0)|(lam<=0):
        return 1e6
    else:
        return -loglike_with_counts(tau, lam, P0, Q1, Q2, Q0, Chi, result_to_index, obs_counts)

def f_ode(pars):
    tau = pars[0]
    lam = pars[1]
    if (tau<=0)|(lam<=0):
        return 1e6
    else:
        return -loglike_with_counts_odes(tau, lam, obs_counts)

t_range = tau_0 * arange(.2, 4., .2)
t_start = t_range[0]
t_end = t_range[-1]
l_range = lambda_0 * arange(.2, 10., .5)
l_start = l_range[0]
l_end = l_range[-1]

me_start = time()
llh_array_matrix_exp = array([-f([t, l]) for t in t_range for l in l_range]).reshape(len(t_range), len(l_range))
print("LLH array with matrix exponential calculated in", time() - me_start)
mle_arg_matrix_exp = unravel_index(llh_array_matrix_exp.argmax(), llh_array_matrix_exp.shape)

ode_start = time()
llh_array_ode = array([-f_ode([t, l]) for t in t_range for l in l_range]).reshape(len(t_range), len(l_range))
print("LLH array with ODE calculated in", time() - ode_start)
mle_arg_ode = unravel_index(llh_array_ode.argmax(), llh_array_ode.shape)

llh_diff = abs(llh_array_ode - llh_array_matrix_exp) / abs(llh_array_ode)
print("Error in LLH array =", llh_diff.sum())

fig, (ax1, ax2) = subplots(1, 2)
ax1.imshow(llh_array_matrix_exp,
           cmap = cm.magma,
           origin='lower',
           extent=(l_start, l_end, t_start, t_end))
ax1.scatter([lambda_0], [tau_0], color = 'k')
ax1.set_aspect((l_end-l_start)/(t_end-t_start))
ax1.set_title('Matrix exponential')
ax2.imshow(llh_array_ode,
           cmap = cm.magma,
           origin='lower',
           extent=(l_start, l_end, t_start, t_end))
ax2.set_aspect((l_end-l_start)/(t_end-t_start))
ax2.scatter([lambda_0], [tau_0], color = 'k')
ax2.set_title('ODE')
fig.show()

# Now try repeat fitting with numerical optimisation:

# Set cutoffs for abandoning fits:
tau_cutoff = 10 * tau_0
lambda_cutoff = 10 * lambda_0

def run_synth_inference():
    # Make dataset
    mhd = run_simulation(lambda_0, tau_0)

    # Convert to counts
    obs_outcomes, oc = unique(array(mhd), axis=0, return_counts = True)

    def f_mhd(pars):
        tau = pars[0]
        lam = pars[1]
        if (tau <= 0) | (lam <= 0):
            return 1e6
        else:
            return -loglike_with_counts(tau, lam, P0, Q1, Q2, Q0, Chi, result_to_index, oc)

    mle = optimize.minimize(f_mhd,
                            array([1., 1.]),
                            bounds=((1e-9, tau_cutoff), (1e-9, lambda_cutoff)),
                            method='Nelder-Mead')
    return (mle.x)

# IMPORTANT: Synthetic data generation and inference executes on the order of seconds,
# so set n_sample at your own risk!
n_sample = 5
mle_samples = [run_synth_inference() for i in tqdm(range(n_sample))]
tau_samples = array([m[0] for m in mle_samples])
lambda_samples = array([m[1] for m in mle_samples])

fail_locs = where((tau_cutoff - tau_samples) < 1e-9)[0]
fail_rate = len(fail_locs) / n_sample
print("Failure rate of inference:", fail_rate)

tau_samples = delete(tau_samples, fail_locs)
lambda_samples = delete(lambda_samples, fail_locs)

tau_mean = tau_samples.mean()
lambda_mean = lambda_samples.mean()
tau_ci = percentile(tau_samples, [2.5, 97.5])
lamdba_ci = percentile(lambda_samples, [2.5, 97.5])
print("tau_mle = ", tau_mean, tau_ci)
print("lambda_mle = ", lambda_mean, lamdba_ci)

tau_max = ceil(max(tau_samples))
lam_max = ceil(max(lambda_samples))

fig, ax = subplots(1, 1)
ax.scatter(lambda_samples,
           tau_samples,)
ax.plot([0, lam_max], [tau_0, tau_0], 'k--')
ax.plot([lambda_0, lambda_0], [0, tau_max], 'k--')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$\tau$')
ax.set_xlim(0, lam_max)
ax.set_ylim(0, tau_max)
fig.show()