'''This is a test example to look at how we can simulate individual-based models.
Each individual is assumed to be of their own unique risk class, giving us more of an ability to track individual-level
as opposed to aggregate health outcomes.
'''
from numpy import arange, array, diag, exp, log, ones, sum, where, zeros
from numpy.linalg import eig
from os import mkdir
from os.path import isdir, isfile
from pandas import read_csv
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix as sparse
from time import time
from model.specs import LATENT_PERIOD, PRODROME_PERIOD, TRANCHE2_SITP, SYMPTOM_PERIOD
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.common import UnloopedSEIRRateEquations
from model.imports import FixedImportModel, NoImportModel

beta_int = .1
beta_ext = 0.1
x0 = [.1]
t_eval = arange(0, 361., 30.)

# Function for constructing individual-based model of a single household with homogeneous mixing:
def build_indiv_system(hh_size):

    composition_list = array([[1] * hh_size])
    # Proportion of households which are in each composition
    comp_dist = array([1.])

    SPEC = {
        'compartmental_structure': 'SEIR', # This is which subsystem key to use
        'beta_int': beta_int,                     # Internal infection rate
        'density_expo' : 1.,
        'recovery_rate': 1 / (PRODROME_PERIOD +
                              SYMPTOM_PERIOD),           # Recovery rate
        'incubation_rate': 1 / LATENT_PERIOD,         # E->I incubation rate
        'sus': array([1] * hh_size),          # Relative susceptibility by
                                      # age/vulnerability class
        'fit_method' : 'EL',
        'k_home': (1 / hh_size) * ones((hh_size, hh_size), dtype=float),
        'k_ext': ones((hh_size, hh_size), dtype=float),
        'skip_ext_scale' : True
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
    return(SPEC, model_input, household_population, rhs)

# Build systems for a range of household sizes:
hh_size_list = [2, 3, 4, 5, 6, 7, 8]
SPEC_LIST = []
model_input_list = []
household_population_list = []
rhs_list = []
H0_list = []
sol_list = []
time_series_list = []
build_time_list = []
solve_time_list = []
for hh_size in hh_size_list:
    build_start = time()
    (SPEC, model_input, household_population, rhs) = build_indiv_system(hh_size)
    SPEC_LIST.append(SPEC)
    model_input_list.append(model_input)
    household_population_list.append(household_population)
    rhs_list.append(rhs)
    build_time = time() - build_start
    build_time_list.append(build_time)
    print("Construction for hh size",
          hh_size,
          "took",
          build_time,
          "seconds.")

    H0 = zeros((household_population.total_size), )
    all_sus = where(sum(rhs.states_exp_only + rhs.states_inf_only + rhs.states_rec_only, 1) < 1e-1)[0]
    H0[all_sus] = 1. / len(all_sus)
    H0_list.append(H0)
    tspan = (0.0, 361.)
    sol_start = time()
    solution = solve_ivp(rhs,
                         tspan,
                         H0,
                         first_step=0.001,
                         atol=1e-16,
                         t_eval=t_eval)
    solve_time = time()- sol_start
    solve_time_list.append(solve_time)
    print("Solver took", solve_time, "seconds.")
    sol_list.append(solution)

    t = solution.t
    H = solution.y
    S = H.T.dot(household_population.states[:, ::4])
    E = H.T.dot(household_population.states[:, 1::4])
    I = H.T.dot(household_population.states[:, 2::4])
    R = H.T.dot(household_population.states[:, 3::4])
    time_series = {
    'time':t,
    'S':S,
    'E':E,
    'I':I,
    'R':R
    }
    time_series_list.append(time_series)

# Next: compare outputs from an individual-based and non-individual-based model:
def do_aggregated_model(hh_size):
    composition_list = array([[hh_size]])
    # Proportion of households which are in each composition
    comp_dist = array([1.])
    
    int_mix_rate = max(eig(ones((hh_size, hh_size)))[0])
    ext_mix_rate = max(eig(ones((hh_size, hh_size)))[0])
    
    SPEC = {
        'compartmental_structure': 'SEIR', # This is which subsystem key to use
        'beta_int': beta_int,                     # Internal infection rate
        'density_expo' : 1.,
        'recovery_rate': 1 / (PRODROME_PERIOD +
                              SYMPTOM_PERIOD),           # Recovery rate
        'incubation_rate': 1 / LATENT_PERIOD,         # E->I incubation rate
        'sus': array([1]),          # Relative susceptibility by
                                      # age/vulnerability class
        'fit_method' : 'EL',
        'k_home': ones((1, 1), dtype=float),
        'k_ext': ones((1, 1), dtype=float),
        'skip_ext_scale' : True
    }
    
    model_input = SEIRInput(SPEC, composition_list, comp_dist)
    model_input.k_ext *= beta_ext
    
    # With the parameters chosen, we calculate Q_int:
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input)
    
    # Set up import model
    no_imports = NoImportModel(4, 1)
    base_rhs = UnloopedSEIRRateEquations(model_input, household_population, no_imports)
    fixed_imports = FixedImportModel(4,
                                     1,
                                     base_rhs,
                                     array(x0))
    
    rhs = UnloopedSEIRRateEquations(model_input,
                                    household_population,
                                    fixed_imports,
                                    sources="IMPORT")
    H0 = zeros(household_population.total_size, )
    all_sus = where(sum(rhs.states_exp_only + rhs.states_inf_only + rhs.states_rec_only, 1) < 1e-1)[0]
    H0[all_sus] = 1. / len(all_sus)
    tspan = (0.0, 361)
    solution = solve_ivp(rhs,
                         tspan,
                         H0,
                         first_step=0.001,
                         atol=1e-16,
                         t_eval=t_eval)
    
    t = solution.t
    H = solution.y
    S = H.T.dot(household_population.states[:, ::4])
    E = H.T.dot(household_population.states[:, 1::4])
    I = H.T.dot(household_population.states[:, 2::4])
    R = H.T.dot(household_population.states[:, 3::4])
    time_series = {
    'time':t,
    'S':S,
    'E':E,
    'I':I,
    'R':R
    }
    return time_series
agg_time_series_list = [do_aggregated_model(hh_size) for hh_size in hh_size_list]
from matplotlib.pyplot import subplots

for i in range(len(hh_size_list)):

    print("hh_size =", hh_size_list[i])
    print("Max relative error in S:",
          (abs(time_series_list[i]['S'][1:].sum(1) - agg_time_series_list[i]['S'][1:].T) /
           time_series_list[i]['S'][1:].sum(1)).max())
    print("Max relative error in E:",
          (abs(time_series_list[i]['E'][1:].sum(1) - agg_time_series_list[i]['E'][1:].T) /
           time_series_list[i]['E'][1:].sum(1)).max())
    print("Max relative error in I:",
          (abs(time_series_list[i]['I'][1:].sum(1) - agg_time_series_list[i]['I'][1:].T) /
           time_series_list[i]['I'][1:].sum(1)).max())
    print("Max relative error in R:",
          (abs(time_series_list[i]['R'][1:].sum(1) - agg_time_series_list[i]['R'][1:].T) /
           time_series_list[i]['R'][1:].sum(1)).max())

    fig, (ax_ibm, ax_agg) = subplots(1, 2)
    ax_ibm.plot(time_series_list[i]['S'].sum(1))
    ax_ibm.plot(time_series_list[i]['E'].sum(1))
    ax_ibm.plot(time_series_list[i]['I'].sum(1))
    ax_ibm.plot(time_series_list[i]['R'].sum(1))
    ax_agg.plot(agg_time_series_list[i]['S'].sum(1))
    ax_agg.plot(agg_time_series_list[i]['E'].sum(1))
    ax_agg.plot(agg_time_series_list[i]['I'].sum(1))
    ax_agg.plot(agg_time_series_list[i]['R'].sum(1))
    ax_ibm.set_title("HH size = " + str(hh_size_list[i]))
    fig.show()
