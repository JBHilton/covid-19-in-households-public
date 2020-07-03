''' This runs a simple example with external importation to explore the
impact of time-varying imports on execution time'''

from os.path import isfile
from pickle import load, dump
from numpy import arange, array, where, zeros, diag, dot, concatenate
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.integrate import solve_ivp
from model.preprocessing import TwoAgeModelInput, build_household_population
from model.specs import DEFAULT_SPEC
from model.common import RateEquationsWithImports, RateEquationsWithStepFunctionImports, RateEquationsWithExponentialImports
# pylint: disable=invalid-name

model_input = TwoAgeModelInput(DEFAULT_SPEC)

# List of observed household compositions
composition_list = array([[0,1],[0,2],[1,1],[1,2],[2,1],[2,2]])
# Proportion of households which are in each composition
comp_dist = array([0.2,0.2,0.1,0.1,0.1,0.1])

if isfile('import-vars.pkl') is True:
    with open('import-vars.pkl', 'rb') as f:
        Q_int, states, which_composition, \
                system_sizes, cum_sizes, \
                inf_event_row, inf_event_col, inf_event_class \
            = load(f)
else:
    # With the parameters chosen, we calculate Q_int:
    Q_int, states, which_composition, \
            system_sizes, cum_sizes, \
            inf_event_row, inf_event_col, inf_event_class \
        = build_household_population(composition_list, model_input)
    with open('import-vars.pkl', 'wb') as f:
        dump((
            Q_int, states, which_composition, system_sizes, cum_sizes,
            inf_event_row, inf_event_col, inf_event_class), f)

epsilon = 1 #Relative strength of between-household transmission compared to external imports

'''Next step: loop over time with updating det_imports and undet_imports'''

no_days = 100

external_prev = rand(2,no_days)
det_imports = diag(array([0,0])).dot(external_prev)
undet_imports = diag(array([0,0])).dot(external_prev)
import_times = arange(no_days)

def time_import_model(t,H):
    rhs = RateEquationsWithStepFunctionImports(
    t,
    model_input,
    Q_int,
    composition_list,
    which_composition,
    states,
    inf_event_row,
    inf_event_col,
    inf_event_class,
    epsilon,
    det_imports,
    undet_imports,
    import_times)
    return rhs(t,H)

rhs = RateEquationsWithImports(
    model_input,
    Q_int,
    composition_list,
    which_composition,
    states,
    inf_event_row,
    inf_event_col,
    inf_event_class,
    det_imports[:,0],
    undet_imports[:,0])

# Initialisation
fully_sus = where(rhs.states_sus_only.sum(axis=1) == states.sum(axis=1))[0]
i_is_one = where((rhs.states_det_only + rhs.states_undet_only).sum(axis=1) == 1)[0]
H0 = zeros(len(which_composition))
# Assign probability of 1e-5 to each member of each composition being sole infectious person in hh
H0[i_is_one] = (1.0e-5) * comp_dist[which_composition[i_is_one]]
# Assign rest of probability to there being no infection in the household
H0[fully_sus] = (1 - 1e-5 * sum(comp_dist[which_composition[i_is_one]])) * comp_dist


tspan = (0.0, 1)
stepping_model_start = get_time()
solution = solve_ivp(rhs, tspan, H0, first_step=0.001)

time = solution.t
H = solution.y

for t in range(1,no_days):
    rhs = RateEquationsWithImports(
        model_input,
        Q_int,
        composition_list,
        which_composition,
        states,
        inf_event_row,
        inf_event_col,
        inf_event_class,
        det_imports[:,t],
        undet_imports[:,t])
    solution = solve_ivp(rhs, tspan, H[:,-1], first_step=0.001)
    time = concatenate((time, time[-1]+solution.t))
    H = concatenate((H, solution.y),axis=1)

stepping_model_end = get_time()
D = H.T.dot(states[:, 2::5])
U = H.T.dot(states[:, 3::5])

with open('static-import-results.pkl', 'wb') as f:
    dump((time, H, D, U, model_input.coarse_bds), f)

# Now do timed version

tspan = (0.0, no_days)
inhom_model_start = get_time()
solution = solve_ivp(time_import_model, tspan, H0, first_step=0.001)
inhom_model_end = get_time()
time = solution.t
H = solution.y

D = H.T.dot(states[:, 2::5])
U = H.T.dot(states[:, 3::5])

with open('timed-import-results.pkl', 'wb') as f:
    dump((time, H, D, U, model_input.coarse_bds), f)

# Now do exponential import model_input

epsilon = 0.2
det_profile = model_input.det
undet_profile = [1,1] - det_profile
r = model_input.gamma*DEFAULT_SPEC['R0'] - model_input.gamma

def exponential_import_model(t,H):
    rhs = RateEquationsWithExponentialImports(
    t,
    model_input,
    Q_int,
    composition_list,
    which_composition,
    states,
    inf_event_row,
    inf_event_col,
    inf_event_class,
    epsilon,
    det_profile,
    undet_profile,
    r)
    return rhs(t,H)

exp_model_start = get_time()
solution = solve_ivp(exponential_import_model, tspan, H0, first_step=0.001)
exp_model_end = get_time()
time = solution.t
H = solution.y

D = H.T.dot(states[:, 2::5])
U = H.T.dot(states[:, 3::5])

with open('exponential-import-results.pkl', 'wb') as f:
    dump((time, H, D, U, model_input.coarse_bds), f)

print('Stepping model took ',stepping_model_end-stepping_model_start,' seconds.')
print('Inhomogeneous model took ',inhom_model_end-inhom_model_start,' seconds.')
print('Exponential model took ',exp_model_end-exp_model_start,' seconds.')
