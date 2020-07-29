''' This runs a simple example with external importation to explore the
impact of different import functions on execution time'''

from os.path import isfile
from pickle import load, dump
from numpy import arange, array, diag, concatenate, vstack
from numpy.random import rand
from time import time as get_time
from scipy.integrate import solve_ivp
from model.preprocessing import TwoAgeModelInput, build_household_population
from model.preprocessing import make_initial_condition
from model.specs import DEFAULT_SPEC
from model.common import RateEquations
from model.imports import (
    ExponentialImportModel, StepImportModel, FixedImportModel, NoImportModel)
# pylint: disable=invalid-name

model_input = TwoAgeModelInput(DEFAULT_SPEC)

# List of observed household compositions
composition_list = array([
    [0, 1], [0, 2], [1, 1], [1, 2], [2, 1], [2, 2]])
# Proportion of households which are in each composition
comp_dist = array([0.2, 0.2, 0.1, 0.1, 0.1, 0.1])

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

# Relative strength of between-household transmission compared to external
# imports
epsilon = 0.5

no_days = 100

external_prev = rand(no_days)
detected_profile = array([0.1, 0.9])
undetected_profile = array([0.9, 0.1])
import_times = arange(no_days)

step_import_model = StepImportModel(
    import_times,
    external_prev,
    detected_profile,
    undetected_profile)

step_import_rhs = RateEquations(
    model_input,
    Q_int,
    composition_list,
    which_composition,
    states,
    inf_event_row,
    inf_event_col,
    inf_event_class,
    step_import_model,
    epsilon)

no_import_rhs = RateEquations(
    model_input,
    Q_int,
    composition_list,
    which_composition,
    states,
    inf_event_row,
    inf_event_col,
    inf_event_class,
    epsilon=1.0,
    importation_model=NoImportModel())

H0 = make_initial_condition(
    comp_dist, states, no_import_rhs, which_composition)

tspan = (0.0, no_days)
simple_model_start = get_time()
solution = solve_ivp(no_import_rhs, tspan, H0, first_step=0.001)
simple_model_end = get_time()

time = solution.t
H = solution.y
D = H.T.dot(states[:, 2::5])
U = H.T.dot(states[:, 3::5])

with open('simple-results.pkl', 'wb') as f:
    dump((time, H, D, U, model_input.coarse_bds), f)

fixed_import_model = FixedImportModel(
    step_import_model.detected(0.0),
    step_import_model.undetected(0.0))

fixed_rhs = RateEquations(
    model_input,
    Q_int,
    composition_list,
    which_composition,
    states,
    inf_event_row,
    inf_event_col,
    inf_event_class,
    epsilon,
    fixed_import_model)

with open('static-import-results.pkl', 'wb') as f:
    dump((time, H, D, U, model_input.coarse_bds), f)

# Now do timed version

tspan = (0.0, no_days)
true_step_start = get_time()
solution = solve_ivp(step_import_rhs, tspan, H0, first_step=0.001)
true_step_end = get_time()
time = solution.t
H = solution.y

D = H.T.dot(states[:, 2::5])
U = H.T.dot(states[:, 3::5])

with open('timed-import-results.pkl', 'wb') as f:
    dump((time, H, D, U, model_input.coarse_bds), f)

# Now do exponential import model_input

det_profile = model_input.det
# TODO: is this always 1?
undet_profile = array([1, 1]) - det_profile
r = model_input.gamma*DEFAULT_SPEC['R0'] - model_input.gamma
importation = ExponentialImportModel(r, det_profile, undet_profile)

rhs = RateEquations(
    model_input,
    Q_int,
    composition_list,
    which_composition,
    states,
    inf_event_row,
    inf_event_col,
    inf_event_class,
    importation,
    epsilon)

exponential_start = get_time()
solution = solve_ivp(rhs, tspan, H0, first_step=0.001)
exponential_end = get_time()
time = solution.t
H = solution.y

D = H.T.dot(states[:, 2::5])
U = H.T.dot(states[:, 3::5])

with open('exponential-import-results.pkl', 'wb') as f:
    dump((time, H, D, U, model_input.coarse_bds), f)


message = '{0} model took {1} seconds'
print(message.format('No import', simple_model_end-simple_model_start))
print(message.format('"True" step function', true_step_end-true_step_start))
print(message.format('Exponential', exponential_end-exponential_start))
