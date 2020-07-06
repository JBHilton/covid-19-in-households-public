'''This runs the UK-like model with a single set of parameters for 100 days
'''
from os.path import isfile
from pickle import load, dump
from numpy import where, zeros
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import ModelInput, build_household_population
from model.specs import DEFAULT_SPEC
from model.common import NoImportRateEquations
# pylint: disable=invalid-name

model_input = ModelInput(DEFAULT_SPEC)

# List of observed household compositions
composition_list = read_csv(
    'inputs/uk_composition_list.csv',
    header=None).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/uk_composition_dist.csv',
    header=None).to_numpy().squeeze()

if isfile('vars.pkl') is True:
    with open('vars.pkl', 'rb') as f:
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
    with open('vars.pkl', 'wb') as f:
        dump((
            Q_int, states, which_composition, system_sizes, cum_sizes,
            inf_event_row, inf_event_col, inf_event_class), f)

rhs = NoImportRateEquations(
    model_input,
    Q_int,
    composition_list,
    which_composition,
    states,
    inf_event_row,
    inf_event_col,
    inf_event_class)

# Initialisation
fully_sus = where(rhs.states_sus_only.sum(axis=1) == states.sum(axis=1))[0]
i_is_one = where((rhs.states_det_only + rhs.states_undet_only).sum(axis=1) == 1)[0]
H0 = zeros(len(which_composition))
# Assign probability of 1e-5 to each member of each composition being sole infectious person in hh
H0[i_is_one] = (1.0e-5) * comp_dist[which_composition[i_is_one]]
# Assign rest of probability to there being no infection in the household
H0[fully_sus] = (1 - 1e-5 * sum(comp_dist[which_composition[i_is_one]])) * comp_dist

tspan = (0.0, 10)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001)

time = solution.t
H = solution.y
D = H.T.dot(states[:, 2::5])
U = H.T.dot(states[:, 3::5])

with open('uk_like.pkl', 'wb') as f:
    dump((time, H, D, U, model_input.coarse_bds), f)
