'''This sets up and runs a simple system which is low-dimensional enough to
do locally'''
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import ModelInput, build_household_population
from model.specs import DEFAULT_SPEC as spec
from model.common import RateEquations

spec['R0'] = 2.0
spec['gamma'] = 0.5
spec['det_model']['type'] = 'constant'
spec['det_model']['constant'] = 0.2

composition_list = array([
    [0,1],
    [0,2],
    [1,1],
    [2,1],
    [1,2],
    [2,2])
comp_dist = array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2]);

Q_int, states, which_composition, \
        system_sizes, cum_sizes, \
        inf_event_row, inf_event_col \
    = build_household_population(composition_list, model_input)

rhs = RateEquations(
    model_input,
    Q_int,
    composition_list,
    which_composition,
    states,
    inf_event_row,
    inf_event_col)

# Initialisation
fully_sus = where(rhs.states_sus_only.sum(axis=1) == states.sum(axis=1))[0]
i_is_one = where((rhs.states_det_only + rhs.states_undet_only).sum(axis=1) == 1)[0]
H0 = zeros(len(which_composition))
# Assign probability of 1e-5 to each member of each composition being sole infectious person in hh
H0[i_is_one] = (1.0e-5) * comp_dist[which_composition[i_is_one]]
# Assign rest of probability to there being no infection in the household
H0[fully_sus] = (1 - 1e-5 * sum(comp_dist[which_composition[i_is_one]])) * comp_dist

tspan = (0.0, 100)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001)

time = solution.t
H = solution.y
D = H.T.dot(states[:, 2::5])
U = H.T.dot(states[:, 3::5])

with open('simple.pkl', 'wb') as f:
    dump((time, H, D, U, model_input.coarse_bds), f)
