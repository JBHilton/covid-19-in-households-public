'''This script constructs the internal transmission matrix for a UK-like
population and a single instance of the external importation matrix.
'''
from numpy import where, zeros
from pandas import read_csv
from model.preprocessing import build_household_population, ModelInput
from model.common import RateEquations
from model.specs import DEFAULT_SPEC as spec
# pylint: disable=invalid-name

spec['R0'] = 1.0
spec['gamma'] = 1.0
spec['tau'] = 0.5
spec['det_model']['type'] = 'constant'
spec['det_model']['constant'] = 0.2

model_input = ModelInput(spec)

# List of observed household compositions
composition_list = read_csv(
    'inputs/uk_composition_list.csv',
    header=None).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/uk_composition_dist.csv',
    header=None).to_numpy().squeeze()

# With the parameters chosen, we calculate Q_int:
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

fully_sus = where(rhs.states_sus_only.sum(axis=1) == states.sum(axis=1))[0]
i_is_one = where((rhs.states_det_only + rhs.states_undet_only).sum(axis=1) == 1)[0]
H = zeros(len(which_composition))
# Assign probability of 1e-5 to each member of each composition being sole infectious person in hh
H[i_is_one] = (1e-5) * comp_dist[which_composition[i_is_one]]
# Assign rest of probability to there being no infection in the household
H[fully_sus] = (1 - 1e-5 * sum(comp_dist[which_composition[i_is_one]])) * comp_dist

Q_ext_det, Q_ext_undet = rhs.external_matrices(H)
