'''This script constructs the internal transmission matrix for a UK-like
population and a single instance of the external importation matrix.
'''
from numpy import where, zeros
from pandas import read_csv
from model.preprocessing import (
        HouseholdPopulation, ModelInput, make_initial_condition)
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
household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

rhs = RateEquations(
        model_input,
        household_population)

H = make_initial_condition(household_population, rhs)
Q_ext_det, Q_ext_undet = rhs.external_matrices(0.0, H)
