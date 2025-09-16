'''This runs the UK-like model with a single set of parameters for 100 days
'''
from copy import deepcopy
from numpy import arange, array, log
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import (estimate_beta_ext, estimate_growth_rate, estimate_hh_reproductive_ratio,
                                 SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector,
                                 estimate_hh_reproductive_ratio)
from model.specs import TWO_AGE_SEIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import UnloopedSEIRRateEquations as SEIRRateEquations
from model.imports import NoImportModel
# pylint: disable=invalid-name

if isdir('outputs/uk') is False:
    mkdir('outputs/uk')

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
print('Estimated beta is',beta_ext)


# With the parameters chosen, we calculate Q_int:
household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

rhs = SEIRRateEquations(model_input, household_population, NoImportModel(4,2))

r_est = estimate_growth_rate(household_population, rhs, [0.001, 5], 1e-9)


print('Estimated growth rate is',r_est,'.')
print('Estimated doubling time is',log(2) / r_est,'.')
print('Estimated R_* is', estimate_hh_reproductive_ratio(household_population, rhs), '.')

# Calculate R* and r for a range of beta_ext values

gr_interval = [-SPEC['recovery_rate'], 1] # Interval used in growth rate estimation
gr_tol = 1e-3 # Absolute tolerance for growth rate estimation

external_mix_range = arange(0.1, .25, 0.01)
def r_from_scale(ext_scale):
    rhs_scaled = deepcopy(rhs)
    rhs_scaled.update_ext_rate(ext_scale)
    return estimate_growth_rate(household_population,
                                           rhs_scaled,
                                           gr_interval,
                                           gr_tol,
                                           x0=1e-3,
                                           r_min_discount=0.99)
def rstar_from_scale(ext_scale):
    rhs_scaled = deepcopy(rhs)
    rhs_scaled.update_ext_rate(ext_scale)
    return estimate_hh_reproductive_ratio(household_population,
                                           rhs_scaled)

r_by_scale = [r_from_scale(es) for es in external_mix_range]
rstar_by_scale = [rstar_from_scale(es) for es in external_mix_range]
