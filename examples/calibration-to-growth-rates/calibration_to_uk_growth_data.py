'''This runs the model with repeated calibration to weekly UK growth rate estimates.
'''
from copy import deepcopy
from numpy import arange, array, log, where
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp

from examples.uk.compare_matrix_import_runs_SEIR import base_rhs
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import TWO_AGE_SEIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import SEIRRateEquations, MatrixImportSEIRRateEquations, UnloopedSEIRRateEquations
from model.imports import FixedImportModel, NoImportModel, ExponentialImportModel

# pylint: disable=invalid-name

if isdir('outputs/calibration-to-growth-rates') is False:
    mkdir('outputs/calibration-to-growth-rates')

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
beta_ext = 1.
model_input = deepcopy(model_input_to_fit)
model_input.k_ext *= beta_ext

# With the parameters chosen, we calculate Q_int:
household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

no_imports = NoImportModel(4, 2)

base_rhs = UnloopedSEIRRateEquations(model_input, household_population, no_imports, sources="BETWEEN")

def recalibrate_to_growth_rate(rhs,
                               growth_rate,
                               prev=0.,
                               starting_immunity=0.):
    beta_ext_new = estimate_beta_ext(household_population, rhs, growth_rate)[0]
    rhs.update_ext_rate(beta_ext_new)
    return rhs

scaled_rhs = recalibrate_to_growth_rate(base_rhs, 1.)
r_est = estimate_growth_rate(household_population, scaled_rhs, [0.001, 5], 1e-9)
print(r_est)

scaled_rhs = recalibrate_to_growth_rate(base_rhs, .5)
r_est = estimate_growth_rate(household_population, scaled_rhs, [-SPEC["recovery_rate"] + 1e-6, 5], 1e-9)
print(r_est)

# Try for a range of values
gr_list = arange(.5, 5., .251)
new_gr_list = [estimate_growth_rate(household_population,
                                          recalibrate_to_growth_rate(base_rhs, gr_list[ir]),
                                          [0.001, 5],
                                          1e-9) for ir in range(len(gr_list))]
print("Max absolute error in growth rate estimates is",
      abs(gr_list - array(new_gr_list)).max())