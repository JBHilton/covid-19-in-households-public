from copy import deepcopy
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from numpy import arange, argmin, array, diag,  log, ones, where, zeros
from numpy.linalg import eig
from numpy.random import rand
from os.path import isfile
from pandas import read_csv
from pickle import load, dump
from scipy.integrate import solve_ivp
from scipy.sparse import eye, identity
from scipy.sparse import csc_matrix as sparse
from scipy.sparse.linalg import inv
from time import time as get_time
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEPIRInput, HouseholdPopulation,
        make_initial_condition)
from model.specs import (draw_random_two_age_SEPIR_specs, TWO_AGE_SEPIR_SPEC,
    TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC)
from model.common import SEPIRRateEquations
from model.imports import NoImportModel

def fit_beta_from_beta(beta_ext, spec, composition_list, comp_dist):
    fitted_model_input = SEPIRInput(spec, composition_list, comp_dist)
    fitted_model_input.k_ext = beta_ext * fitted_model_input.k_ext
    fitted_household_population = HouseholdPopulation(
        composition_list, comp_dist, fitted_model_input)

    fitted_rhs = SEPIRRateEquations(fitted_model_input, fitted_household_population, NoImportModel(5,2))
    r_est = estimate_growth_rate(fitted_household_population,fitted_rhs,[0.001,5])

    model_input_to_fit = SEPIRInput(spec, composition_list, comp_dist)

    household_population_to_fit = HouseholdPopulation(
        composition_list, comp_dist, model_input_to_fit)

    rhs_to_fit = SEPIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(5,2))

    beta_ext_guess = estimate_beta_ext(household_population_to_fit, rhs_to_fit, r_est)

    return beta_ext_guess

# List of observed household compositions
composition_list = read_csv(
    'inputs/eng_and_wales_adult_child_composition_list.csv',
    header=0).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/eng_and_wales_adult_child_composition_dist.csv',
    header=0).to_numpy().squeeze()

SPEC_TO_FIT = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}

beta_in = []
beta_out = []

no_samples = 5

start_time = get_time()
for i in range(no_samples):
    specs = draw_random_two_age_SEPIR_specs(SPEC_TO_FIT)
    beta_rand =  0.9 + rand(1,)
    beta_fit = fit_beta_from_beta(beta_rand, specs, composition_list, comp_dist)
    beta_in.append(beta_rand)
    beta_out.append(beta_fit)
    time_now = get_time()
    print(i,
          'of',
          no_samples,
          'calculations completed',
          time_now-start_time,
          'seconds elapsed,estimated',
         (no_samples-(i+1))*(time_now-start_time)/(i+1),
         'seconds remaining.')
print('beta_in=',beta_in)
print('beta_ext=',beta_out)
