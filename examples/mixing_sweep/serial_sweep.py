''' In this script we do projections  of the impact reducing within- and
between-household transmission by doing a 2D parameter sweep'''

from os.path import isfile
from pickle import load, dump
from copy import deepcopy
from numpy import arange, array, exp, log, sum
from numpy.linalg import eig
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.integrate import solve_ivp
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from model.preprocessing import (
        SEPIRInput, HouseholdPopulation, make_initial_condition_with_recovereds)
from model.specs import TWO_AGE_SEPIR_SPEC, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations
from model.imports import FixedImportModel

IMPORT_ARRAY = array([1e-5, 1e-5])

class DataObject(): # Very hacky way to store output
    def __init__(self,thing):
        self.thing = thing

basic_spec = {**TWO_AGE_SEPIR_SPEC, **TWO_AGE_UK_SPEC}
print('Approx R_int is', -log(1-basic_spec['AR']))
# List of observed household compositions
composition_list = read_csv(
    'inputs/eng_and_wales_adult_child_composition_list.csv',
    header=0).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/eng_and_wales_adult_child_composition_dist.csv',
    header=0).to_numpy().squeeze()

prev=1.0e-2 # Starting prevalence
antibody_prev=0 # Starting antibody prev/immunity
AR=1.0 # Starting attack ratio - visited households are fully recovered

# internal_mix_range = array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
# external_mix_range = array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

AR_range = array([0.3, 0.45, 0.6])

internal_mix_range = arange(0.0,1.0,0.25)
external_mix_range = arange(0.0,1.0,0.25)

internal_mix_range = array([0.2,0.6])
external_mix_range = array([0.2,0.6])

AR_len = len(AR_range)
internal_mix_len = len(internal_mix_range)
external_mix_len = len(external_mix_range)

for i in range(AR_len):

    filename_stem_i = 'mix_sweep_results_AR' + str(AR_range[i])

    spec = deepcopy(basic_spec)
    spec['AR'] = AR_range[i]

    for j in range(internal_mix_len):

        filename_stem_j = filename_stem_i + '_intred' + str(internal_mix_range[j])

        for k in range(external_mix_len):

            filename = filename_stem_j + '_extred' + str(external_mix_range[k])

            iter_start = get_time()

            model_input = SEPIRInput(spec, composition_list, comp_dist)
            model_input.k_home = (1-internal_mix_range[j]) * model_input.k_home
            model_input.k_ext = (1-external_mix_range[k]) * model_input.k_ext

            household_population = HouseholdPopulation(
                composition_list, comp_dist, model_input)

            rhs = SEPIRRateEquations(model_input, household_population, FixedImportModel(6, 2, IMPORT_ARRAY))

            H0 = make_initial_condition_with_recovereds(household_population, rhs, prev, antibody_prev, AR)

            no_days = 100
            tspan = (0.0, no_days)
            solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)
            iter_end = get_time()

            results = DataObject(0)
            results.t = solution.t
            results.H = solution.y

            ave_hh_size = household_population.composition_distribution.dot(sum(household_population.composition_list, axis=1))

            results.I = (results.H.T.dot(household_population.states[:, 3::5])).sum(axis=1)/ave_hh_size
            results.R = (results.H.T.dot(household_population.states[:, 4::5])).sum(axis=1)/ave_hh_size

            # print(max(results.I))
            # print(results.R[-1])

            with open(filename + '.pkl', 'wb') as f:
                dump((AR_range[i], household_population, results),
                 f)

            print('Iteration', internal_mix_len*external_mix_len*i + external_mix_len*j + k,'of',AR_len*internal_mix_len*external_mix_len,'took',iter_end-iter_start,'seconds.')
