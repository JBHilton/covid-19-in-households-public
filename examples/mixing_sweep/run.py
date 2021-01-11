''' In this script we do projections  of the impact reducing within- and
between-household transmission by doing a 2D parameter sweep'''

from os.path import isfile
from pickle import load, dump
from copy import deepcopy
from numpy import arange, array, exp, log
from numpy.linalg import eig
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.integrate import solve_ivp
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from model.preprocessing import TwoAgeWithVulnerableInput, HouseholdPopulation
from model.preprocessing import add_vulnerable_hh_members, make_initial_SEPIRQ_condition
from model.common import SEPIRQRateEquations, within_household_SEPIRQ
from model.imports import ( FixedImportModel)
from model.specs import SEPIRQ_SPEC
from examples.mixing_sweep.common import (SEPIR_SPEC, SEPIRInput, DataObject,
make_initial_condition_with_recovereds, within_household_SEPIR, RateEquations)

class DataObject(): # Very hacky way to store output
    def __init__(self,thing):
        self.thing = thing

basic_spec = SEPIR_SPEC
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
antibody_prev=5.6e-2 # Starting antibody prev/immunity
AR=1.0 # Starting attack ratio - visited households are fully recovered

# internal_mix_range = array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
# external_mix_range = array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

AR_range = array([0.15,0.3,0.45])

internal_mix_range = arange(0.0,1.0,0.1)
external_mix_range = arange(0.0,1.0,0.1)

AR_len = len(AR_range)
internal_mix_len = len(internal_mix_range)
external_mix_len = len(external_mix_range)

for i in range(AR_len):

    filename_stem_i = 'mix_sweep_results_' + str(i)

    spec = deepcopy(basic_spec)
    spec['AR'] = AR_range[i]

    for j in range(internal_mix_len):

        filename_stem_j = filename_stem_i + str(j)

        for k in range(external_mix_len):

            filename = filename_stem_j + str(k)

            iter_start = get_time()

            model_input = SEPIRInput(spec)
            model_input.k_home = (1-internal_mix_range[j]) * model_input.k_home
            model_input.k_ext = (1-external_mix_range[k]) * model_input.k_ext

            household_population = HouseholdPopulation(
                composition_list, comp_dist, model_input, within_household_SEPIR,5,True)

            rhs = RateEquations(model_input, household_population)

            H0 = make_initial_condition_with_recovereds(household_population, rhs, prev, antibody_prev, AR)

            no_days = 30
            tspan = (0.0, no_days)
            solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)
            iter_end = get_time()

            results = DataObject(0)
            results.t = solution.t
            results.H = solution.y

            with open(filename + '.pkl', 'wb') as f:
                dump((AR_range[i], household_population, results),
                 f)

            print('Iteration', internal_mix_len*external_mix_len*i + external_mix_len*j + k,'of',AR_len*internal_mix_len*external_mix_len,'took',iter_end-iter_start,'seconds.')
