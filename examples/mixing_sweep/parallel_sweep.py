''' In this script we do projections  of the impact reducing within- and
between-household transmission by doing a 2D parameter sweep'''

from argparse import ArgumentParser
from os.path import isfile
from pickle import load, dump
from copy import deepcopy
from multiprocessing import Pool
from numpy import arange, array, exp, log, sum
from numpy.linalg import eig
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.integrate import solve_ivp
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from model.preprocessing import ( estimate_growth_rate,
        SEPIRInput, HouseholdPopulation, make_initial_condition_with_recovereds)
from model.specs import TWO_AGE_SEPIR_SPEC, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations
from model.imports import FixedImportModel

IMPORT_ARRAY = array([1e-5, 1e-5])

# class DataObject(): # Very hacky way to store output
#     def __init__(self,thing):
#         self.thing = thing
#
# basic_spec = {**TWO_AGE_SEPIR_SPEC, **TWO_AGE_UK_SPEC}
# print('Approx R_int is', -log(1-basic_spec['AR']))
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

class MixingAnalysis:
    def __init__(self):
        self.basic_spec = {**TWO_AGE_SEPIR_SPEC, **TWO_AGE_UK_SPEC}

    def __call__(self, p):
        try:
            result = self._implement_mixing(p)
        except ValueError as err:
            print('Exception raised for parameters={0}\n\tException: {1}'.format(
                p, err))
            return 0.0
        return result

    def _implement_mixing(self, p):
        this_spec = deepcopy(self.basic_spec)
        this_spec['AR'] = p[0]
        model_input = SEPIRInput(this_spec, composition_list, comp_dist)
        model_input.k_home = (1 - p[1]) * model_input.k_home
        model_input.k_ext = (1 - p[2]) * model_input.k_ext

        household_population = HouseholdPopulation(
            composition_list, comp_dist, model_input)

        rhs = SEPIRRateEquations(model_input, household_population, FixedImportModel(6, 2, IMPORT_ARRAY))

        beta_ext = estimate_growth_rate(household_population, rhs, [-5, 5], 1e-2)
        if beta_ext is None:
            beta_ext = 0

        H0 = make_initial_condition_with_recovereds(household_population, rhs, prev, antibody_prev, AR)

        no_days = 100
        tspan = (0.0, no_days)
        solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)
        iter_end = get_time()

        t = solution.t
        H = solution.y

        ave_hh_size = household_population.composition_distribution.dot(sum(household_population.composition_list, axis=1))

        I = (H.T.dot(household_population.states[:, 3::5])).sum(axis=1)/ave_hh_size
        R = (H.T.dot(household_population.states[:, 4::5])).sum(axis=1)/ave_hh_size

        peaks = 100 * max(I)
        R_end = 100 * R[-1]

        return [beta_ext, peaks, R_end]

def main(no_of_workers):
    mixing_system = MixingAnalysis()
    results = []
    ar_range = array([0.3, 0.45, 0.6])
    internal_mix_range = array([0.2,0.6])
    external_mix_range = array([0.2,0.6])
    params = array([
        [a, i, e]
        for a in ar_range
        for i in internal_mix_range
        for e in external_mix_range])

    with Pool(no_of_workers) as pool:
        results = pool.map(mixing_system, params)

    beta_ext_data = array([r[0] for r in results]).reshape(
        len(ar_range),
        len(internal_mix_range),
        len(external_mix_range))
    peak_data = array([r[1] for r in results]).reshape(
        len(ar_range),
        len(internal_mix_range),
        len(external_mix_range))
    end_data = array([r[2] for r in results]).reshape(
        len(ar_range),
        len(internal_mix_range),
        len(external_mix_range))

    fname = 'mixing_sweep_output.pkl'
    with open(fname, 'wb') as f:
        dump(
            (beta_ext_data, peak_data, end_data, params),
            f)

    return -1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=2)
    args = parser.parse_args()

    main(args.no_of_workers)
