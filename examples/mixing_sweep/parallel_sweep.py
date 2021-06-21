''' In this script we do projections  of the impact reducing within- and
between-household transmission by doing a 2D parameter sweep'''

from argparse import ArgumentParser
from os import mkdir
from os.path import isdir, isfile
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
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEPIRInput, HouseholdPopulation, make_initial_condition_with_recovereds)
from model.specs import TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations
from model.imports import NoImportModel, FixedImportModel

if isdir('outputs/mixing_sweep') is False:
    mkdir('outputs/mixing_sweep')

IMPORT_ARRAY = array([1e-5, 1e-5])

# List of observed household compositions
composition_list = read_csv(
    'inputs/eng_and_wales_adult_child_composition_list.csv',
    header=0).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/eng_and_wales_adult_child_composition_dist.csv',
    header=0).to_numpy().squeeze()

if isfile('outputs/mixing_sweep/beta_ext.pkl') is True:
    with open('outputs/mixing_sweep/beta_ext.pkl', 'rb') as f:
        beta_ext = load(f)
else:
    SPEC = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
    growth_rate = log(2) / 3 # Doubling time of 3 days
    model_input_to_fit = SEPIRInput(SPEC, composition_list, comp_dist)
    household_population_to_fit = HouseholdPopulation(
        composition_list, comp_dist, model_input_to_fit)
    rhs_to_fit = SEPIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(5,2))
    beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
    with open('outputs/mixing_sweep/beta_ext.pkl', 'wb') as f:
        dump(beta_ext, f)

prev=1.0e-5 # Starting prevalence
antibody_prev=0 # Starting antibody prev/immunity
AR=1.0 # Starting attack ratio - visited households are fully recovered

class MixingAnalysis:
    def __init__(self):
        self.basic_spec = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}

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
        model_input.k_ext = (1 - p[2]) * beta_ext * model_input.k_ext

        household_population = HouseholdPopulation(
            composition_list, comp_dist, model_input)

        rhs = SEPIRRateEquations(model_input, household_population, NoImportModel(5,2))

        growth_rate = estimate_growth_rate(household_population, rhs, [-0.9*this_spec['recovery_rate'], 5], 1e-2)
        if growth_rate is None:
            growth_rate = 0

        H0 = make_initial_condition_with_recovereds(household_population, rhs, prev, antibody_prev, AR)

        no_days = 100
        tspan = (0.0, no_days)
        solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

        t = solution.t
        H = solution.y

        ave_hh_size = household_population.composition_distribution.dot(sum(household_population.composition_list, axis=1))

        I = (H.T.dot(household_population.states[:, 3::5])).sum(axis=1)/ave_hh_size
        R = (H.T.dot(household_population.states[:, 4::5])).sum(axis=1)/ave_hh_size

        peaks = 100 * max(I)
        R_end = 100 * R[-1]

        return [growth_rate, peaks, R_end]

def main(no_of_workers,
         ar_vals,
         internal_mix_vals,
         external_mix_vals):
    mixing_system = MixingAnalysis()
    results = []
    ar_range = arange(ar_vals[0], ar_vals[1], ar_vals[2])
    internal_mix_range = arange(internal_mix_vals[0],
                                internal_mix_vals[1],
                                internal_mix_vals[2])
    external_mix_range = arange(external_mix_vals[0],
                                external_mix_vals[1],
                                external_mix_vals[2])
    params = array([
        [a, i, e]
        for a in ar_range
        for i in internal_mix_range
        for e in external_mix_range])

    with Pool(no_of_workers) as pool:
        results = pool.map(mixing_system, params)

    growth_rate_data = array([r[0] for r in results]).reshape(
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

    fname = 'outputs/mixing_sweep/results.pkl'
    with open(fname, 'wb') as f:
        dump(
            (growth_rate_data,
             peak_data,
             end_data,
             ar_range,
             internal_mix_range,
             external_mix_range),
            f)

    return -1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=8)
    parser.add_argument('--ar_vals', type=int, default=[0.25, 1, 0.25])
    parser.add_argument('--internal_mix_vals', type=int, default=[0, 1.0, 0.25])
    parser.add_argument('--external_mix_vals', type=int, default=[0, 1.0, 0.25])
    args = parser.parse_args()

    main(args.no_of_workers,
         args.ar_vals,
         args.internal_mix_vals,
         args.external_mix_vals)
