from os import mkdir
from os.path import isdir
from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import Pool
from numpy import (arange, argmin, array, diag,  log, mean, ones, sqrt, where,
        zeros)
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
        SEPIRInput, HouseholdPopulation)
from model.specs import (draw_random_two_age_SEPIR_specs, TWO_AGE_SEPIR_SPEC,
    TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC)
from model.common import SEPIRRateEquations
from model.imports import NoImportModel

if isdir('outputs/between_hh_fitting') is False:
    mkdir('outputs/between_hh_fitting')

class BetaEstimator:
    def __init__(self):
        # List of observed household compositions
        self.composition_list = read_csv(
            'inputs/eng_and_wales_adult_child_composition_list.csv',
            header=0).to_numpy()
        # Proportion of households which are in each composition
        self.comp_dist = read_csv(
            'inputs/eng_and_wales_adult_child_composition_dist.csv',
            header=0).to_numpy().squeeze()

        SPEC = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
        model_input_to_fit = SEPIRInput(SPEC,
                                        self.composition_list,
                                        self.comp_dist)
        self.household_population_to_fit = HouseholdPopulation(
            self.composition_list, self.comp_dist, model_input_to_fit)
        self.rhs_to_fit = SEPIRRateEquations(model_input_to_fit,
                                             self.household_population_to_fit,
                                             NoImportModel(5,2))

    def __call__(self, p):
        try:
            result = self._fit_random_beta()
        except ValueError as err:
            print('Exception raised')
            return 0.0
        return result

    def _fit_random_beta(self):

        growth_rate =  -0.1 + 0.6 * rand(1,)[0]
        try:
            beta_ext_guess = estimate_beta_ext(self.household_population_to_fit,
                                               self.rhs_to_fit,
                                               growth_rate)
            return [growth_rate, beta_ext_guess[0]]
        except:
            print('Calculation failed for r =', growth_rate)
            return [growth_rate, 0.]

def main(no_of_workers, no_samples):
    estimator = BetaEstimator()
    results = []
    with Pool(no_of_workers) as pool:
        results = pool.map(estimator, ones((no_samples, 1)))

    growth_rate_samples = array([r[0] for r in results])
    beta_out = array([r[1] for r in results])

    print(len(where(beta_out==0)[0]), 'of', no_samples,'estimates failed.')

    with open('outputs/between_hh_fitting/repeat_mixing_fits.pkl', 'wb') as f:
        dump((growth_rate_samples, beta_out), f)

    return -1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=8)
    parser.add_argument('--no_samples', type=int, default=100)
    args = parser.parse_args()

    main(args.no_of_workers, args.no_samples)
