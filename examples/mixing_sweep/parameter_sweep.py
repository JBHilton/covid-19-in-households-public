''' In this script we do projections  of the impact reducing within- and
between-household transmission by doing a 2D parameter sweep'''

from argparse import ArgumentParser
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from copy import deepcopy
from multiprocessing import Pool
from numpy import arange, array, exp, log, sum, where
from numpy.linalg import eig
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.integrate import solve_ivp
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEPIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations
from model.imports import NoImportModel

if isdir('outputs/mixing_sweep') is False:
    mkdir('outputs/mixing_sweep')

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

if isfile('outputs/mixing_sweep/beta_ext.pkl') is True:
    with open('outputs/mixing_sweep/beta_ext.pkl', 'rb') as f:
        beta_ext = load(f)
else:
    SPEC = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
    model_input_to_fit = SEPIRInput(SPEC, composition_list, comp_dist)
    household_population_to_fit = HouseholdPopulation(
        composition_list, comp_dist, model_input_to_fit)
    rhs_to_fit = SEPIRRateEquations(model_input_to_fit,
                                    household_population_to_fit,
                                    NoImportModel(5,2))
    beta_ext = estimate_beta_ext(household_population_to_fit,
                                 rhs_to_fit,
                                 growth_rate)
    with open('outputs/mixing_sweep/beta_ext.pkl', 'wb') as f:
        dump(beta_ext, f)

prev=1.0e-5 # Starting prevalence
antibody_prev=0 # Starting antibody prev/immunity
gr_interval = [-0.5*this_spec['recovery_rate'], 1] # Interval used in growth rate estimation
gr_tol = 1e-3 # Absolute tolerance for growth rate estimation

class MixingAnalysis:
    def __init__(self):
        self.basic_spec = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}

    def __call__(self, p):
        try:
            result = self._implement_mixing(p)
        except ValueError as err:
            print(
                'Exception raised for parameters={0}\n\tException: {1}'.format(
                p, err)
                )
            return 0.0
        return result

    def _implement_mixing(self, p):
        print('p=',p)
        this_spec = deepcopy(self.basic_spec)
        this_spec['AR'] = p[0]
        model_input = SEPIRInput(this_spec, composition_list, comp_dist)
        model_input.k_home = (1 - p[1]) * model_input.k_home
        model_input.k_ext = (1 - p[2]) * beta_ext * model_input.k_ext

        household_population = HouseholdPopulation(
            composition_list, comp_dist, model_input)

        rhs = SEPIRRateEquations(model_input,
                                 household_population,
                                 NoImportModel(5,2))

        growth_rate = estimate_growth_rate(household_population,
                                           rhs,
                                           gr_interval,
                                           gr_tol)
        if growth_rate is None:
            growth_rate = 0

        H0 = make_initial_condition_by_eigenvector(growth_rate,
                                                   model_input,
                                                   household_population,
                                                   rhs,
                                                   prev,
                                                   antibody_prev)

        no_days = 100
        tspan = (0.0, no_days)
        solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

        t = solution.t
        H = solution.y

        ave_hh_size = household_population.composition_distribution.dot(
                            sum(household_population.composition_list, axis=1))

        I = (H.T.dot(household_population.states[:, 3::5])).sum(axis=1)/ \
                                                                    ave_hh_size
        R = (H.T.dot(household_population.states[:, 4::5])).sum(axis=1)/ \
                                                                    ave_hh_size
        R_end_vec = H[:, -1] * \
                        household_population.states[:, 4::5].sum(axis=1)
        attack_ratio = \
                    (household_population.state_to_comp_matrix.T.dot(R_end_vec))
        attack_ratio = model_input.composition_distribution.dot(
                                        attack_ratio / model_input.hh_size_list)

        recovered_states = where(
            ((rhs.states_sus_only + rhs.states_rec_only).sum(axis=1)
                    == household_population.states.sum(axis=1))
            & ((rhs.states_rec_only).sum(axis=1) > 0))[0]
        hh_outbreak_prop = H[recovered_states, -1].sum()

        peaks = 100 * max(I)
        R_end = 100 * R[-1]

        return [growth_rate, peaks, R_end, hh_outbreak_prop, attack_ratio]

def main(no_of_workers,
         sip_vals,
         internal_mix_vals,
         external_mix_vals):
    main_start = get_time()
    mixing_system = MixingAnalysis()
    results = []
    sip_range = arange(sip_vals[0], sip_vals[1], sip_vals[2])
    internal_mix_range = arange(internal_mix_vals[0],
                                internal_mix_vals[1],
                                internal_mix_vals[2])
    external_mix_range = arange(external_mix_vals[0],
                                external_mix_vals[1],
                                external_mix_vals[2])
    params = array([
        [a, i, e]
        for a in sip_range
        for i in internal_mix_range
        for e in external_mix_range])

    with Pool(no_of_workers) as pool:
        results = pool.map(mixing_system, params)

    growth_rate_data = array([r[0] for r in results]).reshape(
        len(sip_range),
        len(internal_mix_range),
        len(external_mix_range))
    peak_data = array([r[1] for r in results]).reshape(
        len(sip_range),
        len(internal_mix_range),
        len(external_mix_range))
    end_data = array([r[2] for r in results]).reshape(
        len(sip_range),
        len(internal_mix_range),
        len(external_mix_range))
    hh_prop_data = array([r[3] for r in results]).reshape(
        len(sip_range),
        len(internal_mix_range),
        len(external_mix_range))
    attack_ratio_data = array([r[4] for r in results]).reshape(
        len(sip_range),
        len(internal_mix_range),
        len(external_mix_range))

    fname = 'outputs/mixing_sweep/results.pkl'
    with open(fname, 'wb') as f:
        dump(
            (growth_rate_data,
             peak_data,
             end_data,
             hh_prop_data,
             attack_ratio_data,
             sip_range,
             internal_mix_range,
             external_mix_range),
            f)
    print('Parameter sweep took',get_time()-main_start,'seconds.')

    return -1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=8)
    parser.add_argument('--sip_vals', type=int, default=[0.25, 1.0, 0.25])
    parser.add_argument('--internal_mix_vals',
                        type=int,
                        default=[0.0, 0.99, 0.1])
    parser.add_argument('--external_mix_vals',
                        type=int,
                        default=[0.0, 0.99, 0.1])
    args = parser.parse_args()

    main(args.no_of_workers,
         args.sip_vals,
         args.internal_mix_vals,
         args.external_mix_vals)