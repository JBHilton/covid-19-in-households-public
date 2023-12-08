''' In this script we do projections  of the impact of vaccination by doing a 2D parameter sweep'''

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
from model.preprocessing import ( AR_by_size, estimate_beta_ext,
        estimate_growth_rate, SEPIRInput, HouseholdPopulation,
        make_initial_condition_by_eigenvector)
from functions import HOSTEL_VACC_SEIR_SPEC, HostelSEIRInput
from model.common import SEPIRRateEquations
from model.imports import FixedImportModel, NoImportModel
from pickle import load, dump
# pylint: disable=invalid-name

if isdir('outputs/hostel_vacc') is False:
    mkdir('outputs/hostel_vacc')

SPEC = HOSTEL_VACC_SEIR_SPEC

vacc_efficacy = 0.5
vacc_inf_reduction = 0.5

# List of observed care home compositions
composition_list = []
for nv in range(21):
    composition_list.append([nv, 20-nv])
composition_list = array(composition_list)
# Proportion of care homes which are in each composition
comp_dist = array(composition_list.shape[0] * [1 / composition_list.shape[0]])

model_input = HOSTEL_VACC_SEIR_SPEC(SPEC, composition_list, comp_dist)

prev=0 # Starting prevalence
antibody_prev=0 # Starting antibody prev/immunity

class VaccAnalysis:
    def __init__(self):
        self.basic_spec = HOSTEL_VACC_SEIR_SPEC

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
        this_spec['sus'] = [1-p[0], 1] # p[0] is efficacy in terms of susceptibility reduction
        this_spec['inf_scales'] = [1-p[1], 1] # p[1] is efficacy in terms of 
        model_input = SEPIRInput(this_spec, composition_list, comp_dist)
        model_input.k_home = (1 - p[0]) * model_input.k_home
        model_input.k_ext = (1 - p[1]) * beta_ext * model_input.k_ext

        household_population = HouseholdPopulation(
            composition_list, comp_dist, model_input)

        rhs = SEPIRRateEquations(model_input,
                                 household_population,
                                 NoImportModel(5,2))

        growth_rate = estimate_growth_rate(household_population,
                                           rhs,
                                           gr_interval,
                                           gr_tol,
                                           x0=1e-3,
                                           r_min_discount=0.99)
        if growth_rate is None:
            growth_rate = 0

        H0, first_pass_ar = make_initial_condition_by_eigenvector(growth_rate,
                                                   model_input,
                                                   household_population,
                                                   rhs,
                                                   prev,
                                                   antibody_prev,
                                                   True)

        no_days = 90
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
        attack_ratio = 100 * model_input.composition_distribution.dot(
                                        attack_ratio / model_input.hh_size_list)

        recovered_states = where(
            ((rhs.states_sus_only + rhs.states_rec_only).sum(axis=1)
                    == household_population.states.sum(axis=1))
            & ((rhs.states_rec_only).sum(axis=1) > 0))[0]
        hh_outbreak_prop = 100 * H[recovered_states, -1].sum()

        peaks = 100 * max(I) #
        R_end = 100 * R[-1]

        ar_by_size = AR_by_size(household_population, H, R_comp)

        return [growth_rate,
                peaks,
                R_end,
                hh_outbreak_prop,
                attack_ratio,
                ar_by_size,
                first_pass_ar]

def main(no_of_workers,
         internal_mix_vals,
         external_mix_vals):
    main_start = get_time()
    mixing_system = MixingAnalysis()
    results = []
    internal_mix_range = arange(internal_mix_vals[0],
                                internal_mix_vals[1],
                                internal_mix_vals[2])
    external_mix_range = arange(external_mix_vals[0],
                                external_mix_vals[1],
                                external_mix_vals[2])
    params = array([
        [i, e]
        for i in internal_mix_range
        for e in external_mix_range])

    with Pool(no_of_workers) as pool:
        results = pool.map(mixing_system, params)


    print('Parameter sweep took',get_time()-main_start,'seconds.')

    growth_rate_data = array([r[0] for r in results]).reshape(
        len(internal_mix_range),
        len(external_mix_range))
    peak_data = array([r[1] for r in results]).reshape(
        len(internal_mix_range),
        len(external_mix_range))
    end_data = array([r[2] for r in results]).reshape(
        len(internal_mix_range),
        len(external_mix_range))
    hh_prop_data = array([r[3] for r in results]).reshape(
        len(internal_mix_range),
        len(external_mix_range))
    attack_ratio_data = array([r[4] for r in results]).reshape(
        len(internal_mix_range),
        len(external_mix_range))
    ar_by_size_data = array([r[5] for r in results]).reshape(
        len(internal_mix_range),
        len(external_mix_range),
        model_input_to_fit.max_hh_size)
    first_pass_ar_data = array([r[6] for r in results]).reshape(
        len(internal_mix_range),
        len(external_mix_range),
        model_input_to_fit.max_hh_size)

    fname = 'outputs/mixing_sweep/results.pkl'
    with open(fname, 'wb') as f:
        dump(
            (growth_rate_data,
             peak_data,
             end_data,
             hh_prop_data,
             attack_ratio_data,
             ar_by_size_data,
             first_pass_ar_data,
             internal_mix_range,
             external_mix_range),
            f)

    return -1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=8)
    parser.add_argument('--internal_mix_vals',
                        type=int,
                        default=[0.0, 0.99, 0.05])
    parser.add_argument('--external_mix_vals',
                        type=int,
                        default=[0.0, 0.99, 0.05])
    args = parser.parse_args()

    main(args.no_of_workers,
         args.internal_mix_vals,
         args.external_mix_vals)
