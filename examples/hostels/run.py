''' In this script we do projections  of the impact of vaccination by doing a 2D parameter sweep'''

from argparse import ArgumentParser
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from copy import deepcopy
from multiprocessing import Pool
from numpy import arange, array, exp, log, ones, sum, where
from numpy.linalg import eig
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.stats import binom
from scipy.integrate import solve_ivp
from model.preprocessing import ( AR_by_size, estimate_beta_ext,
        estimate_growth_rate, SEPIRInput, HouseholdPopulation,
        make_initial_condition_by_eigenvector)
from examples.hostels.functions import HOSTEL_VACC_SEIR_SPEC, HostelSEIRInput
from model.common import SEIRRateEquations
from model.imports import CoupledSEIRImports, FixedImportModel, NoImportModel
from pickle import load, dump
# pylint: disable=invalid-name

no_days = 90

# Assume external prevalence is fixed at 10% and comes in weekly
wks = arange(1, no_days+7, 7)
prev_curve = 0.1 * ones(len(wks),)

if isdir('outputs/hostel_vacc') is False:
    mkdir('outputs/hostel_vacc')

HH_SIZE = 20 # Total size of households

SPEC = HOSTEL_VACC_SEIR_SPEC
R_comp = 4

# List of observed care home compositions
composition_list = []
for nv in range(HH_SIZE + 1):
    composition_list.append([nv, HH_SIZE - nv])
composition_list = array(composition_list)
# Proportion of care homes which are in each composition
comp_dist = array(composition_list.shape[0] * [1 / composition_list.shape[0]])

model_input = HostelSEIRInput(SPEC, composition_list, comp_dist)

prev=0 # Starting prevalence
antibody_prev=0 # Starting antibody prev/immunity

class VaccAnalysis:
    def __init__(self):
        self.basic_spec = HOSTEL_VACC_SEIR_SPEC

    def __call__(self, p):
        try:
            result = self._implement_vacc(p)
        except ValueError as err:
            print(
                'Exception raised for parameters={0}\n\tException: {1}'.format(
                p, err)
                )
            return 0.0
        return result

    def _implement_vacc(self, p):

        # For now I'll assume that p[0] is uptake and p[1] is connection to external population
        print('p=',p)
        this_comp_dist = binom.pmf(arange(HH_SIZE, 0, -1), HH_SIZE, p[0])
        this_spec = deepcopy(basic_spec)
        this_spec['beta_ext'] = p[1]
        model_input = HostelSEIRInput(this_spec, composition_list, this_comp_dist)

        household_population = HouseholdPopulation(
            composition_list, this_comp_dist, model_input)

        rhs = SEIRRateEquations(model_input,
                                 household_population,
                                 CoupledSEIRImports())

        # Initialise with zero infecteds - shouldn't matter what we use for growth rate
        H0, first_pass_ar = make_initial_condition_by_eigenvector(1e-2,
                                                   model_input,
                                                   household_population,
                                                   rhs,
                                                   prev,
                                                   antibody_prev,
                                                   True)

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

        return [peaks,
                R_end,
                hh_outbreak_prop,
                attack_ratio,
                ar_by_size,
                first_pass_ar]

def main(no_of_workers,
         uptake_vals,
         external_mix_vals):
    main_start = get_time()
    mixing_system = VaccAnalysis()
    results = []
    uptake_range = arange(uptake_vals[0],
                                uptake_vals[1],
                                uptake_vals[2])
    external_mix_range = arange(external_mix_vals[0],
                                external_mix_vals[1],
                                external_mix_vals[2])
    params = array([
        [i, e]
        for i in uptake_range
        for e in external_mix_range])

    with Pool(no_of_workers) as pool:
        results = pool.map(mixing_system, params)


    print('Parameter sweep took',get_time()-main_start,'seconds.')

    peak_data = array([r[0] for r in results]).reshape(
        len(uptake_range),
        len(external_mix_range))
    end_data = array([r[1] for r in results]).reshape(
        len(uptake_range),
        len(external_mix_range))
    hh_prop_data = array([r[2] for r in results]).reshape(
        len(uptake_range),
        len(external_mix_range))
    attack_ratio_data = array([r[3] for r in results]).reshape(
        len(uptake_range),
        len(external_mix_range))

    fname = 'outputs/hostel_vacc/results.pkl'
    with open(fname, 'wb') as f:
        dump(
            (peak_data,
             end_data,
             hh_prop_data,
             attack_ratio_data,
             uptake_range,
             external_mix_range),
            f)

    return -1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=8)
    parser.add_argument('--uptake_vals',
                        type=int,
                        default=[0.0, 0.99, 0.05])
    parser.add_argument('--external_mix_vals',
                        type=int,
                        default=[0.0, 0.99, 0.05])
    args = parser.parse_args()

    main(args.no_of_workers,
         args.uptake_vals,
         args.external_mix_vals)
