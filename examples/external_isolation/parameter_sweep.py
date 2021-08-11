''' In this script we do projections  of the impact of support bubble policies
 by doing a 2D parameter sweep'''

from argparse import ArgumentParser
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from copy import deepcopy
from multiprocessing import Pool
from numpy import append, arange, array, exp, log, sum, vstack, where
from numpy.linalg import eig
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.integrate import solve_ivp
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from model.preprocessing import ( build_support_bubbles, estimate_beta_ext, estimate_growth_rate,
        SEPIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations
from model.imports import NoImportModel

if isdir('outputs/oohi') is False:
    mkdir('outputs/oohi')

DOUBLING_TIME = 100
growth_rate = log(2) / DOUBLING_TIME


ext_spec = {**TWO_AGE_UK_SPEC, **TWO_AGE_EXT_SEPIRQ_SPEC}
int_spec = {**TWO_AGE_UK_SPEC, **TWO_AGE_INT_SEPIRQ_SPEC}

# List of observed household compositions
composition_list = read_csv(
    'inputs/eng_and_wales_adult_child_vuln_composition_list.csv',
    header=0).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/eng_and_wales_adult_child_vuln_composition_dist.csv',
    header=0).to_numpy().squeeze()

if isfile('outputs/oohi/beta_ext.pkl') is True:
    with open('outputs/oohi/beta_ext.pkl', 'rb') as f:
        beta_ext = load(f)
else:
    # List of observed household compositions
    fitting_comp_list = read_csv(
        'inputs/eng_and_wales_adult_child_composition_list.csv',
        header=0).to_numpy()
    # Proportion of households which are in each composition
    fitting_comp_dist = read_csv(
        'inputs/eng_and_wales_adult_child_composition_dist.csv',
        header=0).to_numpy().squeeze()
    model_input_to_fit = SEPIRInput(sepir_sepc, fitting_comp_list, fitting_comp_dist)
    household_population_to_fit = HouseholdPopulation(
        fitting_comp_list, fitting_comp_dist, model_input_to_fit)
    print('number of states for 2-class pop is',household_population_to_fit.Q_int.shape)
    rhs_to_fit = SEPIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(5,2))
    beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
    with open('outputs/oohi/beta_ext.pkl', 'wb') as f:
        dump(beta_ext, f)

vuln_prop = 2.2/56
adult_class = 1

pre_npi_input =  SEPIRInput(sepir_sepc, composition_list, comp_dist)
pre_npi_input.k_ext *= beta_ext
pre_npi_input = add_vuln_class(pre_npi_input,
                    vuln_prop,
                    adult_class)

prev=1.0e-5 # Starting prevalence
starting_immunity=0 # Starting antibody prev/immunity

class BubbleAnalysis:
    def __init__(self):
        basic_mixed_comp_list, basic_mixed_comp_dist = build_support_bubbles(
                composition_list,
                comp_dist,
                MAX_ADULTS,
                MAX_BUBBLE_SIZE,
                1)

        basic_mixed_comp_dist = basic_mixed_comp_dist/sum(basic_mixed_comp_dist)
        self.basic_bubbled_input = SEPIRInput(SPEC, basic_mixed_comp_list, basic_mixed_comp_dist)

        self.bubbled_household_population = HouseholdPopulation(
            basic_mixed_comp_list, basic_mixed_comp_dist, self.basic_bubbled_input)

        rhs = SEPIRRateEquations(self.basic_bubbled_input, self.bubbled_household_population, NoImportModel(5,2))

    def __call__(self, p):
        print('now calling')
        try:
            result = self._implement_bubbles(p)
        except ValueError as err:
            print('Exception raised for parameters={0}\n\tException: {1}'.format(
                p, err))
            return 0.0
        return result

    def _implement_bubbles(self, p):
        print('p=',p)
        mixed_comp_list, mixed_comp_dist = build_support_bubbles(
                composition_list,
                comp_dist,
                MAX_ADULTS,
                MAX_BUBBLE_SIZE,
                p[0])

        mixed_comp_dist = mixed_comp_dist/sum(mixed_comp_dist)
        bubbled_model_input = deepcopy(self.basic_bubbled_input)
        bubbled_model_input.composition_distribution = mixed_comp_dist
        bubbled_model_input.k_ext = \
                        (1 - p[1]) * beta_ext * bubbled_model_input.k_ext

        household_population = deepcopy(self.bubbled_household_population)
        household_population.composition_distribution = mixed_comp_dist

        rhs = SEPIRRateEquations(bubbled_model_input,
                                 household_population,
                                 NoImportModel(5,2))

        print('calculating growth rate, p=',p)

        growth_rate = estimate_growth_rate(household_population,
                                           rhs,
                                           [-0.5*SPEC['recovery_rate'], 1],
                                           1e-2,
                                           (1 - p[1]) * X0)
        if growth_rate is None:
            growth_rate = 0

        print('initialising, p=',p)

        H0 = make_initial_condition_by_eigenvector(growth_rate,
                                                   bubbled_model_input,
                                                   household_population,
                                                   rhs, prev,
                                                   starting_immunity)

        print('solving, p=',p)

        no_days = 100
        tspan = (0.0, no_days)
        solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

        print('calculating output, p=',p)

        t = solution.t
        H = solution.y

        I = (H.T.dot(household_population.states[:, 3::5])).sum(axis=1)/ \
                                                household_population.ave_hh_size
        R = (H.T.dot(household_population.states[:, 4::5])).sum(axis=1)/ \
                                                household_population.ave_hh_size

        peaks = 100 * max(I)
        R_end = 100 * R[-1]

        return [growth_rate, peaks, R_end]

def main(no_of_workers,
         bubble_prob_vals,
         external_mix_vals):
    bubble_system = BubbleAnalysis()
    print('built bubble system')
    results = []
    bubble_prob_range = arange(bubble_prob_vals[0],
                                bubble_prob_vals[1],
                                bubble_prob_vals[2])
    external_mix_range = arange(external_mix_vals[0],
                                external_mix_vals[1],
                                external_mix_vals[2])
    params = array([
        [b, e]
        for b in bubble_prob_range
        for e in external_mix_range])

    print('about to pool')
    prepool_time = get_time()

    with Pool(no_of_workers) as pool:
        results = pool.map(bubble_system, params)

    print('Calculations took', (get_time()-prepool_time)/60, 'minutes.')

    growth_rate_data = array([r[0] for r in results]).reshape(
        len(bubble_prob_range),
        len(external_mix_range))
    peak_data = array([r[1] for r in results]).reshape(
        len(bubble_prob_range),
        len(external_mix_range))
    end_data = array([r[2] for r in results]).reshape(
        len(bubble_prob_range),
        len(external_mix_range))

    fname = 'outputs/oohi/results.pkl'
    with open(fname, 'wb') as f:
        dump(
            (growth_rate_data,
             peak_data,
             end_data,
             bubble_prob_range,
             external_mix_range),
            f)

    return -1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=4)
    parser.add_argument('--det_rate_vals', type=int, default=[0.0, 1.0, 0.25])
    parser.add_argument('--iso_prob_vals', type=int, default=[0.0, 1.0, 0.25])
    args = parser.parse_args()

    main(args.no_of_workers,
         args.bubble_prob_vals,
         args.external_mix_vals)
