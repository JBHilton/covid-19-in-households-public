''' In this script we do projections  of the impact of support bubble policies
 by doing a 2D parameter sweep'''

from argparse import ArgumentParser
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from copy import deepcopy
from multiprocessing import Pool
from numpy import append, arange, array, exp, log, ones, sum, vstack, where
from numpy.linalg import eig
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.integrate import solve_ivp, trapezoid
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from model.preprocessing import (add_vuln_class, add_vulnerable_hh_members,
estimate_beta_ext, HouseholdPopulation, make_initial_condition_by_eigenvector,
map_SEPIR_to_SEPIRQ,SEPIRInput, SEPIRQInput)
from model.common import SEPIRRateEquations, SEPIRQRateEquations
from model.imports import NoImportModel
from model.specs import (TWO_AGE_UK_SPEC, TWO_AGE_EXT_SEPIRQ_SPEC,
TWO_AGE_INT_SEPIRQ_SPEC, TWO_AGE_SEPIR_SPEC_FOR_FITTING)

if isdir('outputs/oohi') is False:
    mkdir('outputs/oohi')

DOUBLING_TIME = 100
growth_rate = log(2) / DOUBLING_TIME

SEPIR_SPEC = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
EXT_SPEC = {**TWO_AGE_UK_SPEC, **TWO_AGE_EXT_SEPIRQ_SPEC}
INT_SPEC = {**TWO_AGE_UK_SPEC, **TWO_AGE_INT_SEPIRQ_SPEC}

relative_iso_rates = [1.0, 1.0, 0.5]


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
    model_input_to_fit = SEPIRInput(SEPIR_SPEC,
                                    fitting_comp_list,
                                    fitting_comp_dist)
    household_population_to_fit = HouseholdPopulation(
        fitting_comp_list, fitting_comp_dist, model_input_to_fit)
    print('number of states for 2-class pop is',
            household_population_to_fit.Q_int.shape)
    rhs_to_fit = SEPIRRateEquations(model_input_to_fit,
                                    household_population_to_fit,
                                    NoImportModel(5,2))
    beta_ext = estimate_beta_ext(household_population_to_fit,
                                 rhs_to_fit,
                                 growth_rate)
    with open('outputs/oohi/beta_ext.pkl', 'wb') as f:
        dump(beta_ext, f)

vuln_prop = 2.2/56
adult_class = 1

import_model = NoImportModel(6,3)

prev=1.0e-5 # Starting prevalence
starting_immunity=1e-2 # Starting antibody prev/immunity

class OOHIAnalysis:
    def __init__(self):
        self.pre_npi_input =  SEPIRInput(SEPIR_SPEC,
                                         composition_list,
                                         comp_dist)
        self.pre_npi_input.k_ext *= beta_ext
        self.pre_npi_input = add_vuln_class(self.pre_npi_input,
                            vuln_prop,
                            adult_class)
        self.pre_npi_household_population = HouseholdPopulation(
            composition_list, comp_dist, self.pre_npi_input)
        self.pre_npi_rhs = SEPIRRateEquations(
            self.pre_npi_input,
            self.pre_npi_household_population,
            import_model)
        self.H0_pre_npi = make_initial_condition_by_eigenvector(
                                            growth_rate,
                                            self.pre_npi_input,
                                            self.pre_npi_household_population,
                                            self.pre_npi_rhs,
                                            prev,
                                            starting_immunity)

    def __call__(self, p):
        print('now calling')
        try:
            result = self._simulate_isolation(p)
        except ValueError as err:
            print(
                'Exception raised for parameters={0}\n\tException: {1}'.format(
                p,
                err))
            return 0.0
        return result

    def _simulate_isolation(self, p):
        print('p=',p)

        if p[0]==0:
            this_spec = deepcopy(EXT_SPEC)
            this_spec['exp_iso_rate'] = p[1] * relative_iso_rates[0] * ones(2,)
            this_spec['pro_iso_rate'] = p[1] * relative_iso_rates[1] * ones(2,)
            this_spec['inf_iso_rate'] = p[1] * relative_iso_rates[2] * ones(2,)
            this_spec['ad_prob'] = p[2]
            model_input = SEPIRQInput(this_spec, composition_list, comp_dist)
            model_input.k_ext = beta_ext * model_input.k_ext
            model_input = add_vuln_class(model_input,
                                vuln_prop,
                                adult_class)
        else:
            this_spec = deepcopy(INT_SPEC)
            this_spec['exp_iso_rate'] = p[1] * relative_iso_rates[0] * ones(2,)
            this_spec['pro_iso_rate'] = p[1] * relative_iso_rates[1] * ones(2,)
            this_spec['inf_iso_rate'] = p[1] * relative_iso_rates[2] * ones(2,)
            this_spec['ad_prob'] = p[2]
            model_input = SEPIRQInput(this_spec, composition_list, comp_dist)
            model_input.k_ext = beta_ext * model_input.k_ext
            model_input = add_vuln_class(model_input,
                                vuln_prop,
                                adult_class)

        household_population = HouseholdPopulation(
            composition_list, comp_dist, model_input)
        rhs = SEPIRQRateEquations(
            model_input,
            household_population,
            import_model)

        map_matrix = map_SEPIR_to_SEPIRQ(self.pre_npi_household_population,
                                     household_population)
        H0_iso = self.H0_pre_npi * map_matrix

        no_days = 100
        tspan = (0.0, no_days)
        solution = solve_ivp(rhs,
                             tspan,
                             H0_iso,
                             first_step=0.001,
                             atol=1e-16)

        t = solution.t
        H = solution.y

        S = H.T.dot(household_population.states[:, ::6])
        E = H.T.dot(household_population.states[:, 1::6])
        P = H.T.dot(household_population.states[:, 2::6])
        I = H.T.dot(household_population.states[:, 3::6])
        R = H.T.dot(household_population.states[:, 4::6])

        if p[0] == 0:
            Q = H.T.dot(household_population.states[:, 5::6])
        else:
            states_iso_only = household_population.states[:,5::6]
            total_iso_by_state = states_iso_only.sum(axis=1)
            iso_present = total_iso_by_state>0
            Q = H[iso_present,:].T.dot(
                household_population.composition_by_state[iso_present,:])

        vuln_peak = 100 * max(I[:, 2]) / \
                        model_input.ave_hh_by_class[2]
        vuln_end = 100 * R[-1, 2] / \
                        model_input.ave_hh_by_class[2]
        iso_peak = 100 * max(Q.sum(axis=1)) / model_input.ave_hh_size
        Q_all = Q.sum(axis=1) / model_input.ave_hh_size
        cum_iso = 100 * trapezoid(Q_all, t)

        return [vuln_peak, vuln_end, iso_peak, cum_iso]

def main(no_of_workers,
         iso_rate_vals,
         iso_prob_vals):
    print('using',no_of_workers,'cores')
    oohi_system = OOHIAnalysis()
    results = []
    iso_method_range = [0, 1] # 0 is external, 1 is internal. We code this as integers to avoid problems with typing of the parameters.
    iso_rate_range = arange(iso_rate_vals[0],
                                iso_rate_vals[1],
                                iso_rate_vals[2])
    iso_prob_range = arange(iso_prob_vals[0],
                                iso_prob_vals[1],
                                iso_prob_vals[2])
    params = array([
        [m, b, e]
        for m in iso_method_range
        for b in iso_rate_range
        for e in iso_prob_range])

    print('about to pool')
    prepool_time = get_time()

    with Pool(no_of_workers) as pool:
        results = pool.map(oohi_system, params)

    print('Calculations took', (get_time()-prepool_time)/60, 'minutes.')

    vuln_peak_data = array([r[0] for r in results]).reshape(
        len(iso_method_range),
        len(iso_rate_range),
        len(iso_prob_range))
    vuln_end_data = array([r[1] for r in results]).reshape(
        len(iso_method_range),
        len(iso_rate_range),
        len(iso_prob_range))
    iso_peak_data = array([r[2] for r in results]).reshape(
        len(iso_method_range),
        len(iso_rate_range),
        len(iso_prob_range))
    cum_iso_data = array([r[3] for r in results]).reshape(
        len(iso_method_range),
        len(iso_rate_range),
        len(iso_prob_range))

    fname = 'outputs/oohi/results.pkl'
    with open(fname, 'wb') as f:
        dump(
            (vuln_peak_data,
             vuln_end_data,
             iso_peak_data,
             cum_iso_data,
             iso_method_range,
             iso_rate_range,
             iso_prob_range),
            f)

    return -1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=8)
    parser.add_argument('--iso_rate_vals', type=int, default=[0.2, 2.5, 0.1])
    parser.add_argument('--iso_prob_vals', type=int, default=[0.0, 1.0, 0.1])
    args = parser.parse_args()

    main(args.no_of_workers,
         args.iso_rate_vals,
         args.iso_prob_vals)
