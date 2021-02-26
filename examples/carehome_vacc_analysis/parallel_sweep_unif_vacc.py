'''This sets up and runs a single solve of the care homes vaccine model'''
from copy import deepcopy
from argparse import ArgumentParser
from numpy import array, hstack, ones, meshgrid, stack, zeros
from time import time as get_time
from scipy.integrate import solve_ivp
from model.preprocessing import (
    HouseholdPopulation, make_initial_condition)
from functions import THREE_CLASS_CH_EPI_SPEC, THREE_CLASS_CH_SPEC, SEMCRDInput, SEMCRDRateEquations, combine_household_populations, simple_initialisation
from model.imports import FixedImportModel, NoImportModel
from pickle import dump
from multiprocessing import Pool
# pylint: disable=invalid-name

SPEC = {**THREE_CLASS_CH_EPI_SPEC,
        **THREE_CLASS_CH_SPEC}

sus_red = 0.5
death_red = 0.5
PATIENT_UPTAKE = 0.9
UNSCALED_IMPORT_ARRAY = array([0,1,1])

scenario2composition = {
    0: array([[2, 1, 1]]),
    1: array([[5, 3, 2]])}

scenario2start_state = {
    0: [
        (2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0),
        (1,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0)],
    1: [
        (5,0,0,0,0,0,3,0,0,0,0,0,2,0,0,0,0,0),
        (3,0,0,0,0,2,3,0,0,0,0,0,2,0,0,0,0,0)]}
# start_state_list = [(2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0),
#                    (1,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0)]
# start_state_list = [(2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0),
#                    (1,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0)]
# Proportion of care homes which are in each composition
comp_dist = array([1.0])

class DeathReductionComputation:
    def __init__(self, i_scale, scenario):
        self.model_input = SEMCRDInput(
            SPEC, scenario2composition[scenario], comp_dist)

        self.baseline_population = HouseholdPopulation(
            self.model_input.composition_list,
            comp_dist,
            self.model_input)

        self.import_array = i_scale*UNSCALED_IMPORT_ARRAY
        ''' Project baseline outbreak with no vaccination '''

        no_vacc_rhs = SEMCRDRateEquations(
            self.model_input,
            self.baseline_population,
            FixedImportModel(6, 2, self.import_array))

        start_state_list = scenario2start_state[scenario]

        start_state_weightings = [0.5, 0.5]

        H0_no_vacc = simple_initialisation(
            self.baseline_population,
            no_vacc_rhs,
            start_state_list,
            start_state_weightings)

        ''' Check we have a valid initial condition'''
        if H0_no_vacc.sum()==1:
            print('Sucessfully built initial condition')
        else:
            print('Failed to build valid initial condition')

        tspan = (0.0, 4*30.0)
        no_vacc_start = get_time()
        solution = solve_ivp(no_vacc_rhs, tspan, H0_no_vacc, first_step=0.001, atol=1e-12)
        no_vacc_end = get_time()
        print(
            'Run without vaccination took ',
            no_vacc_end-no_vacc_start,
            ' seconds.')

        H_no_vacc = solution.y
        D_P_no_vacc = H_no_vacc.T.dot(self.baseline_population.states[:, 5])
        self.D_P_final_no_vacc = D_P_no_vacc[-1]

        no_vacc_output ={
        't': solution.t,
        'S_no_vacc': H_no_vacc.T.dot(self.baseline_population.states[:, ::6]),
        'E_no_vacc': H_no_vacc.T.dot(self.baseline_population.states[:, 1::6]),
        'M_no_vacc': H_no_vacc.T.dot(self.baseline_population.states[:, 2::6]),
        'C_no_vacc': H_no_vacc.T.dot(self.baseline_population.states[:, 3::6]),
        'R_no_vacc': H_no_vacc.T.dot(self.baseline_population.states[:, 4::6]),
        'D_no_vacc': H_no_vacc.T.dot(self.baseline_population.states[:, 5::6])
        }
        fname = 'carehome_no_vacc_sol_{0:f}_s{1:d}.pkl'
        with open(fname.format(i_scale, scenario),'wb') as f:
            dump((no_vacc_output, self.model_input.ave_hh_by_class),f)

        self.H0 = H0_no_vacc

    '''For this quick sketch we will record maximum critical and maximum empty beds
    as output'''
    def __call__(self, p):
        try:
            result = self._compute_death_reduction(p)
        except ValueError as err:
            print('Exception raised for parameters={0}\n\tException: {1}'.format(
                p, err))
            return 0.0
        return result

    def _compute_death_reduction(self, p):
        '''Assume vaccinated staff and agency workers are split evenly
        across vaccinated and unvaccinated homes'''

        model_input_vacc = deepcopy(self.model_input)
        model_input_vacc.crit_prob[0] = \
            (PATIENT_UPTAKE*(1-death_red) + (1-PATIENT_UPTAKE)) \
            * model_input_vacc.crit_prob[0]
        model_input_vacc.inf_scales[0][0] = \
            p[0] \
            * model_input_vacc.inf_scales[0][0]
        model_input_vacc.crit_prob[1] = \
            (p[1]  * (1-death_red) + (1-p[1])) \
            * model_input_vacc.crit_prob[1]
        model_input_vacc.inf_scales[0][1] = \
            p[0] \
            * model_input_vacc.inf_scales[0][1]
        model_input_vacc.crit_prob[2] = \
            (p[2]  * (1-death_red) + (1-p[2])) \
            * model_input_vacc.crit_prob[2]
        model_input_vacc.inf_scales[0][2] = \
            p[0] \
            * model_input_vacc.inf_scales[0][2]
        model_input_vacc.sus = array([1-sus_red,
                                        (1 - p[1]) + (1 - sus_red) * p[1] ,
                                        (1 - p[2]) + (1 - sus_red) * p[2]]) * \
                                model_input_vacc.sus

        hh_pop_vacc = HouseholdPopulation(
            self.model_input.composition_list,
            comp_dist,
            model_input_vacc)

        rhs = SEMCRDRateEquations(
            model_input_vacc,
            hh_pop_vacc,
            FixedImportModel(6, 2, self.import_array))

        tspan = (0.0, 4*30.0)
        solution = solve_ivp(rhs, tspan, self.H0, first_step=0.001, atol=1e-12)

        H = solution.y
        D_P = H.T.dot(hh_pop_vacc.states[:, 5])
        return (self.D_P_final_no_vacc - D_P[-1]) / self.D_P_final_no_vacc



# '''We initialise the model by solving for 10 years with no infection to reach
# the equilibrium level of empty beds'''
#
# no_inf_rhs = SEMCRDRateEquations(
#     model_input,
#     baseline_population,
#     NoImportModel(6, 2))
#
# no_inf_H0 = make_initial_condition(
#     baseline_population, no_inf_rhs, 0)
#
# tspan = (0.0, 10*365.0)
# initialise_start = get_time()
# solution = solve_ivp(no_inf_rhs, tspan, no_inf_H0, first_step=0.001)
# initialise_end = get_time()
#
# print(
#     'Initialisation took ',
#     initialise_end-initialise_start,
#     ' seconds.')

# H0_no_vacc = solution.y[:,-1]
# H0 = hstack((solution.y[:,-1], solution.y[:,-1]))

def main(i_scale, no_of_workers, scenario):
    compute_death_reduction = DeathReductionComputation(
        i_scale, scenario)
    results = []
    inf_red_range = [0.6]
    staff_uptake_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    agency_uptake_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    params = array([
        [r, s, a]
        for r in inf_red_range
        for s in staff_uptake_range
        for a in agency_uptake_range])
    #for p in params:
    #    results.append(compute_death_reduction(p))
    with Pool(no_of_workers) as pool:
        results = pool.map(compute_death_reduction, params)


    death_reduction_data = array([r for r in results]).reshape(
        len(inf_red_range),
        len(staff_uptake_range),
        len(agency_uptake_range))

    fname = 'carehome_sweep_data_import_scale_{0:f}_s{1:d}.pkl'
    with open(fname.format(i_scale, scenario), 'wb') as f:
        dump(
            (
                death_reduction_data,
                params),
            f)

    print(death_reduction_data)
    return -1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('i_scale', type=float)
    parser.add_argument('--no_of_workers', type=int, default=2)
    parser.add_argument('--scenario', type=int, default=0)
    args = parser.parse_args()

    main(args.i_scale, args.no_of_workers, args.scenario)
