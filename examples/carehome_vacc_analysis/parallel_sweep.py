'''This sets up and runs a single solve of the care homes vaccine model'''
from copy import deepcopy
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

NO_OF_WORKERS = 2
SPEC = {**THREE_CLASS_CH_EPI_SPEC,
        **THREE_CLASS_CH_SPEC}

sus_red = 0.5
death_red = 0.9
PATIENT_UPTAKE = 0.9
IMPORT_ARRAY = (1e-5)*ones(2)

# List of observed care home compositions
composition_list = array(
    [[5, 3, 2]])
# Proportion of care homes which are in each composition
comp_dist = array([1.0])

class DeathReductionComputation:
    def __init__(self):
        self.model_input = SEMCRDInput(SPEC, composition_list, comp_dist)

        self.baseline_population = HouseholdPopulation(
            composition_list, comp_dist, self.model_input)
        ''' Project baseline outbreak with no vaccination '''

        no_vacc_rhs = SEMCRDRateEquations(
            self.model_input,
            self.baseline_population,
            FixedImportModel(6, 2, IMPORT_ARRAY))

        # start_state_list = [(5,0,0,0,0,0,3,0,0,0,0,0,2,0,0,0,0,0),
                            # (3,0,0,0,0,2,3,0,0,0,0,0,2,0,0,0,0,0)]
        start_state_list = [(2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0),
                            (1,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0)]
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

        self.H0 = hstack((
            (1-PATIENT_UPTAKE)*H0_no_vacc,
            PATIENT_UPTAKE*H0_no_vacc))

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
        SPEC_UNVACC = deepcopy(SPEC)
        SPEC_UNVACC['critical_inf_prob'][1] = \
            (p[1] * (1-death_red) + (1-p[1])) \
            * SPEC_UNVACC['critical_inf_prob'][1]
        SPEC_UNVACC['sus'][1] = \
            (p[1]  * (1-sus_red) + (1-p[1])) \
            * SPEC_UNVACC['sus'][1]
        SPEC_UNVACC['mild_trans_scaling'][1] = \
            p[1] \
            * p[0] \
            * SPEC_UNVACC['mild_trans_scaling'][1]
        SPEC_UNVACC['critical_inf_prob'][2] = \
            (p[2] * (1-death_red) + (1-p[2])) \
            * SPEC_UNVACC['critical_inf_prob'][2]
        SPEC_UNVACC['sus'][2] = \
            (p[2]  * (1-sus_red) + (1-p[2])) \
            * SPEC_UNVACC['sus'][2]
        SPEC_UNVACC['mild_trans_scaling'][2] = \
            p[2] \
            * p[0] \
            * SPEC_UNVACC['mild_trans_scaling'][2]

        SPEC_VACC_P = deepcopy(SPEC)
        SPEC_VACC_P['critical_inf_prob'][0] = \
            (1-death_red) \
            * SPEC_VACC_P['critical_inf_prob'][0]
        SPEC_VACC_P['sus'][0] = \
            (1-sus_red) \
            * SPEC_VACC_P['sus'][0]
        SPEC_VACC_P['mild_trans_scaling'][0] = \
            p[0] \
            * SPEC_VACC_P['mild_trans_scaling'][0]
        SPEC_VACC_P['critical_inf_prob'][1] = \
            (p[1]  * (1-death_red) + (1-p[1])) \
            * SPEC_VACC_P['critical_inf_prob'][1]
        SPEC_VACC_P['sus'][1] = \
            (p[1]  * (1-sus_red) + (1-p[1])) \
            * SPEC_UNVACC['sus'][1]
        SPEC_VACC_P['mild_trans_scaling'][1] = \
            p[1] \
            * p[0] \
            * SPEC_VACC_P['mild_trans_scaling'][1]
        SPEC_VACC_P['critical_inf_prob'][2] = \
            (p[2]  * (1-death_red) + (1-p[2])) \
            * SPEC_VACC_P['critical_inf_prob'][2]
        SPEC_VACC_P['sus'][2] = \
            (p[2]  * (1-sus_red) + (1-p[2])) \
            * SPEC_UNVACC['sus'][2]
        SPEC_VACC_P['mild_trans_scaling'][2] = \
            p[2] \
            * p[0] \
            * SPEC_VACC_P['mild_trans_scaling'][2]

        model_input_unvacc = SEMCRDInput(SPEC_UNVACC, composition_list, comp_dist)
        model_input_vacc_P = SEMCRDInput(SPEC_VACC_P, composition_list, comp_dist)

        hh_pop_unvacc = HouseholdPopulation(
            composition_list,
            comp_dist,
            model_input_unvacc)
        hh_pop_P = HouseholdPopulation(
            composition_list,
            comp_dist,
            model_input_vacc_P)

        combined_pop = combine_household_populations(
            [hh_pop_unvacc, hh_pop_P],
            [1 - PATIENT_UPTAKE, PATIENT_UPTAKE])

        rhs = SEMCRDRateEquations(
            self.model_input,
            combined_pop,
            FixedImportModel(6, 2, IMPORT_ARRAY))

        tspan = (0.0, 4*30.0)
        solution = solve_ivp(rhs, tspan, self.H0, first_step=0.001, atol=1e-12)

        H = solution.y
        D_P = H.T.dot(combined_pop.states[:, 5])
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


if __name__ == '__main__':
    compute_death_reduction = DeathReductionComputation()
    results = []
    inf_red_range = [0.0, 0.7, 1.0]
    staff_uptake_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    agency_uptake_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    params = array([
        [r, s, a]
        for r in inf_red_range
        for s in staff_uptake_range
        for a in agency_uptake_range])
    #for p in params:
    #    results.append(compute_death_reduction(p))
    with Pool(NO_OF_WORKERS) as pool:
        results = pool.map(compute_death_reduction, params)


    death_reduction_data = array([r for r in results]).reshape(
        len(inf_red_range),
        len(staff_uptake_range),
        len(agency_uptake_range))

    with open('carehome_sweep_data.pkl', 'wb') as f:
        dump(
            (
                death_reduction_data,
                params,
                compute_death_reduction.model_input),
            f)

    print(death_reduction_data)
