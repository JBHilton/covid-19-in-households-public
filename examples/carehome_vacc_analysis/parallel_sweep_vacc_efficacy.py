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

sus_red = 1.0
death_red = 0.5
PATIENT_UPTAKE = 0.9
STAFF_UPTAKE = 0.8

scenario2composition = {
    0: array([[2, 1, 1]]),
    1: array([[5, 3, 2]])}

scenario2start_state = {
    0: [
        (2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0),
        (2,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0),
        (2,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0),
        (1,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0),
        (1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0),
        (1,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0)],
    1: [
        (5,0,0,0,0,0,3,0,0,0,0,0,2,0,0,0,0,0),
        (5,0,0,0,0,0,2,1,0,0,0,0,2,0,0,0,0,0),
        (5,0,0,0,0,0,3,0,0,0,0,0,1,1,0,0,0,0),
        (3,0,0,0,0,2,3,0,0,0,0,0,2,0,0,0,0,0),
        (3,0,0,0,0,2,2,1,0,0,0,0,2,0,0,0,0,0),
        (3,0,0,0,0,2,3,0,0,0,0,0,1,1,0,0,0,0)]}
# start_state_list = [(2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0),
#                    (1,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0)]
# start_state_list = [(2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0),
#                    (1,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0)]
# Proportion of care homes which are in each composition
comp_dist = array([1.0])

class DetermineCriticality:
    def __init__(self, scenario):

        self.scenario = scenario

        self.model_input = SEMCRDInput(
            SPEC, scenario2composition[scenario], comp_dist)

        self.baseline_population = HouseholdPopulation(
            self.model_input.composition_list,
            comp_dist,
            self.model_input)

        self.import_array = array([0,0,0])
        ''' Project baseline outbreak with no vaccination '''

        no_vacc_rhs = SEMCRDRateEquations(
            self.model_input,
            self.baseline_population,
            FixedImportModel(6, 2, self.import_array))

        start_state_list = scenario2start_state[scenario]

        '''Set start state weightings so ~1e-3 of homes have one staff member'''
        '''infectious'''
        start_state_weightings = [0.495,
                                  0.0025,
                                  0.0025,
                                  0.495,
                                  0.0025,
                                  0.0025]

        H0_no_vacc = simple_initialisation(
            self.baseline_population,
            no_vacc_rhs,
            start_state_list,
            start_state_weightings)

        ''' Check we have a valid initial condition'''
        if H0_no_vacc.sum()==1:
            print('Sucessfully built initial condition')
        else:
            print('Failed to build valid initial condition, sum(H0=)',H0_no_vacc.sum())

        self.H0 = H0_no_vacc

    def __call__(self, p):
        try:
            result = self._determine_criticality(p)
        except ValueError as err:
            print('Exception raised for parameters={0}\n\tException: {1}'.format(
                p, err))
            return 0.0
        return result

    def _determine_criticality(self, p):
        '''Assume vaccinated staff and agency workers are split evenly
        across vaccinated and unvaccinated homes'''

        this_spec = deepcopy(SPEC)
        this_spec['within_ch_contact'] = p[0]*this_spec['within_ch_contact']
        for diag_id in range(3):
            this_spec['within_ch_contact'][diag_id,diag_id] = 1

        self.model_input = SEMCRDInput(
            this_spec, scenario2composition[self.scenario], comp_dist)
        print('Did model input with p=',p)

        model_input_vacc = deepcopy(self.model_input)
        model_input_vacc.crit_prob[0] = \
            (PATIENT_UPTAKE*(1-death_red) + (1-PATIENT_UPTAKE)) \
            * model_input_vacc.crit_prob[0]
        model_input_vacc.inf_scales[0][0] = \
            p[1] \
            * model_input_vacc.inf_scales[0][0]
        model_input_vacc.crit_prob[1] = \
            (STAFF_UPTAKE  * (1-death_red) + (1-STAFF_UPTAKE)) \
            * model_input_vacc.crit_prob[1]
        model_input_vacc.inf_scales[0][1] = \
            p[2] \
            * model_input_vacc.inf_scales[0][1]
        model_input_vacc.crit_prob[2] = \
            (STAFF_UPTAKE  * (1-death_red) + (1-STAFF_UPTAKE)) \
            * model_input_vacc.crit_prob[2]
        model_input_vacc.inf_scales[0][2] = \
            p[2] \
            * model_input_vacc.inf_scales[0][2]
        model_input_vacc.sus = array([(1-PATIENT_UPTAKE) + (1-p[1]) * PATIENT_UPTAKE,
                                        (1 - STAFF_UPTAKE) + (1 - p[2]) * STAFF_UPTAKE ,
                                        (1 - STAFF_UPTAKE) + (1 - p[2]) * STAFF_UPTAKE]) * \
                                model_input_vacc.sus

        hh_pop_vacc = HouseholdPopulation(
            self.model_input.composition_list,
            comp_dist,
            model_input_vacc)

        rhs = SEMCRDRateEquations(
            model_input_vacc,
            hh_pop_vacc,
            NoImportModel(6, 3))

        tspan = (0.0, 1.0)
        solution = solve_ivp(rhs, tspan, self.H0, first_step=0.001, atol=1e-12)

        H = solution.y
        S = H.T.dot(hh_pop_vacc.states[:, 0]) + \
            H.T.dot(hh_pop_vacc.states[:, 6]) + \
            H.T.dot(hh_pop_vacc.states[:, 12])
        return (S[0] - S[-1])/S[0]



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

def main(no_of_workers, scenario):
    criticality = DetermineCriticality(scenario)
    results = []
    waifw_scale_range = [0.1, 1.0, 10.0]
    patient_eff_range = [0.0, 0.5, 1.0]
    staff_eff_range = [0.0, 0.5, 1.0]
    params = array([
        [b, p, s]
        for b in waifw_scale_range
        for p in patient_eff_range
        for s in staff_eff_range])
    #for p in params:
    #    results.append(compute_death_reduction(p))
    with Pool(no_of_workers) as pool:
        results = pool.map(criticality, params)


    criticality_data = array([r for r in results]).reshape(
        len(waifw_scale_range),
        len(patient_eff_range),
        len(staff_eff_range))

    fname = 'carehome_efficacy_sweep_data_s' + str(scenario) + '.pkl'
    with open(fname, 'wb') as f:
        dump(
            (criticality_data, params),
            f)

    print(criticality_data)
    return -1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=2)
    parser.add_argument('--scenario', type=int, default=0)
    args = parser.parse_args()

    main(args.no_of_workers, args.scenario)
