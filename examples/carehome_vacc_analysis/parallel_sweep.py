'''This sets up and runs a single solve of the care homes vaccine model'''
from copy import deepcopy
from numpy import array, hstack, ones, meshgrid, stack, zeros
from time import time as get_time
from scipy.integrate import solve_ivp
from model.preprocessing import (
    HouseholdPopulation, make_initial_condition)
from functions import THREE_CLASS_CH_EPI_SPEC, THREE_CLASS_CH_SPEC, SEMCRDInput, SEMCRDRateEquations, combine_household_populations
from model.imports import FixedImportModel, NoImportModel
from pickle import dump
from multiprocessing import Pool
# pylint: disable=invalid-name

NO_OF_WORKERS = 2
SPEC = {**THREE_CLASS_CH_EPI_SPEC,
        **THREE_CLASS_CH_SPEC}

vacc_efficacy = 0.7
vacc_inf_reduction = 0.5

# List of observed care home compositions
composition_list = array(
    [[2, 1, 1]])
# Proportion of care homes which are in each composition
comp_dist = array([1.0])

model_input = SEMCRDInput(SPEC, composition_list, comp_dist)

baseline_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

'''We initialise the model by solving for 10 years with no infection to reach
the equilibrium level of empty beds'''

no_inf_rhs = SEMCRDRateEquations(
    model_input,
    baseline_population,
    NoImportModel(6, 2))

no_inf_H0 = make_initial_condition(
    baseline_population, no_inf_rhs, 0)

tspan = (0.0, 10*365.0)
initialise_start = get_time()
solution = solve_ivp(no_inf_rhs, tspan, no_inf_H0, first_step=0.001)
initialise_end = get_time()

print(
    'Initialisation took ',
    initialise_end-initialise_start,
    ' seconds.')

H0 = hstack((solution.y[:,-1], solution.y[:,-1]))

import_array = (1e-5)*ones(2)

'''Now create populations with vaccine.'''

patient_uptake_range = [0.9, 0.95]
staff_uptake_range = [0.6, 0.8]
agency_uptake_range = [0.2, 0.8]
params = array([
    [p, s, a]
    for p in patient_uptake_range
    for s in staff_uptake_range
    for a in agency_uptake_range])

'''For this quick sketch we will record maximum critical and maximum empty beds
as output'''

def compute_c_and_d(p):
    '''Assume vaccinated staff and agency workers are split evenly
    across vaccinated and unvaccinated homes'''
    SPEC_UNVACC = deepcopy(SPEC)
    SPEC_UNVACC['critical_inf_prob'][1] = \
        p[1] \
        * (1-vacc_efficacy) \
        * SPEC_UNVACC['critical_inf_prob'][1]
    SPEC_UNVACC['mild_trans_scaling'][1] = \
        p[1] \
        * vacc_inf_reduction \
        * SPEC_UNVACC['mild_trans_scaling'][1]
    SPEC_UNVACC['critical_inf_prob'][2] = \
        p[2] \
        * (1-vacc_efficacy) \
        * SPEC_UNVACC['critical_inf_prob'][2]
    SPEC_UNVACC['mild_trans_scaling'][2] = \
        p[2] \
        * vacc_inf_reduction \
        * SPEC_UNVACC['mild_trans_scaling'][2]

    SPEC_VACC_P = deepcopy(SPEC)
    SPEC_VACC_P['critical_inf_prob'][0] = \
        (1-vacc_efficacy) \
        * SPEC_VACC_P['critical_inf_prob'][0]
    SPEC_VACC_P['mild_trans_scaling'][0] = \
        vacc_inf_reduction \
        * SPEC_VACC_P['mild_trans_scaling'][0]
    SPEC_VACC_P['critical_inf_prob'][1] = \
        p[1] \
        * (1 - vacc_efficacy) \
        * SPEC_VACC_P['critical_inf_prob'][1]
    SPEC_VACC_P['mild_trans_scaling'][1] = \
        p[1] \
        * vacc_inf_reduction \
        * SPEC_VACC_P['mild_trans_scaling'][1]
    SPEC_VACC_P['critical_inf_prob'][2] = \
        p[2] \
        * (1-vacc_efficacy) \
        * SPEC_VACC_P['critical_inf_prob'][2]
    SPEC_VACC_P['mild_trans_scaling'][2] = \
        p[2] \
        * vacc_inf_reduction \
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
        [1 - p[0], p[0]])

    rhs = SEMCRDRateEquations(
        model_input,
        combined_pop,
        FixedImportModel(6, 2, import_array))

    tspan = (0.0, 365.0)
    solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-12)

    H = solution.y
    C = H.T.dot(combined_pop.states[:, 3::6])
    D = H.T.dot(combined_pop.states[:, 5::6])
    return C.max(), D.max()

if __name__ == '__main__':
    with Pool(NO_OF_WORKERS) as pool:
        results = pool.map(compute_c_and_d, params)


    max_critical = array([r[0] for r in results]).reshape(
        len(patient_uptake_range),
        len(staff_uptake_range),
        len(agency_uptake_range))
    max_empty = array([r[1] for r in results]).reshape(
        len(patient_uptake_range),
        len(staff_uptake_range),
        len(agency_uptake_range))

    with open('carehome_sweep_data.pkl', 'wb') as f:
        dump((max_critical, max_empty, params, model_input), f)
