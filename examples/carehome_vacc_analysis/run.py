'''This sets up and runs a single solve of the care homes vaccine model'''
from copy import copy
from numpy import array, hstack, ones
from time import time as get_time
from scipy.integrate import solve_ivp
from model.preprocessing import (
    SEPIRInput, HouseholdPopulation, make_initial_condition)
from functions import THREE_CLASS_CH_EPI_SPEC, THREE_CLASS_CH_SPEC, SEMCRDInput, SEMCRDRateEquations, combine_household_populations
from model.common import SEPIRRateEquations
from model.imports import FixedImportModel, NoImportModel
from pickle import load, dump
# pylint: disable=invalid-name

SPEC = {**THREE_CLASS_CH_EPI_SPEC,
        **THREE_CLASS_CH_SPEC}

vacc_efficacy = 0.9
vacc_inf_reduction = 0.5

# List of observed care home compositions
composition_list = array(
    [[2,1,1], [2,2,1], [2,2,2]])
# Proportion of care homes which are in each composition
comp_dist = array([0.4, 0.3, 0.3])

model_input = SEMCRDInput(SPEC, composition_list, comp_dist)

household_population = HouseholdPopulation(
    composition_list, comp_dist,model_input)

'''Now create populations with vaccine.'''

SPEC_VACC_P = copy(SPEC)
SPEC_VACC_P['critical_inf_prob'][0] = (1-vacc_efficacy) * \
                                    SPEC_VACC_P['critical_inf_prob'][0]
SPEC_VACC_P['mild_trans_scaling'][0] = vacc_inf_reduction * \
                                    SPEC_VACC_P['mild_trans_scaling'][0]
SPEC_VACC_PS = copy(SPEC)
SPEC_VACC_PS['critical_inf_prob'][0] = (1-vacc_efficacy) * \
                                    SPEC_VACC_PS['critical_inf_prob'][0]
SPEC_VACC_PS['critical_inf_prob'][1] = (1-vacc_efficacy) * \
                                    SPEC_VACC_PS['critical_inf_prob'][1]
SPEC_VACC_PS['mild_trans_scaling'][0] = vacc_inf_reduction * \
                                    SPEC_VACC_PS['mild_trans_scaling'][0]
SPEC_VACC_PS['mild_trans_scaling'][1] = vacc_inf_reduction * \
                                    SPEC_VACC_PS['mild_trans_scaling'][1]
SPEC_VACC_PSA = copy(SPEC)
SPEC_VACC_PSA['critical_inf_prob'][0] = (1-vacc_efficacy) * \
                                    SPEC_VACC_PSA['critical_inf_prob'][0]
SPEC_VACC_PSA['critical_inf_prob'][1] = (1-vacc_efficacy) * \
                                    SPEC_VACC_PSA['critical_inf_prob'][1]
SPEC_VACC_PSA['critical_inf_prob'][2] = (1-vacc_efficacy) * \
                                    SPEC_VACC_PSA['critical_inf_prob'][2]
SPEC_VACC_PSA['mild_trans_scaling'][0] = vacc_inf_reduction * \
                                    SPEC_VACC_PSA['mild_trans_scaling'][0]
SPEC_VACC_PSA['mild_trans_scaling'][1] = vacc_inf_reduction * \
                                    SPEC_VACC_PSA['mild_trans_scaling'][1]
SPEC_VACC_PSA['mild_trans_scaling'][2] = vacc_inf_reduction * \
                                    SPEC_VACC_PSA['mild_trans_scaling'][2]

model_input_vacc_P = SEMCRDInput(SPEC_VACC_P, composition_list, comp_dist)
model_input_vacc_PS = SEMCRDInput(SPEC_VACC_PS, composition_list, comp_dist)
model_input_vacc_PSA = SEMCRDInput(SPEC_VACC_PSA, composition_list, comp_dist)

hh_pop_P = HouseholdPopulation(composition_list,
                                comp_dist,
                                model_input_vacc_P)
hh_pop_PS = HouseholdPopulation(composition_list,
                                comp_dist,
                                model_input_vacc_PS)
hh_pop_PSA = HouseholdPopulation(composition_list,
                                comp_dist,
                                model_input_vacc_PSA)

uptake_by_class = array([0.9, 0.6, 0.25])                  # Percentage of each class vaccinated
weightings = hstack((array([1]), uptake_by_class)) - \
                hstack((uptake_by_class, array([0])))  # weighting for each home-level vaccine status is calculated to match uptake

combined_pop = combine_household_populations([household_population,
                                                hh_pop_P,
                                                hh_pop_PS,
                                                hh_pop_PSA],
                                                weightings)

'''We initialise the model by solving for 10 years with no infection to reach
the equilibrium level of empty beds'''

no_inf_rhs = SEMCRDRateEquations(
    model_input,
    combined_pop,
    NoImportModel(6,2))

no_inf_H0 = make_initial_condition(
    combined_pop, no_inf_rhs, 0)

tspan = (0.0, 10*365.0)
initialise_start = get_time()
solution = solve_ivp(no_inf_rhs, tspan, no_inf_H0, first_step=0.001)
initialise_end = get_time()

print(
    'Initialisation took ',
    initialise_end-initialise_start,
    ' seconds.')

H0 = solution.y[:,-1]

import_array = (1e-5)*ones(2)

rhs = SEMCRDRateEquations(
    model_input,
    combined_pop,
    FixedImportModel(6,2,import_array))

tspan = (0.0, 365.0)
solver_start = get_time()
solution = solve_ivp(rhs, tspan, H0, first_step=0.001,atol=1e-12)
solver_end = get_time()

print(
    'Solution took',
    solver_end - solver_start,
    'seconds.')

time = solution.t
H = solution.y
S = H.T.dot(combined_pop.states[:, ::6])
E = H.T.dot(combined_pop.states[:, 1::6])
M = H.T.dot(combined_pop.states[:, 2::6])
C = H.T.dot(combined_pop.states[:, 3::6])
R = H.T.dot(combined_pop.states[:, 4::6])
D = H.T.dot(combined_pop.states[:, 5::6])
time_series = {
'time':time,
'S':S,
'E':E,
'M':M,
'C':C,
'R':R,
'D':D
}


with open('carehome_vacc_output.pkl', 'wb') as f:
    dump((H, time_series, model_input), f)
