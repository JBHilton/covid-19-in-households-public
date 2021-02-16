'''This sets up and runs a single solve of the care homes vaccine model'''
from copy import copy
from numpy import array, hstack, ones, vstack, zeros
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

vacc_efficacy = 0.7
vacc_inf_reduction = 0.5

# List of observed care home compositions
composition_list = array(
    [[2,1,1]])
# Proportion of care homes which are in each composition
comp_dist = array([1.0])

model_input = SEMCRDInput(SPEC, composition_list, comp_dist)

baseline_population = HouseholdPopulation(
    composition_list, comp_dist,model_input)

'''We initialise the model by solving for 10 years with no infection to reach
the equilibrium level of empty beds'''

no_inf_rhs = SEMCRDRateEquations(
    model_input,
    baseline_population,
    NoImportModel(6,2))

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

'''For this quick sketch we will record maximum critical and maximum empty beds
as output'''

max_critical = zeros(shape=[len(patient_uptake_range),
                            len(staff_uptake_range),
                            len(agency_uptake_range)])
max_empty = zeros(shape=[len(patient_uptake_range),
                            len(staff_uptake_range),
                            len(agency_uptake_range)])

for i in range(len(patient_uptake_range)):
    for j in range(len(staff_uptake_range)):
        for k in range(len(agency_uptake_range)):

            '''Assume vaccinated staff and agency workers are split evenly
            across vaccinated and unvaccinated homes'''
            SPEC_UNVACC = copy(SPEC)
            SPEC_UNVACC['critical_inf_prob'][1] = staff_uptake_range[j] * \
                                                    (1-vacc_efficacy) * \
                                            SPEC_UNVACC['critical_inf_prob'][1]
            SPEC_UNVACC['mild_trans_scaling'][1] = staff_uptake_range[j] * \
                                                    vacc_inf_reduction * \
                                            SPEC_UNVACC['mild_trans_scaling'][1]
            SPEC_UNVACC['critical_inf_prob'][2] = agency_uptake_range[k] * \
                                                    (1-vacc_efficacy) * \
                                            SPEC_UNVACC['critical_inf_prob'][2]
            SPEC_UNVACC['mild_trans_scaling'][2] = agency_uptake_range[k] * \
                                                    vacc_inf_reduction * \
                                            SPEC_UNVACC['mild_trans_scaling'][2]

            SPEC_VACC_P = copy(SPEC)
            SPEC_VACC_P['critical_inf_prob'][0] = (1-vacc_efficacy) * \
                                            SPEC_VACC_P['critical_inf_prob'][0]
            SPEC_VACC_P['mild_trans_scaling'][0] = vacc_inf_reduction * \
                                            SPEC_VACC_P['mild_trans_scaling'][0]
            SPEC_VACC_P['critical_inf_prob'][1] = staff_uptake_range[j] * \
                                                    (1-vacc_efficacy) * \
                                            SPEC_VACC_P['critical_inf_prob'][1]
            SPEC_VACC_P['mild_trans_scaling'][1] = staff_uptake_range[j] * \
                                                    vacc_inf_reduction * \
                                            SPEC_VACC_P['mild_trans_scaling'][1]
            SPEC_VACC_P['critical_inf_prob'][2] = agency_uptake_range[k] * \
                                                    (1-vacc_efficacy) * \
                                            SPEC_VACC_P['critical_inf_prob'][2]
            SPEC_VACC_P['mild_trans_scaling'][2] = agency_uptake_range[k] * \
                                                    vacc_inf_reduction * \
                                            SPEC_VACC_P['mild_trans_scaling'][2]

            model_input_unvacc = SEMCRDInput(SPEC_UNVACC, composition_list, comp_dist)
            model_input_vacc_P = SEMCRDInput(SPEC_VACC_P, composition_list, comp_dist)

            hh_pop_unvacc = HouseholdPopulation(composition_list,
                                            comp_dist,
                                            model_input_unvacc)
            hh_pop_P = HouseholdPopulation(composition_list,
                                            comp_dist,
                                            model_input_vacc_P)

            combined_pop = combine_household_populations([hh_pop_unvacc,
                                                        hh_pop_P],
                                                [1 - patient_uptake_range[i],
                                                    patient_uptake_range[i]])

            rhs = SEMCRDRateEquations(
                model_input,
                combined_pop,
                FixedImportModel(6,2,import_array))

            tspan = (0.0, 365.0)
            solver_start = get_time()
            solution = solve_ivp(rhs, tspan, H0, first_step=0.001,atol=1e-12)
            solver_end = get_time()

            print(
                'Solution', [i,j,k],'took',
                solver_end - solver_start,
                'seconds.')

            time = solution.t
            H = solution.y
            C = H.T.dot(combined_pop.states[:, 3::6])
            D = H.T.dot(combined_pop.states[:, 5::6])
            max_critical[i,j,k] = C.max()
            max_empty[i,j,k] = D.max()
# SPEC_VACC_PS = copy(SPEC)
# SPEC_VACC_PS['critical_inf_prob'][0] = (1-vacc_efficacy) * \
#                                     SPEC_VACC_PS['critical_inf_prob'][0]
# SPEC_VACC_PS['critical_inf_prob'][1] = (1-vacc_efficacy) * \
#                                     SPEC_VACC_PS['critical_inf_prob'][1]
# SPEC_VACC_PS['mild_trans_scaling'][0] = vacc_inf_reduction * \
#                                     SPEC_VACC_PS['mild_trans_scaling'][0]
# SPEC_VACC_PS['mild_trans_scaling'][1] = vacc_inf_reduction * \
#                                     SPEC_VACC_PS['mild_trans_scaling'][1]
# SPEC_VACC_PSA = copy(SPEC)
# SPEC_VACC_PSA['critical_inf_prob'][0] = (1-vacc_efficacy) * \
#                                     SPEC_VACC_PSA['critical_inf_prob'][0]
# SPEC_VACC_PSA['critical_inf_prob'][1] = (1-vacc_efficacy) * \
#                                     SPEC_VACC_PSA['critical_inf_prob'][1]
# SPEC_VACC_PSA['critical_inf_prob'][2] = (1-vacc_efficacy) * \
#                                     SPEC_VACC_PSA['critical_inf_prob'][2]
# SPEC_VACC_PSA['mild_trans_scaling'][0] = vacc_inf_reduction * \
#                                     SPEC_VACC_PSA['mild_trans_scaling'][0]
# SPEC_VACC_PSA['mild_trans_scaling'][1] = vacc_inf_reduction * \
#                                     SPEC_VACC_PSA['mild_trans_scaling'][1]
# SPEC_VACC_PSA['mild_trans_scaling'][2] = vacc_inf_reduction * \
#                                     SPEC_VACC_PSA['mild_trans_scaling'][2]

# model_input_vacc_P = SEMCRDInput(SPEC_VACC_P, composition_list, comp_dist)
# model_input_vacc_PS = SEMCRDInput(SPEC_VACC_PS, composition_list, comp_dist)
# model_input_vacc_PSA = SEMCRDInput(SPEC_VACC_PSA, composition_list, comp_dist)

# hh_pop_P = HouseholdPopulation(composition_list,
#                                 comp_dist,
#                                 model_input_vacc_P)
# hh_pop_PS = HouseholdPopulation(composition_list,
#                                 comp_dist,
#                                 model_input_vacc_PS)
# hh_pop_PSA = HouseholdPopulation(composition_list,
#                                 comp_dist,
#                                 model_input_vacc_PSA)

# uptake_by_class = array([0.9, 0.6, 0.25])                  # Percentage of each class vaccinated
# weightings = hstack((array([1]), uptake_by_class)) - \
#                 hstack((uptake_by_class, array([0])))  # weighting for each home-level vaccine status is calculated to match uptake

# combined_pop = combine_baseline_populations([baseline_population,
#                                                 hh_pop_P,
#                                                 hh_pop_PS,
#                                                 hh_pop_PSA],
#                                                 weightings)





# rhs = SEMCRDRateEquations(
#     model_input,
#     combined_pop,
#     FixedImportModel(6,2,import_array))
#
# tspan = (0.0, 365.0)
# solver_start = get_time()
# solution = solve_ivp(rhs, tspan, H0, first_step=0.001,atol=1e-12)
# solver_end = get_time()
#
# print(
#     'Solution took',
#     solver_end - solver_start,
#     'seconds.')
#
# time = solution.t
# H = solution.y
# S = H.T.dot(combined_pop.states[:, ::6])
# E = H.T.dot(combined_pop.states[:, 1::6])
# M = H.T.dot(combined_pop.states[:, 2::6])
# C = H.T.dot(combined_pop.states[:, 3::6])
# R = H.T.dot(combined_pop.states[:, 4::6])
# D = H.T.dot(combined_pop.states[:, 5::6])
# time_series = {
# 'time':time,
# 'S':S,
# 'E':E,
# 'M':M,
# 'C':C,
# 'R':R,
# 'D':D
# }


with open('carehome_sweep_data.pkl', 'wb') as f:
    dump((max_critical, max_empty, model_input), f)
