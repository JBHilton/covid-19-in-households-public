''' This runs a simple example with constant importation where infectious
individuals get isolated outside of the household'''

from copy import deepcopy
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from numpy import arange, array, log
from numpy.linalg import eig
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.integrate import solve_ivp
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from model.preprocessing import SEPIRInput, SEPIRQInput, HouseholdPopulation
from model.preprocessing import (add_vuln_class, add_vulnerable_hh_members,
estimate_beta_ext, make_initial_condition_by_eigenvector, map_SEPIR_to_SEPIRQ)
from model.common import SEPIRRateEquations, SEPIRQRateEquations
from model.imports import NoImportModel
from model.specs import (TWO_AGE_UK_SPEC, TWO_AGE_EXT_SEPIRQ_SPEC,
TWO_AGE_INT_SEPIRQ_SPEC, TWO_AGE_SEPIR_SPEC_FOR_FITTING)

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

sepir_sepc = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}


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

ext_model_input = SEPIRQInput(ext_spec, composition_list, comp_dist)
ext_model_input.k_ext = beta_ext * ext_model_input.k_ext
ext_model_input = add_vuln_class(ext_model_input,
                    vuln_prop,
                    adult_class)
int_model_input = SEPIRQInput(int_spec, composition_list, comp_dist)
int_model_input.k_ext = beta_ext * int_model_input.k_ext
int_model_input = add_vuln_class(int_model_input,
                    vuln_prop,
                    adult_class)

prev = 1e-5
antiprev = 1e-2

# adherence_rate = 1
#
# model_input.E_iso_rates = adherence_rate*1/1
# model_input.P_iso_rates = adherence_rate*1/1
# model_input.I_iso_rates = adherence_rate*1/0.5
# model_input.discharge_rate = 1/14
# model_input.adult_bd = 1
# model_input.class_is_isolating = array([[False, False, False],[False, False, True],[False, False, False]])
# model_input.iso_method = 0
# model_input.iso_prob = 1


# With the parameters chosen, we calculate Q_int:
OOHI_household_population = HouseholdPopulation(
    composition_list, comp_dist, ext_model_input)

import_model = NoImportModel(6,3)

OOHI_rhs = SEPIRQRateEquations(
    ext_model_input,
    OOHI_household_population,
    import_model)

pre_npi_household_population = HouseholdPopulation(
    composition_list, comp_dist, pre_npi_input)
pre_npi_rhs = SEPIRRateEquations(
    pre_npi_input,
    pre_npi_household_population,
    import_model)

# H0 = make_initial_condition_with_recovereds(OOHI_household_population, OOHI_rhs, prev)

if isfile('outputs/oohi/H0_pre_npi.pkl') is True:
    with open('outputs/oohi/H0_pre_npi.pkl', 'rb') as f:
        H0_pre_npi = load(f)
else:
    print('number of states for 3-class pop is',pre_npi_household_population.Q_int.shape)
    H0_pre_npi = make_initial_condition_by_eigenvector(growth_rate,
                                                       pre_npi_input,
                                                       pre_npi_household_population,
                                                       pre_npi_rhs,
                                                       prev,
                                                       antiprev)
    with open('outputs/oohi/H0_pre_npi.pkl', 'wb') as f:
        dump(H0_pre_npi, f)


if isfile('outputs/oohi/map_matrix.pkl') is True:
    with open('outputs/oohi/map_matrix.pkl', 'rb') as f:
        map_matrix = load(f)
else:
    print('getting map matrix')

    map_matrix = map_SEPIR_to_SEPIRQ(pre_npi_household_population,
                                 OOHI_household_population)
    print('got map matrix')
    with open('outputs/oohi/map_matrix.pkl', 'wb') as f:
        dump(map_matrix, f)
S0 = H0_pre_npi.T.dot(pre_npi_household_population.states[:, ::5])
E0 = H0_pre_npi.T.dot(pre_npi_household_population.states[:, 1::5])
P0 = H0_pre_npi.T.dot(pre_npi_household_population.states[:, 2::5])
I0 = H0_pre_npi.T.dot(pre_npi_household_population.states[:, 3::5])
R0 = H0_pre_npi.T.dot(pre_npi_household_population.states[:, 4::5])
start_state = (1/pre_npi_input.ave_hh_size) * array([S0.sum(),
                                                   E0.sum(),
                                                   P0.sum(),
                                                   I0.sum(),
                                                   R0.sum()])
print('Pre NPI start state is', start_state,'.')
print('Initial case profile by class is',E0)

H0 = H0_pre_npi * map_matrix

S0 = H0.T.dot(OOHI_household_population.states[:, ::6])
E0 = H0.T.dot(OOHI_household_population.states[:, 1::6])
P0 = H0.T.dot(OOHI_household_population.states[:, 2::6])
I0 = H0.T.dot(OOHI_household_population.states[:, 3::6])
R0 = H0.T.dot(OOHI_household_population.states[:, 4::6])
Q0 = H0.T.dot(OOHI_household_population.states[:, 5::6])
start_state = (1/ext_model_input.ave_hh_size) * array([S0.sum(),
                                                   E0.sum(),
                                                   P0.sum(),
                                                   I0.sum(),
                                                   R0.sum(),
                                                   Q0.sum()])
print('OOHI start state is', start_state,'.')
print('Initial case profile by class is',E0)

no_days = 100
tspan = (0.0, no_days)
solver_start = get_time()
solution = solve_ivp(OOHI_rhs, tspan, H0, first_step=0.001, atol=1e-16)
solver_end = get_time()

print('Integration completed in', solver_end-solver_start,'seconds.')

OOHI_time = solution.t
OOHI_H = solution.y

S_OOHI = OOHI_H.T.dot(OOHI_household_population.states[:, ::6])
E_OOHI = OOHI_H.T.dot(OOHI_household_population.states[:, 1::6])
P_OOHI = OOHI_H.T.dot(OOHI_household_population.states[:, 2::6])
I_OOHI = OOHI_H.T.dot(OOHI_household_population.states[:, 3::6])
R_OOHI = OOHI_H.T.dot(OOHI_household_population.states[:, 4::6])
Q_OOHI = OOHI_H.T.dot(OOHI_household_population.states[:, 5::6])

children_per_hh = comp_dist.T.dot(composition_list[:,0])
nonv_adults_per_hh = comp_dist.T.dot(composition_list[:,1])
vuln_adults_per_hh = comp_dist.T.dot(composition_list[:,2])

# model_input.class_is_isolating = array([[True, True, True],[True, True, True],[True, True, True]])
# model_input.iso_method = 1

WHQ_household_population = HouseholdPopulation(
    composition_list, comp_dist, int_model_input)

WHQ_rhs = SEPIRQRateEquations(
    int_model_input,
    WHQ_household_population,
    import_model)

tspan = (0.0, no_days)
solver_start = get_time()
solution = solve_ivp(WHQ_rhs, tspan, H0, first_step=0.001,atol=1e-16)
solver_end = get_time()

print('Integration completed in', solver_end-solver_start,'seconds.')

WHQ_time = solution.t
WHQ_H = solution.y

S_WHQ = WHQ_H.T.dot(WHQ_household_population.states[:, ::6])
E_WHQ = WHQ_H.T.dot(WHQ_household_population.states[:, 1::6])
P_WHQ = WHQ_H.T.dot(WHQ_household_population.states[:, 2::6])
I_WHQ = WHQ_H.T.dot(WHQ_household_population.states[:, 3::6])
R_WHQ = WHQ_H.T.dot(WHQ_household_population.states[:, 4::6])
states_iso_only = WHQ_household_population.states[:,5::6]
total_iso_by_state =states_iso_only.sum(axis=1)
iso_present = total_iso_by_state>0
Q_WHQ = WHQ_H[iso_present,:].T.dot(WHQ_household_population.composition_by_state[iso_present,:])

# model_input.class_is_isolating = array([[False, False, False],[False, False, False],[False, False, False]])
# model_input.iso_method = 0
#
# baseline_household_population = HouseholdPopulation(
#     composition_list, comp_dist, model_input, within_household_SEPIRQ,6)
#
# baseline_rhs = SEPIRQRateEquations(
#     model_input,
#     baseline_household_population,
#     import_model)
#
# H0 = make_initial_SEPIRQ_condition(baseline_household_population, baseline_rhs, prev)
#
# tspan = (0.0, no_days)
# solver_start = get_time()
# solution = solve_ivp(baseline_rhs, tspan, H0, first_step=0.001,atol=1e-16)
# solver_end = get_time()
#
# print('Integration completed in', solver_end-solver_start,'seconds.')
#
# baseline_time = solution.t
# baseline_H = solution.y
#
# S_baseline = baseline_H.T.dot(baseline_household_population.states[:, ::6])
# E_baseline = baseline_H.T.dot(baseline_household_population.states[:, 1::6])
# P_baseline = baseline_H.T.dot(baseline_household_population.states[:, 2::6])
# I_baseline = baseline_H.T.dot(baseline_household_population.states[:, 3::6])
# R_baseline = baseline_H.T.dot(baseline_household_population.states[:, 4::6])
# Q_baseline = baseline_H.T.dot(baseline_household_population.states[:, 5::6])

with open('outputs/oohi/results.pkl','wb') as f:
    dump((ext_model_input, I_OOHI,R_OOHI,Q_OOHI,OOHI_time,I_WHQ,R_WHQ,Q_WHQ,WHQ_time),f)
    # dump((I_baseline,R_baseline,Q_baseline,baseline_time,I_OOHI,R_OOHI,Q_OOHI,OOHI_time,I_WHQ,R_WHQ,Q_WHQ,WHQ_time),f)
