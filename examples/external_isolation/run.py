''' This runs a simple example with constant importation where infectious
individuals get isolated outside of the household'''

from os.path import isfile
from pickle import load, dump
from numpy import arange, array
from numpy.linalg import eig
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.integrate import solve_ivp
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from model.preprocessing import SEPIRQInput, HouseholdPopulation
from model.preprocessing import (add_vuln_class, add_vulnerable_hh_members,
make_initial_SEPIRQ_condition)
from model.common import SEPIRQRateEquations, within_household_SEPIRQ
from model.imports import NoImportModel
from model.specs import (TWO_AGE_UK_SPEC, TWO_AGE_EXT_SEPIRQ_SPEC,
TWO_AGE_INT_SEPIRQ_SPEC)

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

vuln_prop = 2.2/56
vector_quants = ['sus',
                 'inf_scales0',
                 'inf_scales1',
                 'inf_scales2',
                 'iso_rateE',
                 'iso_rateP',
                 'iso_rateI']
adult_class = 1

ext_model_input = SEPIRQInput(ext_spec, composition_list, comp_dist)
ext_model_input.inf_scales0 = ext_model_input.inf_scales[0]
ext_model_input.inf_scales1 = ext_model_input.inf_scales[1]
ext_model_input.inf_scales2 = ext_model_input.inf_scales[2]
ext_model_input.iso_rateE = ext_model_input.iso_rates['E']
ext_model_input.iso_rateP = ext_model_input.iso_rates['P']
ext_model_input.iso_rateI = ext_model_input.iso_rates['I']
ext_model_input = add_vuln_class(ext_model_input,
                    vuln_prop,
                    vector_quants,
                    adult_class)
ext_model_input.inf_scales = [ext_model_input.inf_scales0,
                              ext_model_input.inf_scales1,
                              ext_model_input.inf_scales2]
ext_model_input.iso_rates = {'E': ext_model_input.iso_rateE,
                            'P': ext_model_input.iso_rateP,
                            'I': ext_model_input.iso_rateI,}
int_model_input = SEPIRQInput(int_spec, composition_list, comp_dist)
int_model_input.inf_scales0 = int_model_input.inf_scales[0]
int_model_input.inf_scales1 = int_model_input.inf_scales[1]
int_model_input.inf_scales2 = int_model_input.inf_scales[2]
int_model_input.iso_rateE = int_model_input.iso_rates['E']
int_model_input.iso_rateP = int_model_input.iso_rates['P']
int_model_input.iso_rateI = int_model_input.iso_rates['I']
int_model_input = add_vuln_class(int_model_input,
                    vuln_prop,
                    vector_quants,
                    adult_class)
int_model_input.inf_scales = [int_model_input.inf_scales0,
                              int_model_input.inf_scales1,
                              int_model_input.inf_scales2]
int_model_input.iso_rates = {'E': int_model_input.iso_rateE,
                            'P': int_model_input.iso_rateP,
                            'I': int_model_input.iso_rateI,}
prev = 1e-5

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

H0 = make_initial_SEPIRQ_condition(OOHI_household_population, OOHI_rhs, prev)

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

H0 = make_initial_SEPIRQ_condition(WHQ_household_population, WHQ_rhs, prev)

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

with open('isolation_data.pkl','wb') as f:
    dump((I_OOHI,R_OOHI,Q_OOHI,OOHI_time,I_WHQ,R_WHQ,Q_WHQ,WHQ_time),f)
    # dump((I_baseline,R_baseline,Q_baseline,baseline_time,I_OOHI,R_OOHI,Q_OOHI,OOHI_time,I_WHQ,R_WHQ,Q_WHQ,WHQ_time),f)
