''' This runs a simple example with constant importation where infectious
individuals get isolated outside of the household'''

from os.path import isfile
from pickle import load, dump
from numpy import arange, array
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.integrate import solve_ivp
from model.preprocessing import TwoAgeWithVulnerableInput, HouseholdPopulation
from model.preprocessing import add_vulnerable_hh_members, make_initial_SEPIRQ_condition
from model.specs import SEPIRQ_SPEC
from model.common import SEPIRQRateEquations, within_household_SEPIRQ
from model.imports import ( FixedImportModel)
import pdb
# pylint: disable=invalid-name

spec = SEPIRQ_SPEC

model_input = TwoAgeWithVulnerableInput(SEPIRQ_SPEC)
# model_input.alpha2 = 1/1
model_input.E_iso_rate = 1/2
model_input.P_iso_rate = 1/1
model_input.I_iso_rate = 2
model_input.discharge_rate = 1/7
model_input.adult_bd = 1
model_input.class_is_isolating = array([[True, True, True],[True,True,True],[True,True,True]])
model_input.iso_method = 1
model_input.iso_prob = 0.5

if isfile('iso-vars.pkl') is True:
    with open('iso-vars.pkl', 'rb') as f:
        household_population, composition_list, comp_dist = load(f)
else:
    # List of observed household compositions
    composition_list = read_csv(
        'inputs/eng_and_wales_adult_child_vuln_composition_list.csv',
        header=0).to_numpy()
    # Proportion of households which are in each composition
    comp_dist = read_csv(
        'inputs/eng_and_wales_adult_child_vuln_composition_dist.csv',
        header=0).to_numpy().squeeze()
    # With the parameters chosen, we calculate Q_int:
    # composition_list, comp_dist = add_vulnerable_hh_members(baseline_composition_list,baseline_comp_dist,model_input.vuln_prop)
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input, within_household_SEPIRQ,6)
    with open('iso-vars.pkl', 'wb') as f:
        dump((household_population, composition_list, comp_dist), f)

# Relative strength of between-household transmission compared to external
# imports
no_days = 100

import_model = FixedImportModel(
    1e-5,
    1e-5)

rhs = SEPIRQRateEquations(
    model_input,
    household_population,
    import_model
    )

H0 = make_initial_SEPIRQ_condition(household_population, rhs)

tspan = (0.0, 30)
solver_start = get_time()
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)
solver_end = get_time()

time = solution.t
H = solution.y
P = H.T.dot(household_population.states[:, 2::6])
I = H.T.dot(household_population.states[:, 3::6])
if model_input.iso_method==0:
    Q = H.T.dot(household_population.states[:, 5::6])
else:
    states_iso_only = household_population.states[:,5::6]
    total_iso_by_state =states_iso_only.sum(axis=1)
    iso_present = total_iso_by_state>0
    Q = H[iso_present,:].T.dot(household_population.composition_by_state[iso_present,:])

# pdb.set_trace()
children_per_hh = comp_dist.T.dot(composition_list[:,0])
nonv_adults_per_hh = comp_dist.T.dot(composition_list[:,1])
vuln_adults_per_hh = comp_dist.T.dot(composition_list[:,2])

with open('external-isolation-results.pkl', 'wb') as f:
    dump((time, H, P, I, Q, children_per_hh, nonv_adults_per_hh, vuln_adults_per_hh), f)


print('Integration completed in', solver_end-solver_start,'seconds.')
