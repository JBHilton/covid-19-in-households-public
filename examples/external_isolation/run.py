''' This runs a simple example with constant importation where infectious
individuals get isolated outside of the household'''

from os.path import isfile
from pickle import load, dump
from numpy import arange, array
from numpy.random import rand
from time import time as get_time
from scipy.integrate import solve_ivp
from model.preprocessing import TwoAgeModelInput, HouseholdPopulation
from model.preprocessing import make_initial_condition
from model.specs import DEFAULT_SPEC
from model.common import RateEquations, within_household_spread_with_isolation
from model.imports import ( FixedImportModel)
# pylint: disable=invalid-name

model_input = TwoAgeModelInput(DEFAULT_SPEC)
model_input.D_iso_rate = 1/1
model_input.U_iso_rate = 1/2
model_input.discharge_rate = 1/7
model_input.adult_bd = 1
model_input.class_is_isolating = [True, True]

if isfile('iso-vars.pkl') is True:
    with open('iso-vars.pkl', 'rb') as f:
        household_population = load(f)
else:
    # List of observed household compositions
    composition_list = array([
        [0, 1], [0, 2], [1, 1], [1, 2], [2, 1], [2, 2]])
    # Proportion of households which are in each composition
    comp_dist = array([0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
    # With the parameters chosen, we calculate Q_int:
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input, within_household_spread_with_isolation,6)
    with open('iso-vars.pkl', 'wb') as f:
        dump(household_population, f)

# Relative strength of between-household transmission compared to external
# imports
epsilon = 0.5
no_days = 100

import_model = FixedImportModel(
    1e-1,
    1e-1)

rhs = RateEquations(
    model_input,
    household_population,
    import_model,
    epsilon,
    6)

H0 = make_initial_condition(household_population, rhs)

tspan = (0.0, 100)
solver_start = get_time()
solution = solve_ivp(rhs, tspan, H0, first_step=0.001)
solver_end = get_time()

time = solution.t
H = solution.y
D = H.T.dot(household_population.states[:, 2::6])
U = H.T.dot(household_population.states[:, 3::6])
Q = H.T.dot(household_population.states[:, 5::6])

with open('external-isolation-results.pkl', 'wb') as f:
    dump((time, H, D, U, Q, model_input.coarse_bds), f)


print('Integration completed in', solver_end-solver_start,'seconds.')
