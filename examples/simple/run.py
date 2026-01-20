'''This sets up and runs a simple system which is low-dimensional enough to
do locally'''
from numpy import array
from time import time as get_time
from scipy.integrate import solve_ivp
from model.preprocessing import (
    SEPIRInput, HouseholdPopulation, make_initial_condition)
from model.specs import TWO_AGE_SEPIR_SPEC, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations
from model.imports import NoImportModel
from pickle import dump
# pylint: disable=invalid-name

SPEC = {**TWO_AGE_SEPIR_SPEC, **TWO_AGE_UK_SPEC}

# List of observed household compositions
composition_list = array(
    [[0, 1], [0, 2], [1, 1], [1, 2], [2, 1], [2, 2]])
# Proportion of households which are in each composition
comp_dist = array([0.2, 0.2, 0.1, 0.1, 0.1,  0.1])

model_input = SEPIRInput(SPEC, composition_list, comp_dist)

household_population = HouseholdPopulation(
    composition_list, comp_dist,model_input)

rhs = SEPIRRateEquations(
    model_input,
    household_population,
    NoImportModel(5, 2))

H0 = make_initial_condition(
    household_population, rhs)

tspan = (0.0, 365.0)
simple_model_start = get_time()
solution = solve_ivp(rhs, tspan, H0, first_step=0.001)
simple_model_end = get_time()

time = solution.t
H = solution.y
time_series = {
    'time': time,
    'S': H.T.dot(household_population.states[:, ::5]),
    'E': H.T.dot(household_population.states[:, 1::5]),
    'P': H.T.dot(household_population.states[:, 2::5]),
    'I': H.T.dot(household_population.states[:, 3::5]),
    'R': H.T.dot(household_population.states[:, 4::5])
}

print('Solution took {} seconds.'.format(
    simple_model_end - simple_model_start))

with open('simple.pkl', 'wb') as f:
    dump((H, time_series, model_input), f)
