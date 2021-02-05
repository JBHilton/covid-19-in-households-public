'''This sets up and runs a simple system which is low-dimensional enough to
do locally'''
from numpy import array
from time import time as get_time
from scipy.integrate import solve_ivp
from model.preprocessing import (
    TwoAgeModelInput, HouseholdPopulation, make_initial_condition)
from model.specs import DEFAULT_SPEC
from model.common import SEDURRateEquations
from model.imports import NoImportModel
from pickle import load, dump
# pylint: disable=invalid-name

# List of observed household compositions
composition_list = array(
    [[0, 1], [0, 2], [1, 1], [1, 2], [2, 1], [2, 2]])
# Proportion of households which are in each composition
comp_dist = array([0.2, 0.2, 0.1, 0.1, 0.1,  0.1])

model_input = TwoAgeModelInput(DEFAULT_SPEC, composition_list, comp_dist)

household_population = HouseholdPopulation(
    composition_list, comp_dist, 'SEDUR', model_input)

# class SEDURRateEquations(RateEquations):
#     @property
#     def states_det_only(self):
#         return household_population.states[:, 2::self.no_compartments]
#     @property
#     def states_undet_only(self):
#         return household_population.states[:, 3::self.no_compartments]
#     @property
#     def states_rec_only(self):
#         return household_population.states[:, 4::self.no_compartments]

rhs = SEDURRateEquations(
    'SEDUR',
    model_input,
    household_population,
    NoImportModel(5,2))

H0 = make_initial_condition(
    household_population, rhs)

tspan = (0.0, 1000)
simple_model_start = get_time()
solution = solve_ivp(rhs, tspan, H0, first_step=0.001)
simple_model_end = get_time()

time = solution.t
H = solution.y
D = H.T.dot(household_population.states[:, 2::5])
U = H.T.dot(household_population.states[:, 3::5])

print(
    'Simple model took ',
    simple_model_end-simple_model_start,
    ' seconds.')

with open('simple.pkl', 'wb') as f:
    dump((time, H, D, U, model_input.coarse_bds), f)
