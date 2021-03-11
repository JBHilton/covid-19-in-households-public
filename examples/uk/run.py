'''This runs the UK-like model with a single set of parameters for 100 days
'''
from os.path import isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import (
        SEPIRInput, HouseholdPopulation, make_initial_condition)
from model.specs import TWO_AGE_SEPIR_SPEC, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations
from model.imports import NoImportModel
# pylint: disable=invalid-name

SPEC = {**TWO_AGE_SEPIR_SPEC, **TWO_AGE_UK_SPEC}

# List of observed household compositions
composition_list = read_csv(
    'inputs/eng_and_wales_adult_child_composition_list.csv',
    header=0).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/eng_and_wales_adult_child_composition_dist.csv',
    header=0).to_numpy().squeeze()

model_input = SEPIRInput(SPEC, composition_list, comp_dist)

if isfile('vars.pkl') is True:
    with open('vars.pkl', 'rb') as f:
        household_population = load(f)
else:
    # With the parameters chosen, we calculate Q_int:
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input)
    with open('vars.pkl', 'wb') as f:
        dump(household_population, f)

rhs = SEPIRRateEquations(model_input, household_population, NoImportModel(5,2))

H0 = make_initial_condition(household_population, rhs)

tspan = (0.0, 365)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

time = solution.t
H = solution.y
S = H.T.dot(household_population.states[:, ::5])
E = H.T.dot(household_population.states[:, 1::5])
P = H.T.dot(household_population.states[:, 2::5])
I = H.T.dot(household_population.states[:, 3::5])
R = H.T.dot(household_population.states[:, 4::5])
time_series = {
'time':time,
'S':S,
'E':E,
'P':P,
'I':P,
'R':R
}

with open('uk_like.pkl', 'wb') as f:
    dump((H, time_series, model_input), f)
