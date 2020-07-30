'''This runs the UK-like model with a single set of parameters for 100 days
'''
from os.path import isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import (
        ModelInput, HouseholdPopulation, make_initial_condition)
from model.specs import DEFAULT_SPEC
from model.common import RateEquations
# pylint: disable=invalid-name

model_input = ModelInput(DEFAULT_SPEC)

if isfile('vars.pkl') is True:
    with open('vars.pkl', 'rb') as f:
        household_population = load(f)
else:
    # List of observed household compositions
    composition_list = read_csv(
        'inputs/uk_composition_list.csv',
        header=None).to_numpy()
    # Proportion of households which are in each composition
    comp_dist = read_csv(
        'inputs/uk_composition_dist.csv',
        header=None).to_numpy().squeeze()
    # With the parameters chosen, we calculate Q_int:
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input)
    with open('vars.pkl', 'wb') as f:
        dump(household_population, f)

rhs = RateEquations(model_input, household_population)

H0 = make_initial_condition(household_population, rhs)

tspan = (0.0, 100)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001)

time = solution.t
H = solution.y
D = H.T.dot(household_population.states[:, 2::5])
U = H.T.dot(household_population.states[:, 3::5])

with open('uk_like.pkl', 'wb') as f:
    dump((time, H, D, U, model_input.coarse_bds), f)
