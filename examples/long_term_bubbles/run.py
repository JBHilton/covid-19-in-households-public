''' In this script we do projections of the impact of a long-term support
bubble policy where '''

from os.path import isfile
from pickle import load, dump
from copy import deepcopy
from numpy import append, arange, array, exp, vstack, log, sum, where
from numpy.linalg import eig
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.integrate import solve_ivp
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from model.preprocessing import ( build_support_bubbles,
        SEPIRInput, HouseholdPopulation, make_initial_condition)
from model.specs import TWO_AGE_SEPIR_SPEC, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations
from model.imports import NoImportModel


SPEC = {**TWO_AGE_SEPIR_SPEC, **TWO_AGE_UK_SPEC}

# List of observed household compositions
composition_list = read_csv(
    'inputs/eng_and_wales_adult_child_composition_list.csv',
    header=0).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/eng_and_wales_adult_child_composition_dist.csv',
    header=0).to_numpy().squeeze()

bubble_prob = 0.5
max_adults = 1
max_bubble_size = 6

mixed_comp_list, mixed_comp_dist = build_support_bubbles(
        composition_list,
        comp_dist,
        max_adults,
        max_bubble_size,
        bubble_prob)

mixed_comp_dist = mixed_comp_dist/sum(mixed_comp_dist)

baseline_model_input = SEPIRInput(SPEC, composition_list, comp_dist)
bubbled_model_input = SEPIRInput(SPEC, mixed_comp_list, mixed_comp_dist)
bubbled_model_input.k_home = deepcopy(baseline_model_input.k_home)
bubbled_model_input.k_ext = deepcopy(baseline_model_input.k_ext)

if isfile('sb_vars.pkl') is True:
    with open('sb_vars.pkl', 'rb') as f:
        baseline_population, bubbled_population = load(f)
else:
    # With the parameters chosen, we calculate Q_int:
    baseline_population = HouseholdPopulation(
        composition_list, comp_dist, baseline_model_input)
    bubbled_population = HouseholdPopulation(
        mixed_comp_list, mixed_comp_dist, bubbled_model_input)
    with open('sb_vars.pkl', 'wb') as f:
        dump((baseline_population, bubbled_population), f)

bubbled_rhs = SEPIRRateEquations(bubbled_model_input, bubbled_population, NoImportModel(5,2))
baseline_rhs = SEPIRRateEquations(baseline_model_input, baseline_population, NoImportModel(5,2))

H0 = make_initial_condition(bubbled_population, bubbled_rhs)

tspan = (0.0, 90)
solution = solve_ivp(bubbled_rhs, tspan, H0, atol=1e-16)

time = solution.t
H = solution.y
S = H.T.dot(bubbled_population.states[:, ::5])
E = H.T.dot(bubbled_population.states[:, 1::5])
P = H.T.dot(bubbled_population.states[:, 2::5])
I = H.T.dot(bubbled_population.states[:, 3::5])
R = H.T.dot(bubbled_population.states[:, 4::5])
bubbled_time_series = {
'time':time,
'S':S,
'E':E,
'P':P,
'I':P,
'R':R
}

H0 = make_initial_condition(baseline_population, baseline_rhs)

tspan = (0.0, 90)
solution = solve_ivp(baseline_rhs, tspan, H0, atol=1e-16)

time = solution.t
H = solution.y
S = H.T.dot(baseline_population.states[:, ::5])
E = H.T.dot(baseline_population.states[:, 1::5])
P = H.T.dot(baseline_population.states[:, 2::5])
I = H.T.dot(baseline_population.states[:, 3::5])
R = H.T.dot(baseline_population.states[:, 4::5])
baseline_time_series = {
'time':time,
'S':S,
'E':E,
'P':P,
'I':P,
'R':R
}

with open('support_bubble_output.pkl', 'wb') as f:
    dump((baseline_time_series, bubbled_time_series, bubbled_model_input, baseline_model_input), f)
