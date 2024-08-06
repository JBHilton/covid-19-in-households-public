'''This runs the UK-like model with a single set of parameters for 100 days
'''
from copy import deepcopy
from numpy import array, log, where
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import TWO_AGE_SEIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import SEIRRateEquations, MatrixImportSEIRRateEquations
from model.imports import NoImportModel
# pylint: disable=invalid-name

if isdir('outputs/uk') is False:
    mkdir('outputs/uk')

DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

# List of observed household compositions
composition_list = read_csv(
    'inputs/eng_and_wales_adult_child_composition_list.csv',
    header=0).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/eng_and_wales_adult_child_composition_dist.csv',
    header=0).to_numpy().squeeze()


SPEC = {**TWO_AGE_SEIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
model_input_to_fit = SEIRInput(SPEC, composition_list, comp_dist)
household_population_to_fit = HouseholdPopulation(
    composition_list, comp_dist, model_input_to_fit)
rhs_to_fit = SEIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(4,2))
beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
model_input = deepcopy(model_input_to_fit)
model_input.k_ext *= beta_ext
print('Estimated beta is',beta_ext)
with open('outputs/uk/fitted_SEIR_input.pkl', 'wb') as f:
    dump(model_input, f)


# With the parameters chosen, we calculate Q_int:
household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

rhs = SEIRRateEquations(model_input, household_population, NoImportModel(4,2))
rhs_M = MatrixImportSEIRRateEquations(model_input, household_population, NoImportModel(4,2))

r_est = estimate_growth_rate(household_population, rhs_M, [0.001, 5], 1e-9)


print('Estimated growth rate is',r_est,'.')
print('Estimated doubling time is',log(2) / r_est,'.')

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs_M, 1e-5, 0.0,False,3)
S0 = H0.T.dot(household_population.states[:, ::4])
E0 = H0.T.dot(household_population.states[:, 1::4])
I0 = H0.T.dot(household_population.states[:, 2::4])
R0 = H0.T.dot(household_population.states[:, 3::4])
start_state = (1/model_input.ave_hh_size) * array([S0.sum(),
                                                   E0.sum(),
                                                   I0.sum(),
                                                   R0.sum()])
tspan = (0.0, 365)
import time
nm_start = time.time()
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)
print("Non-matrix took", time.time()- nm_start)
m_start = time.time()
solution_M = solve_ivp(rhs_M, tspan, H0, first_step=0.001, atol=1e-16)
print("M took", time.time() - m_start)

time = solution.t
H = solution.y
S = H.T.dot(household_population.states[:, ::4])
E = H.T.dot(household_population.states[:, 1::4])
I = H.T.dot(household_population.states[:, 2::4])
R = H.T.dot(household_population.states[:, 3::4])
time_series = {
'time':time,
'S':S,
'E':E,
'I':I,
'R':R
}

time_M = solution_M.t
H_M = solution_M.y
S_M = H_M.T.dot(household_population.states[:, ::4])
E_M = H_M.T.dot(household_population.states[:, 1::4])
I_M = H_M.T.dot(household_population.states[:, 2::4])
R_M = H_M.T.dot(household_population.states[:, 3::4])
time_series_M = {
'time':time_M,
'S':S_M,
'E':E_M,
'I':I_M,
'R':R_M
}

with open('outputs/uk/SEIR_results.pkl', 'wb') as f:
    dump((H, time_series), f)
