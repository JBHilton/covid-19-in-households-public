'''This runs the UK-like model with a single set of parameters for 100 days
'''
from copy import deepcopy
from numpy import arange, array, log, where
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEPIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations, MatrixImportSEPIRRateEquations, UnloopedSEPIRRateEquations
from model.imports import FixedImportModel, NoImportModel, ExponentialImportModel

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


SPEC = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
model_input_to_fit = SEPIRInput(SPEC, composition_list, comp_dist)
household_population_to_fit = HouseholdPopulation(
    composition_list, comp_dist, model_input_to_fit)
rhs_to_fit = SEPIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(5,2))
beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
model_input = deepcopy(model_input_to_fit)
model_input.k_ext *= beta_ext
print('Estimated beta is',beta_ext)

# With the parameters chosen, we calculate Q_int:
household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

no_imports = NoImportModel(5, 2)
fixed_imports = FixedImportModel(5,2, array([.1, .1]))
base_rhs = SEPIRRateEquations(model_input, household_population, no_imports)
exp_imports = ExponentialImportModel(5,
                                     2,
                                     base_rhs,
                                     growth_rate,
                                     array([1e-5, 1e-5]))

rhs = SEPIRRateEquations(model_input, household_population, exp_imports)
rhs.epsilon = 0
rhs_M = MatrixImportSEPIRRateEquations(model_input, household_population, exp_imports)
rhs_U = UnloopedSEPIRRateEquations(model_input, household_population, exp_imports, sources="IMPORT")

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs_M, 0., 0.0,False,3)
S0 = H0.T.dot(household_population.states[:, ::5])
E0 = H0.T.dot(household_population.states[:, 1::5])
P0 = H0.T.dot(household_population.states[:, 2::5])
I0 = H0.T.dot(household_population.states[:, 3::5])
R0 = H0.T.dot(household_population.states[:, 4::5])
start_state = (1/model_input.ave_hh_size) * array([S0.sum(),
                                                   E0.sum(),
                                                   P0.sum(),
                                                   I0.sum(),
                                                   R0.sum()])
tspan = (0.0, 365)
import time
nm_start = time.time()
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))
print("Non-matrix took", time.time()- nm_start)
# m_start = time.time()
# solution_M = solve_ivp(rhs_M, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))
# print("M took", time.time() - m_start)
u_start = time.time()
solution_U = solve_ivp(rhs_U, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))
print("U took", time.time() - u_start)

H = solution.y
R = H.T.dot(household_population.states[:, 4::5])

# H_M = solution_M.y

H_U = solution_U.y
R_U = H_U.T.dot(household_population.states[:, 4::5])

# Print disagreement with cutoff of 1e-9 so that we don't return high errors when
# both are very close to zero, i.e. if H_M~1e-n, H~1e-m for large n and m we don't
# want the relative error to be 1e-(n-m).
# print("Max relative error in H_M for H>1e-9 is", max(abs((H-H_M))[H>1e-9]/H[H>1e-9]))
print("Max relative error in H_U for H>1e-9 is", max(abs((H-H_U))[H>1e-9]/H[H>1e-9]))
