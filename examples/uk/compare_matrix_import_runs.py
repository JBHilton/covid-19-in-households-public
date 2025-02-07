'''This runs the UK-like model with a single set of parameters for 100 days
'''
from copy import deepcopy
from numpy import arange, array, asarray, cumsum, exp, log, where
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

# Set up background population for models with nonzero imports
base_rhs = SEPIRRateEquations(model_input, household_population, no_imports)
base_H0 = make_initial_condition_by_eigenvector(growth_rate,
                                                model_input,
                                                household_population,
                                                base_rhs,
                                                1e-3,
                                                0.0,
                                                False,
                                                3)
x0 = base_H0.T.dot(household_population.states[:, 1::5])

fixed_imports = FixedImportModel(5,
                                     2,
                                     base_rhs,
                                     x0)
exp_imports = ExponentialImportModel(5,
                                     2,
                                     base_rhs,
                                     growth_rate,
                                     x0)

rhs = SEPIRRateEquations(model_input, household_population, fixed_imports)
rhs.epsilon = 0
rhs_M = MatrixImportSEPIRRateEquations(model_input, household_population, exp_imports)
rhs_U = UnloopedSEPIRRateEquations(model_input, household_population, fixed_imports, sources="IMPORT")

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, 0., 0.0,False,3)
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

T_end = 365.
tspan = (0.0, T_end)
import time
nm_start = time.time()
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., T_end, 1.))
print("Non-matrix took", time.time()- nm_start)
# m_start = time.time()
# solution_M = solve_ivp(rhs_M, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 10., 1.))
# print("M took", time.time() - m_start)
u_start = time.time()
solution_U = solve_ivp(rhs_U, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., T_end, 1.))
print("U took", time.time() - u_start)

H = solution.y
R = H.T.dot(household_population.states[:, 4::5])

# H_M = solution_M.y

H_U = solution_U.y
R_U = H_U.T.dot(household_population.states[:, 4::5])

# Get bottom 99.9% of state distribution at each time step for error calculation. This should be a reasonably
# systematic way to avoid getting big errors from taking ratios between very small probabilities.
desc_order = (-H).argsort(axis=0)
H_desc = H[desc_order]
H_cumsum = 0 * H
H_cumsum[desc_order] = cumsum(H_desc, axis=0) # This gives cumulative sum starting from largest to smallest element without reordering entries
locs_for_err = [where(H_cumsum[:, t]<.999) for t in range(H.shape[1])] # For each time step return a list of locations with cumulative sum below 99.9%

daily_max_err = asarray([
    max((abs((H[locs_for_err[t], t]-H_U[locs_for_err[t], t]))/H[locs_for_err[t], t]))[0] for t in range(H.shape[1])])
print("Max relative error in H_U along bottom 99.99% of distribution is", daily_max_err.max())

# For comparison with static cutoff, check what cutoff we are effectively using here:
leaveout_locs = [where(H_cumsum[:, t]>=.999) for t in range(H.shape[1])]
effective_cutoff = asarray([
    max(H[leaveout_locs[t], t])[0] for t in range(H.shape[1])])
print("Max effective cutoff:", effective_cutoff.max())
print("Mean effective cutoff:", effective_cutoff.mean())

# Print disagreement with cutoff of 1e-9 so that we don't return high errors when
# both are very close to zero, i.e. if H_M~1e-n, H~1e-m for large n and m we don't
# want the relative error to be 1e-(n-m).
# print("Max relative error in H_M for H>1e-9 is", max(abs((H-H_M))[H>1e-9]/H[H>1e-9]))
print("Max relative error in H_U for H>1e-9 is", max(abs((H-H_U))[H>1e-9]/H[H>1e-9]))
