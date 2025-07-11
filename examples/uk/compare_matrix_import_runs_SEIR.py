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
        SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import TWO_AGE_SEIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import SEIRRateEquations, MatrixImportSEIRRateEquations, UnloopedSEIRRateEquations
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


SPEC = {**TWO_AGE_SEIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
model_input_to_fit = SEIRInput(SPEC, composition_list, comp_dist)
household_population_to_fit = HouseholdPopulation(
    composition_list, comp_dist, model_input_to_fit)
rhs_to_fit = SEIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(4,2))
beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
model_input = deepcopy(model_input_to_fit)
model_input.k_ext *= beta_ext
print('Estimated beta is',beta_ext)

# With the parameters chosen, we calculate Q_int:
household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

no_imports = NoImportModel(4, 2)
base_rhs = SEIRRateEquations(model_input, household_population, no_imports)
fixed_imports = FixedImportModel(4,2, base_rhs, array([.1, .1]))
exp_imports = ExponentialImportModel(4,
                                     2,
                                     base_rhs,
                                     growth_rate,
                                     array([1e-5, 1e-5]))

rhs = SEIRRateEquations(model_input, household_population, exp_imports)
rhs_M = MatrixImportSEIRRateEquations(model_input, household_population, exp_imports)
rhs_U = UnloopedSEIRRateEquations(model_input, household_population, exp_imports, sources="ALL")

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
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))
print("Non-matrix took", time.time()- nm_start)
m_start = time.time()
solution_M = solve_ivp(rhs_M, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))
print("M took", time.time() - m_start)
u_start = time.time()
solution_U = solve_ivp(rhs_U, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))
print("U took", time.time() - u_start)

t = solution.t
H = solution.y
S = H.T.dot(household_population.states[:, ::4])
E = H.T.dot(household_population.states[:, 1::4])
I = H.T.dot(household_population.states[:, 2::4])
R = H.T.dot(household_population.states[:, 3::4])
time_series = {
'time':t,
'S':S,
'E':E,
'I':I,
'R':R
}

t_M = solution_M.t
H_M = solution_M.y
S_M = H_M.T.dot(household_population.states[:, ::4])
E_M = H_M.T.dot(household_population.states[:, 1::4])
I_M = H_M.T.dot(household_population.states[:, 2::4])
R_M = H_M.T.dot(household_population.states[:, 3::4])
time_series_M = {
'time':t_M,
'S':S_M,
'E':E_M,
'I':I_M,
'R':R_M
}

t_U = solution_U.t
H_U = solution_U.y
S_U = H_U.T.dot(household_population.states[:, ::4])
E_U = H_U.T.dot(household_population.states[:, 1::4])
I_U = H_U.T.dot(household_population.states[:, 2::4])
R_U = H_U.T.dot(household_population.states[:, 3::4])
time_series_U = {
'time':t_U,
'S':S_U,
'E':E_U,
'I':I_U,
'R':R_U
}

# Print disagreement with cutoff of 1e-9 so that we don't return high errors when
# both are very close to zero, i.e. if H_M~1e-n, H~1e-m for large n and m we don't
# want the relative error to me 1e-(n-m).
print("Max relative error in H_M for H>1e-9 is", max(abs((H-H_M))[H>1e-9]/H[H>1e-9]))
print("Max relative error in H_U for H>1e-9 is", max(abs((H-H_U))[H>1e-9]/H[H>1e-9]))

# Check growth rate estimate works in alternative models:
r_base = estimate_growth_rate(household_population, rhs, [0.001, 5], 1e-9)
r_U = estimate_growth_rate(household_population, rhs_U, [0.001, 5], 1e-9)
print("Growth rate from RateEquations =", r_base)
print("Growth rate from UnloopedRateEquations =", r_U)

# Now check whether directly setting up a model with a given parameter is the same as inputting that parameter as we
# would in an MCMC routine:

# Set up a model where within-household infection is half as strong as in base model:
model_input_half_inf = deepcopy(model_input)
model_input_half_inf.k_home *= 0.5

household_population_half_inf = HouseholdPopulation(
    composition_list, comp_dist, model_input_half_inf)

rhs_half_inf_direct = UnloopedSEIRRateEquations(model_input_half_inf,
                                                household_population_half_inf,
                                                fixed_imports,
                                                sources="BETWEEN")

# And set up by overwriting Q_int in the rhs object for the base model:
rhs_base = UnloopedSEIRRateEquations(model_input, household_population, fixed_imports, sources="BETWEEN")
rhs_half_inf_overwrite = deepcopy(rhs_base)
rhs_half_inf_overwrite.Q_int_inf *= 0.5
rhs_half_inf_overwrite.Q_int = rhs_half_inf_overwrite.Q_int_inf +\
    rhs_half_inf_overwrite.Q_int_fixed

# Check if there is any difference between the two arrays:
Q_int_diffs = {"direct - overwrite": abs(rhs_half_inf_direct.Q_int - rhs_half_inf_overwrite.Q_int).max(None)}

# And double check to make sure the subtraction operation isn't trivially zero for some reason:
Q_int_diffs["direct - base"] = abs(rhs_half_inf_direct.Q_int - rhs_U.Q_int).max(None)
Q_int_diffs["overwrite - base"] = abs(rhs_half_inf_overwrite.Q_int - rhs_U.Q_int).max(None)

# Alternative check: does halving and then multiplying by 2 give the same result as the original model?
rhs_reset = deepcopy(rhs_half_inf_direct)
rhs_reset.Q_int_inf *= 2.
rhs_reset.Q_int = rhs_reset.Q_int_inf +\
    rhs_reset.Q_int_fixed

Q_int_diffs["reset-base"] = abs(rhs_reset.Q_int - rhs_U.Q_int).max(None)

# Now check updating system works by running model with directly scaled matrices and updated int_rate attribute.
rhs_int_rate_update = deepcopy(rhs_base)
rhs_int_rate_update.int_rate *= 0.5

# Do some timing just in case there's a performance difference
start_time = time.time()
sol_direct = solve_ivp(rhs_half_inf_direct, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))
print("rhs_half_inf_direct execution time:", time.time()- start_time)
start_time = time.time()
sol_overwrite = solve_ivp(rhs_half_inf_overwrite, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))
print("rhs_half_inf_overwrite execution time:", time.time()- start_time)
start_time = time.time()
sol_update = solve_ivp(rhs_int_rate_update, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))
print("rhs_int_rate_update execution time:", time.time() - start_time)

# Should find that direct and overwrite give identical results, while update is incorrect:
sol_diffs = {"direct - update" : max(abs((sol_direct.y-sol_update.y))[sol_update.y>1e-9]),
             "direct - overwrite" : max(abs((sol_direct.y-sol_overwrite.y))[sol_overwrite.y>1e-9]),
             "overwrite - update" : max(abs((sol_overwrite.y-sol_update.y))[sol_overwrite.y>1e-9])}

# Now try another approach: update method
rhs_method = deepcopy(rhs_base)
rhs_method.update_int_rate(0.5)

# Should find we get the correct Q_int again
Q_int_diffs["direct-method"] = abs(rhs_half_inf_direct.Q_int - rhs_method.Q_int).max(None)
Q_int_diffs["method-base"] = abs(rhs_method.Q_int - rhs_base.Q_int).max(None)

# Compare solver performance and results
start_time = time.time()
sol_method = solve_ivp(rhs_method, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))
print("rhs_int_rate_update execution time:", time.time() - start_time)
sol_diffs["direct - method"] = max(abs((sol_direct.y-sol_method.y))[sol_method.y>1e-9])
sol_diffs["overwrite - method"] = max(abs((sol_overwrite.y-sol_method.y))[sol_method.y>1e-9])

r_direct = estimate_growth_rate(household_population, rhs_half_inf_direct, [0.001, 5], 1e-9)
r_method = estimate_growth_rate(household_population, rhs_method, [0.001, 5], 1e-9)
print("Growth rate from direct rescaling =", r_direct)
print("Growth rate from method rescaling =", r_method)

rhs_no_int_inf = deepcopy(rhs_base)
rhs_no_int_inf.update_int_rate(0.)
print("Growth rate from model without internal infection =",
      estimate_growth_rate(household_population, rhs_no_int_inf, [0.001, 5], 1e-9))

# How would this look inside an MCMC step?
#
# # Before running MCMC:
# # Set up model input with tau=lambda=1 by rescaling transmission matrices
# model_input_MCMC = deepcopy(model_input)
# model_input_MCMC.k_home *= 1 / model_input_MCMC.beta_int
# model_input_MCMC.k_ext *= 1 / beta_ext
#
# rhs_MCMC = UnloopedSEIRRateEquations(model_input_MCMC,
#                                     household_population,
#                                     fixed_imports,
#                                     sources="ALL")
#
# # On each MCMC step
# def update_function(tau, lmbd, data):
#     rhs_MCMC.int_rate = tau
#     rhs_MCMC.ext_rate = lmbd
#     return evaluate_likelihood(rhs_MCMC, data)