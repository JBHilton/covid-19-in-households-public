'''
Test script to make sure estimate_hh_reproductive_ratio does what we expect:
'''
from copy import deepcopy
from numpy import arange, array, log, argmin, where
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import (estimate_beta_ext, estimate_growth_rate, estimate_hh_reproductive_ratio,
                                 SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import TWO_AGE_SEIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import UnloopedSEIRRateEquations as SEIRRateEquations
from model.imports import NoImportModel

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

rhs = SEIRRateEquations(model_input, household_population, NoImportModel(4,2), sources = "IMPORT")

r_est = estimate_growth_rate(household_population, rhs, [0.001, 5], 1e-9)


print('Estimated growth rate is',r_est,'.')
print('Estimated doubling time is',log(2) / r_est,'.')
print('Estimated R_* is', estimate_hh_reproductive_ratio(household_population, rhs)[0], '.')

inf_states = where(rhs.states_rec_only.sum(1)>0)[0]

H0 = make_initial_condition_by_eigenvector(growth_rate,
                                           model_input,
                                           household_population,
                                           rhs,
                                           1e-3,
                                           0.0,
                                           False,
                                           3)
tspan = (0.0, 365)
solution = solve_ivp(rhs,
                     tspan,
                     H0,
                     first_step=0.001,
                     atol=1e-16)
H = solution.y
R_nz = H[inf_states, -1].T.dot(household_population.states[inf_states, 3::4]) / sum(H[inf_states, -1])
exp_cases = sum(rhs.household_population.model_input.k_ext.dot(R_nz)) / model_input.gamma
print("Estimated R* from direct simulation =", exp_cases)



# Calculate R* and r for a range of beta_ext values

gr_interval = [-SPEC['recovery_rate'], 1] # Interval used in growth rate estimation
gr_tol = 1e-3 # Absolute tolerance for growth rate estimation

external_mix_range = arange(0.1, .25, 0.01)
def r_from_scale(ext_scale):
    rhs_scaled = deepcopy(rhs)
    rhs_scaled.update_ext_rate(ext_scale)
    return estimate_growth_rate(household_population,
                                           rhs_scaled,
                                           gr_interval,
                                           gr_tol,
                                           x0=1e-3,
                                           r_min_discount=0.99)
def rstar_from_scale(ext_scale):
    rhs_scaled = deepcopy(rhs)
    rhs_scaled.update_ext_rate(ext_scale)
    return estimate_hh_reproductive_ratio(household_population,
                                           rhs_scaled)[0]

r_by_scale = [r_from_scale(es) for es in external_mix_range]
rstar_by_scale = [rstar_from_scale(es) for es in external_mix_range]

# Check locations:
print("r=0 at ext_scale=",
      external_mix_range[argmin(abs(array(r_by_scale)))])
print("R*=1 at ext_scale=",
      external_mix_range[argmin(abs(array(rstar_by_scale)-1))])

# Alternative approach: produce R* estimates through direct solution

tspan = (0.0, 365)
def rstar_from_scale_by_ode(idx):
    print("idx=",idx)
    ext_scale = external_mix_range[idx]
    rhs_scaled = deepcopy(rhs)

    H0 = make_initial_condition_by_eigenvector(r_by_scale[idx],
                                               model_input,
                                               household_population,
                                               rhs_scaled,
                                               1e-4,
                                               0.0,
                                               False,
                                               3)
    rhs_scaled.update_ext_rate(0)
    solution = solve_ivp(rhs_scaled,
                         tspan,
                         H0,
                         first_step=0.001,
                         atol=1e-16)
    H = solution.y
    R_nz = H[inf_states, -1].T.dot(household_population.states[inf_states, 3::4]) / sum(H[inf_states, -1])
    exp_cases = sum(ext_scale * rhs_scaled.household_population.model_input.k_ext.dot(R_nz)) / model_input.gamma

    return exp_cases

rstar_by_scale_ode = [rstar_from_scale_by_ode(i) for i in range(len(external_mix_range))]

rstar_err = abs(array(rstar_by_scale_ode) - array(rstar_by_scale).T) / array(rstar_by_scale_ode).T

print("Relative discrepancies in R* estimates:",
      rstar_err)