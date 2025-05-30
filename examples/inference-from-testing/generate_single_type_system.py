'''This sets up a single-household0type model
'''
from numpy import array, log, where, zeros
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import SINGLE_AGE_SEIR_SPEC_FOR_FITTING, SINGLE_AGE_UK_SPEC
from model.common import SEIRRateEquations, UnloopedSEIRRateEquations
from model.imports import ExponentialImportModel, FixedImportModel, NoImportModel
# pylint: disable=invalid-name

# Calculate (negative) growth rate based on estimated prevalences from Vo study:
growth_rate = (1/11) * log((29/2343) / (73/2812))

hh_size = 2
# List of observed household compositions
composition_list = array([[hh_size]])
# Proportion of households which are in each composition
comp_dist = array([1.])



SPEC = {**SINGLE_AGE_SEIR_SPEC_FOR_FITTING, **SINGLE_AGE_UK_SPEC}
model_input_to_fit = SEIRInput(SPEC, composition_list, comp_dist)
household_population_to_fit = HouseholdPopulation(
    composition_list, comp_dist, model_input_to_fit)
rhs_to_fit = SEIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(4,1))
beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
model_input = model_input_to_fit
model_input.k_ext *= beta_ext
print('Estimated beta is',beta_ext)


# With the parameters chosen, we calculate Q_int:
household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

no_imports = NoImportModel(4, 1)
base_rhs = SEIRRateEquations(model_input, household_population, no_imports)
fixed_imports = FixedImportModel(4,1, base_rhs, array([.1]))
exp_imports = ExponentialImportModel(4,
                                     1,
                                     base_rhs,
                                     growth_rate,
                                     array([1e-5]))
rhs = UnloopedSEIRRateEquations(model_input, household_population, fixed_imports, sources="ALL")

Q_int = rhs.Q_int
Q_ext = exp_imports.base_matrix

# Generate a set of initial conditions with S = N - i, E = 0, I = i, R = 0
init_cases = 1 # Change this to get a different number of initial cases
S0 = hh_size - init_cases
E0 = 0
I0 = init_cases
R0 = 0
x0 = zeros(Q_int.shape[0])
init_state = where((rhs.states_sus_only==S0)&
                    (rhs.states_exp_only==E0)&
                    (rhs.states_inf_only==I0)&
                    (rhs.states_rec_only==R0))[0]
print("len(init_state) =",len(init_state)) # Print to make sure we identify a unique state
x0[init_state] = 1.