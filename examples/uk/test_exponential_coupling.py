'''This explores the behaviour of the model with exponential coupling.
We simulate a coupled epidemic, calibrate an exponential curve to it,
and compare an uncoupled epidemic coupled to that curve.
'''
from copy import deepcopy
from matplotlib.pyplot import subplots
from matplotlib.pyplot import yscale
from matplotlib.cm import get_cmap
from numpy import arange, array, exp, log, outer
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEPIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import UnloopedSEPIRRateEquations
from model.imports import ExponentialImportModel, NoImportModel
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
rhs_to_fit = UnloopedSEPIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(5,2))
beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
model_input = deepcopy(model_input_to_fit)
model_input.k_ext *= beta_ext
print('Estimated beta is',beta_ext)

# With the parameters chosen, we calculate Q_int:
household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

no_imports = NoImportModel(5, 2)
base_rhs = UnloopedSEPIRRateEquations(model_input, household_population, no_imports)
base_H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, base_rhs, 1e-5, 0.0)
tspan = (0.0, 365)
base_sol = solve_ivp(base_rhs, tspan, base_H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))
base_H = base_sol.y
base_t = base_sol.t
base_cases = (base_H.T.dot(household_population.states[:, 1::5]) +
               base_H.T.dot(household_population.states[:, 2::5]) +
               base_H.T.dot(household_population.states[:, 3::5])) / model_input.ave_hh_by_class
exp_growth_time = arange(-15, 15, 1)
exp_curve = outer(exp(growth_rate * exp_growth_time), base_cases[15,:])
# Check exponential curve correctly approximates cases:
print("Difference between coupled model and exponential growth in first 30 days:",
      abs(base_cases[:30,:] - exp_curve)/base_cases[:30, :])

# Set up appropriate exponential coupling - needs to be based on prodromal and symptomatic prevalence from coupled
# model weighted by compartment-specific infectivity.
base_exp_pressure = (model_input.inf_scales[0] * base_H.T.dot(household_population.states[:, 2::5]) +
                     model_input.inf_scales[1] * base_H.T.dot(household_population.states[:, 3::5])) / model_input.ave_hh_by_class
# Now set up exponential coupling model
exp_imports = ExponentialImportModel(5,
                                     2,
                                     base_rhs,
                                     growth_rate,
                                     base_exp_pressure[15,:])

rhs = UnloopedSEPIRRateEquations(model_input, household_population, exp_imports, sources="IMPORT")

H0 = base_H[:,15]
tspan = (0.0, 365)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))

t = solution.t
H = solution.y
S = H.T.dot(household_population.states[:, ::5])
E = H.T.dot(household_population.states[:, 1::5])
P = H.T.dot(household_population.states[:, 2::5])
I = H.T.dot(household_population.states[:, 3::5])
R = H.T.dot(household_population.states[:, 4::5])

total_cases = (E+P+I) / model_input.ave_hh_by_class

# Check whether coupling to exponential approximates fully coupled model in early growth phase
print("Difference between coupled model and exponential coupling during exponential period:",
      abs(base_cases[15:30,:] - total_cases[:15, :])/base_cases[15:30, :])


lgd=['Exponential curve',
     'Nonlinear coupled model',
     'Exponential import model']

fig, ax = subplots(1, 1, sharex=True)

cmap = get_cmap('tab20')
alpha = 1
ax.plot(
        exp_growth_time + 15,
        exp_curve,
        'k-',
        label=lgd[0],
        alpha=alpha)
ax.plot(
        base_t[:30],
        base_cases[:30],
        'rx',
        label=lgd[1],
        alpha=alpha)

ax.plot(
        t[:15] + 15,
        total_cases[:15],
        'g.',
        label=lgd[2],
        alpha=alpha)
yscale('log')
ax.set_xlabel('Time in days')
ax.set_ylabel('Prevalence')
ax.legend(ncol=1,
          loc="lower right")
ax.set_box_aspect(1)
fig.show()


# with open('outputs/uk/exponential_coupling_output.pkl', 'wb') as f:
#     dump((H), f)
