'''
Model a single household coupled to an age-structured SEPIR epidemic,
and estimate probability of observing longitudinal testing results.
'''

from numpy import array, log, ones
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEPIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import (TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC,
        TRANCHE2_SITP, PRODROME_PERIOD, PRODROME_SCALING, LATENT_PERIOD, SYMPTOM_PERIOD)
from model.common import SEPIRRateEquations
from model.imports import ImportModel, NoImportModel
# pylint: disable=invalid-name

if isdir('outputs/longitudinal_data_analysis') is False:
    mkdir('outputs/longitudinal_data_analysis')

'''Start by running the population-level epidemic.'''

if isfile('outputs/longitudinal_data_analysis/background_epidemic.pkl'):
    with open('outputs/longitudinal_data_analysis/background_epidemic.pkl', 'rb') as f:
        background_time, background_all_infs = load(f)
else:
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


    if isfile('outputs/longitudinal_data_analysis/background_model_input.pkl') is True:
        with open('outputs/longitudinal_data_analysis/background_model_input.pkl', 'rb') as f:
            model_input = load(f)
    else:
        SPEC = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
        model_input_to_fit = SEPIRInput(SPEC, composition_list, comp_dist)
        household_population_to_fit = HouseholdPopulation(
            composition_list, comp_dist, model_input_to_fit)
        rhs_to_fit = SEPIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(5,2))
        beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
        model_input = model_input_to_fit
        model_input.k_ext *= beta_ext
        print('Estimated beta is',beta_ext)
        with open('outputs/longitudinal_data_analysis/background_model_input.pkl', 'wb') as f:
            dump(model_input, f)

    if isfile('outputs/longitudinal_data_analysis/background_population.pkl') is True:
        with open('outputs/longitudinal_data_analysis/background_population.pkl', 'rb') as f:
            household_population = load(f)
    else:
        # With the parameters chosen, we calculate Q_int:
        household_population = HouseholdPopulation(
            composition_list, comp_dist, model_input)
        with open('outputs/longitudinal_data_analysis/background_population.pkl', 'wb') as f:
            dump(household_population, f)

    rhs = SEPIRRateEquations(model_input, household_population, NoImportModel(5,2))

    H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, 1e-5, 0.0)
    tspan = (0.0, 365)
    solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

    background_time = solution.t
    background_H = solution.y
    background_P = background_H.T.dot(household_population.states[:, 2::5])
    background_I = background_H.T.dot(household_population.states[:, 3::5])
    background_all_infs = model_input.inf_scales[0] * background_P + model_input.inf_scales[1] * background_I

    with open('outputs/longitudinal_data_analysis/background_epidemic.pkl', 'wb') as f:
        dump((background_time, background_all_infs), f)

'''Create an import model coupling single HH to epidemic'''

class CoupledImportModel(ImportModel):
    def __init__(
            self,
            t_vals,
            background_epidemic):       # External prevalence is now a age classes by inf compartments array
        self.prevalence_interpolant = interp1d(
                t_vals.T, background_epidemic.T,
                kind='nearest',
                bounds_error=False,
                fill_value='extrapolate',
                assume_sorted=True)

    def cases(self, t):
        imports = self.prevalence_interpolant(t)
        return imports

'''Now construct the household population object for a single household coupled to the main epidemic'''

composition_list = array([[2, 2]]) # 2 adult 2 child household
comp_dist = array([1])

SINGLE_HH_SEPIR_SPEC = {
    'compartmental_structure': 'SEPIR', # This is which subsystem key to use
    'SITP': TRANCHE2_SITP,                     # Secondary inf probability
    'R*': 1.,                      # Household-level reproduction number
    'recovery_rate': 1 / SYMPTOM_PERIOD,           # Recovery rate
    'incubation_rate': 1 / LATENT_PERIOD,         # E->P incubation rate
    'symp_onset_rate': 1 / PRODROME_PERIOD,         # P->I prodromal to symptomatic rate
    'prodromal_trans_scaling':
     PRODROME_SCALING * ones(2,),          # Prodromal transmission intensity relative to
                                # full inf transmission
    'sus': array([1,1]),          # Relative susceptibility by
                                  # age/vulnerability class
    'fit_method' : 'EL'
}

SPEC = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}

model_input = SEPIRInput(SPEC, composition_list, comp_dist)
model_input.k_ext *= 0
household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

rhs = SEPIRRateEquations(model_input,
                         household_population,
                         CoupledImportModel(background_time,background_all_infs))
H0 = make_initial_condition_by_eigenvector(1., model_input, household_population, rhs, 0.0, 0.0)

'''Now solve'''

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
'I':I,
'R':R
}

with open('outputs/longitudinal_data_analysis/results.pkl', 'wb') as f:
    dump((H, time_series), f)