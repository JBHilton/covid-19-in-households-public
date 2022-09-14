'''
Model a single household coupled to an age-structured SEPIR epidemic,
and estimate probability of observing longitudinal testing results.
'''

from cgi import test
from numpy import arange, array, hstack, log, ones, size, where, zeros
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from random import choices
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

'''Generate some household testing data from the model solution'''
'''First make a function which takes in a day's test positivity
and outputs a conditional distribution and a probability'''

def filter_for_positives(H, pos_by_class):
    possible_states = where(
                    (rhs.states_pro_only+rhs.states_inf_only - pos_by_class).sum(axis=1)==0)[0]
    H_conditional = zeros(size(H))
    H_conditional[possible_states] = H[possible_states]
    test_prob = sum(H_conditional)
    H_conditional *= (1/test_prob)
    return H_conditional, test_prob

'''Now write a function to simulate testing on a given day'''

def simulate_testing(H):
    test_state = choices(range(len(H)), weights=H)
    pos_by_class = rhs.states_pro_only[test_state,:]+rhs.states_inf_only[test_state,:]
    return pos_by_class

'''Now generate the testing data'''

test_dates = arange(1, tspan[-1], 7)

test_series = []

H_all_time = H0.reshape(-1,1)
time_points = array([])

current_H = H0
for d in range(1,len(test_dates)):
    this_week = (test_dates[d-1], test_dates[d])
    solution = solve_ivp(rhs, this_week, current_H, first_step=0.001, atol=1e-16)
    time = solution.t
    time_points = hstack((time_points, time))
    H = solution.y
    H_test_date = H[:, -1]
    test_results = simulate_testing(H_test_date)
    test_series.append(test_results)
    H_cond, prob = filter_for_positives(H_test_date, test_results)
    current_H = H_cond
    H_all_time = hstack((H_all_time, H))

def calculate_test_probability(H0, pos_data):
    series_prob = 1
    current_H = H0
    for d in range(1,len(test_dates)):
        this_week = (test_dates[d-1], test_dates[d])
        solution = solve_ivp(rhs, this_week, current_H, first_step=0.001, atol=1e-16)
        H = solution.y
        H_test_date = H[:, -1]
        test_results = simulate_testing(H_test_date)
        test_series.append(test_results)
        H_cond, prob = filter_for_positives(H_test_date, pos_data[d])
        series_prob *= prob
        current_H = H_cond
    return series_prob