'''
Model a single household coupled to an age-structured SEPIR epidemic,
and estimate probability of observing longitudinal testing results.
'''

from cgi import test
from numpy import arange, array, diag, hstack, log, ones, shape, size, where, zeros
from numpy.linalg import eig
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from random import choices
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate, ModelInput,
        SEPIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import (TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC,
        TRANCHE2_SITP, PRODROME_PERIOD, PRODROME_SCALING, LATENT_PERIOD, SYMPTOM_PERIOD)
from model.common import SEPIRRateEquations
from model.imports import ImportModel, NoImportModel
# pylint: disable=invalid-name

if isdir('outputs/longitudinal_data_analysis') is False:
    mkdir('outputs/longitudinal_data_analysis')

'''Start by running the population-level epidemic.'''

tspan = (0.0, 90)

if isfile('outputs/longitudinal_data_analysis/background_epidemic.pkl'):
    with open('outputs/longitudinal_data_analysis/background_epidemic.pkl', 'rb') as f:
        uk_comp_list, uk_comp_dist, background_time, background_all_infs, background_input = load(f)
else:
    DOUBLING_TIME = 3
    growth_rate = log(2) / DOUBLING_TIME

    # List of observed household compositions
    uk_comp_list = read_csv(
        'inputs/eng_and_wales_adult_child_composition_list.csv',
        header=0).to_numpy()
    # Proportion of households which are in each composition
    uk_comp_dist = read_csv(
        'inputs/eng_and_wales_adult_child_composition_dist.csv',
        header=0).to_numpy().squeeze()


    if isfile('outputs/longitudinal_data_analysis/background_model_input.pkl') is True:
        with open('outputs/longitudinal_data_analysis/background_model_input.pkl', 'rb') as f:
            model_input = load(f)
    else:
        SPEC = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
        model_input_to_fit = SEPIRInput(SPEC, uk_comp_list, uk_comp_dist)
        household_population_to_fit = HouseholdPopulation(
            uk_comp_list, uk_comp_dist, model_input_to_fit)
        rhs_to_fit = SEPIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(5,2))
        beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
        model_input = model_input_to_fit
        model_input.k_ext *= beta_ext
        print('Estimated beta is',beta_ext)
        with open('outputs/longitudinal_data_analysis/background_model_input.pkl', 'wb') as f:
            dump(model_input, f)

    if isfile('outputs/longitudinal_data_analysis/background_population.pkl') is True:
        with open('outputs/longitudinal_data_analysis/background_population.pkl', 'rb') as f:
            uk_comp_list, uk_comp_dist, household_population = load(f)
    else:
        # With the parameters chosen, we calculate Q_int:
        household_population = HouseholdPopulation(
            uk_comp_list, uk_comp_dist, model_input)
        with open('outputs/longitudinal_data_analysis/background_population.pkl', 'wb') as f:
            dump((uk_comp_list, uk_comp_dist, household_population), f)

    rhs = SEPIRRateEquations(model_input, household_population, NoImportModel(5,2))

    H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, 1e-5, 0.0)
    solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

    background_time = solution.t
    background_H = solution.y
    background_P = background_H.T.dot(household_population.states[:, 2::5])
    background_I = background_H.T.dot(household_population.states[:, 3::5])
    background_all_infs = model_input.inf_scales[0] * background_P + model_input.inf_scales[1] * background_I

    background_input = model_input

    with open('outputs/longitudinal_data_analysis/background_epidemic.pkl', 'wb') as f:
        dump((uk_comp_list, uk_comp_dist, background_time, background_all_infs, background_input), f)

fitted_beta_int = background_input.beta_int
fitted_density_expo = background_input.density_expo

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

class SEPIRInputForEstimation(ModelInput):
    def __init__(self, beta_int, density_expo, spec, composition_list, composition_distribution):
        super().__init__(spec, composition_list, composition_distribution)

        self.expandables = ['sus',
                         'inf_scales']

        self.R_compartment = 4

        self.sus = spec['sus']
        self.inf_scales = [spec['prodromal_trans_scaling'],
                ones(shape(spec['prodromal_trans_scaling']))]



        self.alpha_2 = self.spec['symp_onset_rate']

        self.gamma = self.spec['recovery_rate']

        self.ave_trans = \
            ((self.inf_scales[0].dot(self.ave_hh_by_class) / self.ave_hh_size) /
            self.alpha_2) +  \
            ((self.inf_scales[1].dot(self.ave_hh_by_class) / self.ave_hh_size) /
             self.gamma)

        self.prog_rates = array([self.alpha_2, self.gamma])

        self.density_expo = density_expo

        self.beta_int = beta_int

        self.k_home = beta_int * self.k_home

        ext_eig = max(eig(
            diag(self.sus).dot((1/spec['symp_onset_rate']) *
             (self.k_ext).dot(self.inf_scales[0]) +
             (1/spec['recovery_rate']) *
              (self.k_ext).dot(diag(self.inf_scales[1])))
            )[0])
        if spec['fit_method'] == 'R*':
            external_scale = spec['R*'] / (self.ave_hh_size*spec['SITP'])
        else:
            external_scale = 1
        self.k_ext = external_scale * self.k_ext / ext_eig

    @property
    def alpha_1(self):
        return self.spec['incubation_rate']

def make_random_single_hh_system(full_comp_list, full_comp_dist):

    hh_id = choices(range(len(full_comp_list)), weights=full_comp_dist)
    single_hh_comp = full_comp_list[hh_id]

    model_input = SEPIRInputForEstimation( fitted_beta_int,
                                           fitted_density_expo,
                                           SPEC,
                                           single_hh_comp,
                                           array([1]))
    model_input.k_ext *= 0
    household_population = HouseholdPopulation(
        single_hh_comp, array([1]), model_input)

    rhs = SEPIRRateEquations(model_input,
                            household_population,
                            CoupledImportModel(background_time,background_all_infs))
    H0 = make_initial_condition_by_eigenvector(1., model_input, household_population, rhs, 0.0, 0.0)

    return single_hh_comp, H0, rhs

H0_list = []
rhs_list = []

for hh_id, comp in enumerate(uk_comp_list):

    model_input = SEPIRInputForEstimation( fitted_beta_int,
                                           fitted_density_expo,
                                           SPEC,
                                           array([comp]),
                                           array([1]))
    model_input.k_ext *= 0
    household_population = HouseholdPopulation(
        array([comp]), array([1]), model_input)

    rhs = SEPIRRateEquations(model_input,
                            household_population,
                            CoupledImportModel(background_time,background_all_infs))
    H0 = make_initial_condition_by_eigenvector(1., model_input, household_population, rhs, 0.0, 0.0)

    H0_list.append(H0)
    rhs_list.append(rhs)

'''Generate some household testing data from the model solution'''
'''First make a function which takes in a day's test positivity
and outputs a conditional distribution and a probability'''

def filter_for_positives(H, rhs, pos_by_class):
    possible_states = where(
                    (rhs.states_pro_only+rhs.states_inf_only - pos_by_class).sum(axis=1)==0)[0]
    H_conditional = zeros(size(H))
    H_conditional[possible_states] = H[possible_states]
    test_prob = sum(H_conditional)
    H_conditional *= (1/test_prob)
    return H_conditional, test_prob

def filter_for_anti(H, rhs, pos_by_class):
    possible_states = where(
                    (rhs.states_rec_only - pos_by_class).sum(axis=1)==0)[0]
    H_conditional = zeros(size(H))
    H_conditional[possible_states] = H[possible_states]
    test_prob = sum(H_conditional)
    H_conditional *= (1/test_prob)
    return H_conditional, test_prob

'''Now write a function to simulate testing on a given day'''

def simulate_testing(H, rhs):
    test_state = choices(range(len(H)), weights=H)
    pos_by_class = rhs.states_pro_only[test_state,:]+rhs.states_inf_only[test_state,:]
    rec_by_class = rhs.states_rec_only[test_state,:]
    return pos_by_class, rec_by_class

'''Now generate the testing data'''

test_dates = arange(1, tspan[-1], 2)

def generate_test_series(H0, rhs):

    pos_series = []
    anti_series = []

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
        pos, rec = simulate_testing(H_test_date, rhs)
        pos_series.append(pos)
        anti_series.append(rec)
        H_cond, prob = filter_for_positives(H_test_date, rhs, pos)
        H_cond, prob = filter_for_anti(H_cond, rhs, rec)
        current_H = H_cond
        H_all_time = hstack((H_all_time, H))
    
    return pos_series, anti_series

def calculate_test_probability(H0, rhs, pos_data, anti_data):
    log_prob = 0
    current_H = H0
    for d in range(1,len(test_dates)):
        this_week = (test_dates[d-1], test_dates[d])
        solution = solve_ivp(rhs, this_week, current_H, first_step=0.001, atol=1e-16)
        H = solution.y
        H_test_date = H[:, -1]
        H_cond, pos_prob = filter_for_positives(H_test_date, rhs, pos_data[d-1])
        H_cond, anti_prob = filter_for_anti(H_cond, rhs, anti_data[d-1])
        log_prob += log(pos_prob) + log(anti_prob)
        current_H = H_cond
    return log_prob

def calculate_llh(beta_int, density_expo, pos_data, anti_data, hh_id):
    comp = uk_comp_list[hh_id]
    model_input = SEPIRInputForEstimation( beta_int,
                                           density_expo,
                                           SPEC,
                                           array([comp]),
                                           array([1]))
    model_input.k_ext *= 0
    household_population = HouseholdPopulation(
        array([comp]), array([1]), model_input)

    rhs = SEPIRRateEquations(model_input,
                            household_population,
                            CoupledImportModel(background_time,background_all_infs))
    H0 = make_initial_condition_by_eigenvector(1., model_input, household_population, rhs, 0.0, 0.0)
    return calculate_test_probability(H0, rhs, pos_data, anti_data)

beta_int_range = arange(0.0050, 0.015, 0.0025)
density_expo_range = arange(0.0, 1.0, 0.2)

params = array([
        [b, e]
        for b in beta_int_range
        for e in density_expo_range])

def build_systems_with_params(beta_int, density_expo):
        
    temp_H0_list = []
    temp_rhs_list = []

    for hh_id, comp in enumerate(uk_comp_list):

        model_input = SEPIRInputForEstimation( beta_int,
                                            density_expo,
                                            SPEC,
                                            array([comp]),
                                            array([1]))
        model_input.k_ext *= 0
        household_population = HouseholdPopulation(
            array([comp]), array([1]), model_input)

        rhs = SEPIRRateEquations(model_input,
                                household_population,
                                CoupledImportModel(background_time,background_all_infs))
        H0 = make_initial_condition_by_eigenvector(1., model_input, household_population, rhs, 0.0, 0.0)

        temp_H0_list.append(H0)
        temp_rhs_list.append(rhs)

    return temp_H0_list, temp_rhs_list

H0_list_by_param = []
rhs_list_by_param = []

for i in range(len(params)):
    print("Building systems for params=", params[i,:])
    temp_H0_list, temp_rhs_list = build_systems_with_params(params[i, 0], params[i, 1])
    H0_list_by_param.append(temp_H0_list)
    rhs_list_by_param.append(temp_rhs_list)

'''Generate a bunch of samples of testing data so we don't have to do it again on every try:'''

if isfile('outputs/longitudinal_data_analysis/sample_test_data.pkl') is True:
    with open('outputs/longitudinal_data_analysis/sample_test_data.pkl', 'rb') as f:
        sample_comp_ids, sample_H0s, sample_rhss, sample_test_data = load(f)
else:
    no_test_samples = 1000
    sample_comp_ids = choices(range(len(uk_comp_list)), weights=uk_comp_dist, k=no_test_samples)
    sample_H0s = [H0_list[hh_id] for hh_id in sample_comp_ids]
    sample_rhss = [rhs_list[hh_id] for hh_id in sample_comp_ids]
    sample_test_data = [generate_test_series(sample_H0s[i], sample_rhss[i]) for i in range(no_test_samples)]
    with open('outputs/longitudinal_data_analysis/background_population.pkl', 'wb') as f:
        dump((sample_comp_ids, sample_H0s, sample_rhss, sample_test_data), f)

no_hh_in_data = 50
which_samples = choices(range(len(sample_test_data)), k=no_hh_in_data)
test_data_list = [sample_test_data[sample_id] for sample_id in which_samples]

llh_array = zeros(len(params),)

for i in range(len(params)):
    print("Doing likelihood calculations for params=", params[i,:])
    temp_H0_list = H0_list_by_param[i]
    temp_rhs_list = rhs_list_by_param[i]
    for j in range(no_hh_in_data):
        llh_array[i] += calculate_test_probability(temp_H0_list[sample_comp_ids[which_samples[j]]],
                                                temp_rhs_list[sample_comp_ids[which_samples[j]]],
                                                test_data_list[j][0],
                                                test_data_list[j][1])

with open('outputs/longitudinal_data_analysis/llh_array_50_fineres.pkl', 'wb') as f:
    dump(llh_array, f)

from matplotlib.pyplot import axes, close, colorbar, imshow, set_cmap, subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from seaborn import heatmap

if isdir('plots/longitudinal_data_analysis') is False:
    mkdir('plots/longitudinal_data_analysis')

llh_array = llh_array.reshape((len(beta_int_range), len(density_expo_range)))

fig, ax = subplots(1, 1, constrained_layout=True)
axim = ax.contour(llh_array,
                  colors='k',
                  extent=(0, 1, 0, max(beta_int_range)),
                  aspect=1)
ax.scatter([background_input.density_expo], [background_input.beta_int])
ax.clabel(axim, fontsize=9, inline=1, fmt='%1.1f')
ax.set_xlabel("density exponent")
ax.set_ylabel("beta_int")
ax.set_title("Log likelihood")

fig.savefig('plots/longitudinal_data_analysis/llh_array_50_fine_2.png',
            bbox_inches='tight',
            dpi=300,
            transparent=False)
close()