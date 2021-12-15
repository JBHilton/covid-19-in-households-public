'''This runs the UK-like model with a single set of parameters for 100 days
'''
from abc import ABC
from copy import deepcopy
from numpy import array, concatenate, diag, hstack, log, savetxt, shape, where, zeros
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from time import time as get_time
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEPIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import THREE_AGE_SEPIR_SPEC_FOR_FITTING, BDS_20_65_UK_SPEC
from model.common import SEPIRRateEquations
from model.imports import NoImportModel
# pylint: disable=invalid-name

if isdir('outputs/multigen') is False:
    mkdir('outputs/multigen')

class Outputs(ABC):
    def __init__(self, name):
        self.name = name

RED_LEVEL = 0.5
RED_LEVEL_2 = 0.5 # Secondary reduction for bidirectional studies

NO_COMPS = 5
NO_CLASSES = 3
DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

# List of observed household compositions
composition_list = read_csv(
    'inputs/eng_wales_20_65_u6_comp_list.csv',
    header=0).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/eng_wales_20_65_u6_comp_dist.csv',
    header=0).to_numpy().squeeze()


if isfile('outputs/multigen/fitted_model_input.pkl') is True:
    with open('outputs/multigen/fitted_model_input.pkl', 'rb') as f:
        model_input = load(f)
else:
    SPEC = {**THREE_AGE_SEPIR_SPEC_FOR_FITTING, **BDS_20_65_UK_SPEC}
    model_input_to_fit = SEPIRInput(SPEC, composition_list, comp_dist)
    model_input_to_fit.k_home *= 2 # Double within-hh trans for Omicron
    household_population_to_fit = HouseholdPopulation(
        composition_list, comp_dist, model_input_to_fit)
    rhs_to_fit = SEPIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(5,2))
    beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
    model_input = model_input_to_fit
    model_input.k_ext *= beta_ext
    with open('outputs/multigen/fitted_model_input.pkl', 'wb') as f:
        dump(model_input, f)

if isfile('outputs/multigen/household_population.pkl') is True:
    with open('outputs/multigen/household_population.pkl', 'rb') as f:
        household_population = load(f)
else:
    # With the parameters chosen, we calculate Q_int:
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input)
    with open('outputs/multigen/household_population.pkl', 'wb') as f:
        dump(household_population, f)

cls0_size = sum(model_input.pop_pyramid[:4])
cls1_size = sum(model_input.pop_pyramid[4:14])
cls2_size = sum(model_input.pop_pyramid[14:])
size_map = diag(array([cls0_size, cls1_size,cls2_size]))

start_time = get_time()

rhs = SEPIRRateEquations(model_input,
                        household_population,
                        NoImportModel(NO_COMPS, NO_CLASSES))

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, 1e-3, 0.0)
base_H0 = H0
tspan = (0.0, 120)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

baseline = Outputs('baseline')

baseline.t = array(solution.t, ndmin=2).T
H = solution.y

baseline.P = H.T.dot(household_population.states[:, 2::NO_COMPS]) / \
            model_input.ave_hh_by_class
baseline.I = H.T.dot(household_population.states[:, 3::NO_COMPS]) / \
            model_input.ave_hh_by_class
baseline.R = H.T.dot(household_population.states[:, 4::NO_COMPS]) / \
            model_input.ave_hh_by_class
time_series = hstack((baseline.t, (baseline.P + baseline.I).dot(size_map), baseline.R.dot(size_map)))
savetxt('outputs/multigen/baseline_time_series.csv',
        time_series,
        delimiter = ',',
        header='time, 0-19_prev, 20-64_prev, 65+_prev, 0-19_cum_prev, 20-64_cum_prev, 65+_cum_prev')

baseline.P_allhh = H.T.dot(household_population.states[:, 2 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]
baseline.I_allhh = H.T.dot(household_population.states[:, 3 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]
baseline.R_allhh = H.T.dot(household_population.states[:, 4 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]

cls0_by_comp = household_population.composition_by_state[:, 0]
cls1_by_comp = household_population.composition_by_state[:, 1]
cls2_by_comp = household_population.composition_by_state[:, 2]

comp_is_001 = \
    where((cls0_by_comp==0)&((cls1_by_comp==0)&(cls2_by_comp>0)))[0]
comp_is_101 = \
    where((cls0_by_comp>0)&((cls1_by_comp==0)&(cls2_by_comp>0)))[0]
comp_is_011 = \
    where((cls0_by_comp==0)&((cls1_by_comp>0)&(cls2_by_comp>0)))[0]
comp_is_111 = \
    where((cls0_by_comp>0)&((cls1_by_comp>0)&(cls2_by_comp>0)))[0]

denom_001 = cls2_by_comp[comp_is_001].dot(H[comp_is_001,0])
denom_101 = cls2_by_comp[comp_is_101].dot(H[comp_is_101,0])
denom_011 = cls2_by_comp[comp_is_011].dot(H[comp_is_011,0])
denom_111 = cls2_by_comp[comp_is_111].dot(H[comp_is_111,0])

baseline.P_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 2 + 2 * NO_COMPS])/denom_001
baseline.I_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 3 + 2 * NO_COMPS])/denom_001
baseline.R_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 4 + 2 * NO_COMPS])/denom_001
baseline.P_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 2 + 2 * NO_COMPS])/denom_011
baseline.I_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 3 + 2 * NO_COMPS])/denom_011
baseline.R_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 4 + 2 * NO_COMPS])/denom_011
baseline.P_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 2 + 2 * NO_COMPS])/denom_101
baseline.I_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 3 + 2 * NO_COMPS])/denom_101
baseline.R_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 4 + 2 * NO_COMPS])/denom_101
baseline.P_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 2 + 2 * NO_COMPS])/denom_111
baseline.I_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 3 + 2 * NO_COMPS])/denom_111
baseline.R_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 4 + 2 * NO_COMPS])/denom_111

print('Baseline done in',get_time()-start_time,'seconds.')
start_time = get_time()

model_input_whh_red = deepcopy(model_input)
model_input_whh_red.k_home *= 1 - RED_LEVEL

household_population_whh_red = HouseholdPopulation(
        composition_list, comp_dist, model_input_whh_red)

rhs_whh_red = SEPIRRateEquations(model_input_whh_red,
                        household_population_whh_red,
                        NoImportModel(NO_COMPS, NO_CLASSES))

H0 = make_initial_condition_by_eigenvector(growth_rate,
                                           model_input_whh_red,
                                           household_population_whh_red,
                                           rhs_whh_red,
                                           1e-3,
                                           0.0)
tspan = (0.0, 120)
solution = solve_ivp(rhs_whh_red, tspan, H0, first_step=0.001, atol=1e-16)

whh_red = Outputs('within-household measures')

whh_red.t = array(solution.t, ndmin=2).T
H = solution.y

whh_red.P = H.T.dot(household_population.states[:, 2::NO_COMPS]) / \
            model_input.ave_hh_by_class
whh_red.I = H.T.dot(household_population.states[:, 3::NO_COMPS]) / \
            model_input.ave_hh_by_class
whh_red.R = H.T.dot(household_population.states[:, 4::NO_COMPS]) / \
            model_input.ave_hh_by_class
time_series = hstack((whh_red.t, (whh_red.P + whh_red.I).dot(size_map), whh_red.R.dot(size_map)))
savetxt('outputs/multigen/whh_red_time_series_' + \
        str(int(100*RED_LEVEL)) + \
        '_.csv',
        time_series,
        delimiter = ',',
        header='time, 0-19_prev, 20-64_prev, 65+_prev, 0-19_cum_prev, 20-64_cum_prev, 65+_cum_prev')

whh_red.P_allhh = H.T.dot(household_population.states[:, 2 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]
whh_red.I_allhh = H.T.dot(household_population.states[:, 3 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]
whh_red.R_allhh = H.T.dot(household_population.states[:, 4 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]

whh_red.P_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 2 + 2 * NO_COMPS])/denom_001
whh_red.I_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 3 + 2 * NO_COMPS])/denom_001
whh_red.R_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 4 + 2 * NO_COMPS])/denom_001
whh_red.P_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 2 + 2 * NO_COMPS])/denom_011
whh_red.I_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 3 + 2 * NO_COMPS])/denom_011
whh_red.R_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 4 + 2 * NO_COMPS])/denom_011
whh_red.P_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 2 + 2 * NO_COMPS])/denom_101
whh_red.I_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 3 + 2 * NO_COMPS])/denom_101
whh_red.R_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 4 + 2 * NO_COMPS])/denom_101
whh_red.P_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 2 + 2 * NO_COMPS])/denom_111
whh_red.I_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 3 + 2 * NO_COMPS])/denom_111
whh_red.R_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 4 + 2 * NO_COMPS])/denom_111

print('Within-household done in',get_time()-start_time,'seconds.')
start_time = get_time()

model_input_bhh_red = deepcopy(model_input)
model_input_bhh_red.k_ext *= 1 - RED_LEVEL

household_population_bhh_red = HouseholdPopulation(
        composition_list, comp_dist, model_input_bhh_red)

rhs_bhh_red = SEPIRRateEquations(model_input_bhh_red,
                        household_population_bhh_red,
                        NoImportModel(NO_COMPS, NO_CLASSES))

H0 = make_initial_condition_by_eigenvector(growth_rate,
                                           model_input_bhh_red,
                                           household_population_bhh_red,
                                           rhs_bhh_red,
                                           1e-3,
                                           0.0)
tspan = (0.0, 120)
solution = solve_ivp(rhs_bhh_red, tspan, H0, first_step=0.001, atol=1e-16)

bhh_red = Outputs('between-household measures')

bhh_red.t = array(solution.t, ndmin=2).T
H = solution.y

bhh_red.P = H.T.dot(household_population.states[:, 2::NO_COMPS]) / \
            model_input.ave_hh_by_class
bhh_red.I = H.T.dot(household_population.states[:, 3::NO_COMPS]) / \
            model_input.ave_hh_by_class
bhh_red.R = H.T.dot(household_population.states[:, 4::NO_COMPS]) / \
            model_input.ave_hh_by_class
time_series = hstack((bhh_red.t, (bhh_red.P + bhh_red.I).dot(size_map), bhh_red.R.dot(size_map)))
savetxt('outputs/multigen/bhh_red_time_series_' + \
        str(int(100*RED_LEVEL)) + \
        '_.csv',
        time_series,
        delimiter = ',',
        header='time, 0-19_prev, 20-64_prev, 65+_prev, 0-19_cum_prev, 20-64_cum_prev, 65+_cum_prev')

bhh_red.P_allhh = H.T.dot(household_population.states[:, 2 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]
bhh_red.I_allhh = H.T.dot(household_population.states[:, 3 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]
bhh_red.R_allhh = H.T.dot(household_population.states[:, 4 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]

bhh_red.P_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 2 + 2 * NO_COMPS])/denom_001
bhh_red.I_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 3 + 2 * NO_COMPS])/denom_001
bhh_red.R_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 4 + 2 * NO_COMPS])/denom_001
bhh_red.P_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 2 + 2 * NO_COMPS])/denom_011
bhh_red.I_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 3 + 2 * NO_COMPS])/denom_011
bhh_red.R_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 4 + 2 * NO_COMPS])/denom_011
bhh_red.P_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 2 + 2 * NO_COMPS])/denom_101
bhh_red.I_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 3 + 2 * NO_COMPS])/denom_101
bhh_red.R_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 4 + 2 * NO_COMPS])/denom_101
bhh_red.P_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 2 + 2 * NO_COMPS])/denom_111
bhh_red.I_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 3 + 2 * NO_COMPS])/denom_111
bhh_red.R_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 4 + 2 * NO_COMPS])/denom_111

print('Between-household done in',get_time()-start_time,'seconds.')
start_time = get_time()

model_input_both_red = deepcopy(model_input)
model_input_both_red.k_home *= 1 - RED_LEVEL
model_input_both_red.k_ext *= 1 - RED_LEVEL_2

household_population_both_red = HouseholdPopulation(
        composition_list, comp_dist, model_input_both_red)

rhs_both_red = SEPIRRateEquations(model_input_both_red,
                        household_population_both_red,
                        NoImportModel(NO_COMPS, NO_CLASSES))

H0 = make_initial_condition_by_eigenvector(growth_rate,
                                           model_input_both_red,
                                           household_population_both_red,
                                           rhs_both_red,
                                           1e-3,
                                           0.0)
tspan = (0.0, 120)
solution = solve_ivp(rhs_both_red, tspan, H0, first_step=0.001, atol=1e-16)

both_red = Outputs('both measures')

both_red.t = array(solution.t, ndmin=2).T
H = solution.y


both_red.P = H.T.dot(household_population.states[:, 2::NO_COMPS]) / \
            model_input.ave_hh_by_class
both_red.I = H.T.dot(household_population.states[:, 3::NO_COMPS]) / \
            model_input.ave_hh_by_class
both_red.R = H.T.dot(household_population.states[:, 4::NO_COMPS]) / \
            model_input.ave_hh_by_class
time_series = hstack((both_red.t, (both_red.P + both_red.I).dot(size_map), both_red.R.dot(size_map)))
savetxt('outputs/multigen/both_red_time_series_' + \
        str(int(100*RED_LEVEL)) + \
        '_'+str(int(100*RED_LEVEL_2)) + \
        '.csv',
        time_series,
        delimiter = ',',
        header='time, 0-19_prev, 20-64_prev, 65+_prev, 0-19_cum_prev, 20-64_cum_prev, 65+_cum_prev')

both_red.P_allhh = H.T.dot(household_population.states[:, 2 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]
both_red.I_allhh = H.T.dot(household_population.states[:, 3 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]
both_red.R_allhh = H.T.dot(household_population.states[:, 4 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]

both_red.P_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 2 + 2 * NO_COMPS])/denom_001
both_red.I_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 3 + 2 * NO_COMPS])/denom_001
both_red.R_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 4 + 2 * NO_COMPS])/denom_001
both_red.P_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 2 + 2 * NO_COMPS])/denom_011
both_red.I_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 3 + 2 * NO_COMPS])/denom_011
both_red.R_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 4 + 2 * NO_COMPS])/denom_011
both_red.P_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 2 + 2 * NO_COMPS])/denom_101
both_red.I_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 3 + 2 * NO_COMPS])/denom_101
both_red.R_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 4 + 2 * NO_COMPS])/denom_101
both_red.P_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 2 + 2 * NO_COMPS])/denom_111
both_red.I_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 3 + 2 * NO_COMPS])/denom_111
both_red.R_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 4 + 2 * NO_COMPS])/denom_111

print('Both levels done in',get_time()-start_time,'seconds.')
start_time = get_time()

class TargettedInterventionRateEquations(SEPIRRateEquations):
    def __init__(self,
                 model_input,
                 household_population,
                 import_model,
                 shield_comp_list,
                 shield_comp_red = 0.5):
        super().__init__(model_input,
            household_population,
            import_model)
        self.shield_comp_red = shield_comp_red
        self.shield_comps = concatenate(shield_comp_list)

    def get_FOI_by_class(self, t, H):
        '''This calculates the age-stratified force-of-infection (FOI) on each
        household composition'''
        # Average number of each class by household
        denom = H.T.dot(self.composition_by_state)

        FOI = self.states_sus_only.dot(diag(self.import_model.cases(t)))

        for ic in range(self.no_inf_compartments):
            states_inf_only = self.inf_by_state_list[ic]
            inf_by_class = zeros(shape(denom))
            inf_by_class[denom > 0] = (
                H.T.dot(states_inf_only)[denom > 0]
                / denom[denom > 0]).squeeze()
            FOI += self.states_sus_only.dot(
                    diag(self.ext_matrix_list[ic].dot(
                        self.epsilon * inf_by_class.T)))
        FOI[self.shield_comps, :] = \
            self.shield_comp_red * FOI[self.shield_comps, :]

        return FOI

rhs_targ = TargettedInterventionRateEquations(model_input,
                        household_population,
                        NoImportModel(NO_COMPS, NO_CLASSES),
                        [comp_is_001, comp_is_011, comp_is_101, comp_is_111],
                        1 - RED_LEVEL)

solution = solve_ivp(rhs_targ, tspan, base_H0, first_step=0.001, atol=1e-16)

targ = Outputs('targetted')

targ.t = array(solution.t, ndmin=2).T
H = solution.y

targ.P = H.T.dot(household_population.states[:, 2::NO_COMPS]) / \
            model_input.ave_hh_by_class
targ.I = H.T.dot(household_population.states[:, 3::NO_COMPS]) / \
            model_input.ave_hh_by_class
targ.R = H.T.dot(household_population.states[:, 4::NO_COMPS]) / \
            model_input.ave_hh_by_class
time_series = hstack((targ.t, (targ.P + targ.I).dot(size_map), targ.R.dot(size_map)))
savetxt('outputs/multigen/targ_time_series_' + \
        str(int(100*RED_LEVEL)) + \
        '_.csv',
        time_series,
        delimiter = ',',
        header='time, 0-19_prev, 20-64_prev, 65+_prev, 0-19_cum_prev, 20-64_cum_prev, 65+_cum_prev')

targ.P_allhh = H.T.dot(household_population.states[:, 2 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]
targ.I_allhh = H.T.dot(household_population.states[:, 3 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]
targ.R_allhh = H.T.dot(household_population.states[:, 4 + 2 * NO_COMPS]) / \
            model_input.ave_hh_by_class[2]

targ.P_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 2 + 2 * NO_COMPS])/denom_001
targ.I_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 3 + 2 * NO_COMPS])/denom_001
targ.R_001 = H[comp_is_001,:].T.dot(household_population.states[comp_is_001, 4 + 2 * NO_COMPS])/denom_001
targ.P_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 2 + 2 * NO_COMPS])/denom_011
targ.I_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 3 + 2 * NO_COMPS])/denom_011
targ.R_011 = H[comp_is_011,:].T.dot(household_population.states[comp_is_011, 4 + 2 * NO_COMPS])/denom_011
targ.P_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 2 + 2 * NO_COMPS])/denom_101
targ.I_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 3 + 2 * NO_COMPS])/denom_101
targ.R_101 = H[comp_is_101,:].T.dot(household_population.states[comp_is_101, 4 + 2 * NO_COMPS])/denom_101
targ.P_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 2 + 2 * NO_COMPS])/denom_111
targ.I_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 3 + 2 * NO_COMPS])/denom_111
targ.R_111 = H[comp_is_111,:].T.dot(household_population.states[comp_is_111, 4 + 2 * NO_COMPS])/denom_111

print('Targetted done in',get_time()-start_time,'seconds.')
start_time = get_time()

fname = 'outputs/multigen/results_' + \
        str(int(100*RED_LEVEL)) + \
        '_'+str(int(100*RED_LEVEL_2)) + \
        '.pkl'
with open(fname, 'wb') as f:
    dump((baseline, whh_red, bhh_red, both_red, targ), f)
