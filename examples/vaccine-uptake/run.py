'''This runs the UK-like model with a single set of parameters for 100 days
'''
from copy import deepcopy
from numpy import arange, array, diag, log, ones, vstack, where, zeros
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import concat, read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate, estimate_hh_reproductive_ratio,
        SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import SEIR_VACC_SPEC, VACC_INF_RED, VACC_SUS_RED
from model.common import UnloopedSEIRRateEquations
from model.imports import FixedImportModel, NoImportModel, ExponentialImportModel

# pylint: disable=invalid-name

if isdir('outputs/vaccine-uptake') is False:
    mkdir('outputs/vaccine-uptake')

DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

# Composition list should be all possible combinations of unvaccinated and vaccinated:
uptake_df = concat([read_csv('inputs/hh_vax_1.csv').drop(["Unnamed: 0"], axis = 1),
                    read_csv('inputs/hh_vax_2.csv').drop(["Unnamed: 0", "week"], axis = 1),
                    read_csv('inputs/hh_vax_3.csv').drop(["Unnamed: 0", "week"], axis = 1),
                    read_csv('inputs/hh_vax_4.csv').drop(["Unnamed: 0", "week"], axis = 1),
                    read_csv('inputs/hh_vax_5.csv').drop(["Unnamed: 0", "week"], axis = 1),
                    read_csv('inputs/hh_vax_6.csv').drop(["Unnamed: 0", "week"], axis = 1)],
                          axis = 1)

hh_size_by_uptake_column = array(uptake_df.drop('week', axis=1).columns.str.extract('H=(\d+)')).astype(int)
vacc_by_uptake_column = array(uptake_df.drop('week', axis=1).columns.str.extract('V=(\d+)')).astype(int)
n_combinations = len(vacc_by_uptake_column)
composition_list = vstack([array([hh_size_by_uptake_column[i] - vacc_by_uptake_column[i],
                                  vacc_by_uptake_column[i]]).T for i in range(n_combinations)])

def get_rstar_at_time(t):
    composition_dist = uptake_df.drop('week', axis=1).iloc[t].to_numpy()
    composition_dist = composition_dist / sum(composition_dist)

    uptake = (composition_dist.dot(composition_list[:, 1]) / composition_dist.dot(hh_size_by_uptake_column))[0]
    TEMP_SPEC = deepcopy(SEIR_VACC_SPEC)
    TEMP_SPEC["k_ext"] = array([[1 - uptake, uptake],
                                [1 - uptake, uptake]])
    model_input_t = SEIRInput(TEMP_SPEC,
                                   composition_list,
                                   composition_dist)
    household_population_t = HouseholdPopulation(
        composition_list, composition_dist, model_input_t)

    rhs_t = UnloopedSEIRRateEquations(model_input_t,
                                    household_population_t,
                                    NoImportModel(4,
                                                  2))


    return (composition_dist,
            rhs_t,
            estimate_hh_reproductive_ratio(household_population_t,
                                           rhs_t))

results = [get_rstar_at_time(t) for t in range(uptake_df.shape[0])]

uptake_by_time = [r[0].dot(composition_list[:,1]) / r[1].household_population.ave_hh_size for r in results]
rstar_by_time = [r[2] for r in results]

# Compare against model with no vaccination
single_type_composition_list = array([[i] for i in range(1, hh_size_by_uptake_column.max()+1)])
single_type_composition_dist = uptake_df.drop('week', axis=1).iloc[0].to_numpy()
single_type_composition_dist = single_type_composition_dist[where(single_type_composition_dist>0)[0]]
single_type_composition_dist = single_type_composition_dist / sum(single_type_composition_dist)

# Can simulate effect of full vaccination by plugging in appropriate sus_scale and inf_scale, or no vaccination by
# setting both to zero
def get_single_type_rstar(sus_red,
                          inf_red):
    SINGLE_TYPE_SPEC = {
        'compartmental_structure': 'SEIR', # This is which subsystem key to use
        'beta_int': .1,                     # Internal infection rate
        'density_expo' : 1.,
        'recovery_rate': SEIR_VACC_SPEC['recovery_rate'],           # Recovery rate
        'incubation_rate': SEIR_VACC_SPEC['incubation_rate'],         # E->I incubation rate
        'sus': array([1. - sus_red]),          # Relative susceptibility by risk class
        'inf': [array([1. - inf_red])],
        'fit_method' : 'EL',
        'k_home': ones((1, 1), dtype=float),
        'k_ext': ones((1, 1), dtype=float),
        'skip_ext_scale' : True
    }

    single_type_input = SEIRInput(SINGLE_TYPE_SPEC,
                                  single_type_composition_list,
                                  single_type_composition_dist)
    single_type_household_population = HouseholdPopulation(
            single_type_composition_list, single_type_composition_dist, single_type_input)

    single_type_rhs = UnloopedSEIRRateEquations(single_type_input,
                                        single_type_household_population,
                                        NoImportModel(4,
                                                      1))
    return estimate_hh_reproductive_ratio(single_type_household_population,
                                          single_type_rhs)[0][0]

print("At zero uptake, single-type model gives R* =",
      get_single_type_rstar(0., 0.))
print("At full uptake, single-type model gives R* =",
      get_single_type_rstar(VACC_SUS_RED, VACC_INF_RED))