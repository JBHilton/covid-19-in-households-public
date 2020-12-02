from numpy import arange, atleast_2d, hstack, vstack
from os.path import isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from time import time
from model.preprocessing import (
        ModelInput, HouseholdPopulation)
from model.specs import DEFAULT_SPEC
from examples.temp_bubbles.common import (
        build_mixed_compositions, build_mixed_compositions_pairwise, SINGLE_AGE_CLASS_SPEC, SingleClassInput,
        MergedInput,pairwise_merged_initial_condition, pairwise_demerged_initial_condition,
        merged_initial_condition, demerged_initial_condition,
        make_initial_condition, within_household_SEPIR,RateEquations, within_household_SEIR)

unmerged_input = SingleClassInput(SINGLE_AGE_CLASS_SPEC)
hh_to_merge = 2
max_size = 12
mixing_strength = 0.5
merged_input = MergedInput(SINGLE_AGE_CLASS_SPEC,hh_to_merge,mixing_strength)

comp_dist = read_csv(
    'inputs/england_hh_size_dist.csv',
    header=0).to_numpy().squeeze()
comp_dist = comp_dist[:5]
comp_dist = comp_dist/sum(comp_dist)
max_hh_size = len(comp_dist)
composition_list = atleast_2d(arange(1,max_hh_size+1)).T

unmerged_population = HouseholdPopulation(
   composition_list,
   comp_dist,
   unmerged_input,
   within_household_SEPIR,
   5,
   True)

pw_start = time()
pw_merged_comp_list, pw_merged_comp_dist, pw_hh_dimension, pw_pairings = \
 build_mixed_compositions_pairwise(composition_list, comp_dist)
pw_merged_population = HouseholdPopulation(
    pw_merged_comp_list,
    pw_merged_comp_dist,
    merged_input,
    within_household_SEPIR,
    5,
    True)
print('Pairwise construction took', time() - pw_start, 'seconds.')

gen_start = time()
merged_comp_list, merged_comp_dist, hh_dimension, pairings = \
    build_mixed_compositions(composition_list, comp_dist, hh_to_merge, max_size)
merged_population = HouseholdPopulation(
    merged_comp_list,
    merged_comp_dist,
    merged_input,
    within_household_SEPIR,
    5,
    True)
print('General construction took', time() - gen_start, 'seconds.')

merged_input_3 = MergedInput(SINGLE_AGE_CLASS_SPEC,3,mixing_strength)
start_3_6 = time()
merged_comp_list_3_6, merged_comp_dist_3_6, hh_dimension_3_6, pairings_3_6 = \
    build_mixed_compositions(composition_list, comp_dist, 3, 6)
# merged_population_3_6 = HouseholdPopulation(
#     merged_comp_list_3_6,
#     merged_comp_dist_3_6,
#     merged_input_3,
#     within_household_SEPIR,
#     5,
#     True)
# print('3 hh, max 6 construction took', time() - start_3_6, 'seconds.')
# print('System size is', sum(merged_population_3_6.system_sizes),'.')

start_3_6_SEIR = time()
merged_population_3_6_SEIR = HouseholdPopulation(
    merged_comp_list_3_6,
    merged_comp_dist_3_6,
    merged_input_3,
    within_household_SEIR,
    4,
    True)
print('3 hh, max 6 construction took', time() - start_3_6_SEIR, 'seconds.')
print('System size is', sum(merged_population_3_6_SEIR.system_sizes),'.')

start_3_8 = time()
merged_comp_list_3_8, merged_comp_dist_3_8, hh_dimension_3_8, pairings_3_8 = \
    build_mixed_compositions(composition_list, comp_dist, 3, 8)
merged_population_3_8 = HouseholdPopulation(
    merged_comp_list_3_8,
    merged_comp_dist_3_8,
    merged_input_3,
    within_household_SEIR,
    4,
    True)
print('3 hh, max 8 construction took', time() - start_3_8, 'seconds.')
print('System size is', sum(merged_population_3_8.system_sizes),'.')

start_3_12 = time()
merged_comp_list_3_12, merged_comp_dist_3_12, hh_dimension_3_12, pairings_3_12 = \
    build_mixed_compositions(composition_list, comp_dist, 3, 12)
merged_population_3_12 = HouseholdPopulation(
    merged_comp_list_3_12,
    merged_comp_dist_3_12,
    merged_input_3,
    within_household_SEIR,
    4,
    True)
print('3 hh, max 12 construction took', time() - start_3_12, 'seconds.')
print('System size is', sum(merged_population_3_12.system_sizes),'.')
