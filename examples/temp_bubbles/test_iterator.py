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
        build_mixed_compositions, build_general_mixed_compositions, SINGLE_AGE_CLASS_SPEC, SingleClassInput,
        MergedInput,merged_initial_condition, demerged_initial_condition,
        make_initial_condition, within_household_SEPIR,RateEquations)

comp_dist = read_csv(
    'inputs/england_hh_size_dist.csv',
    header=0).to_numpy().squeeze()
comp_dist = comp_dist[:5]
comp_dist = comp_dist/sum(comp_dist)
max_hh_size = len(comp_dist)
composition_list = atleast_2d(arange(1,max_hh_size+1)).T

merged_comp_list, merged_comp_dist, hh_dimension, pairings = \
 build_mixed_compositions(composition_list, comp_dist)

merged_comp_list_gen, merged_comp_dist_gen, hh_dimension_gen, pairings_gen = \
  build_general_mixed_compositions(composition_list, comp_dist,2)

print(merged_comp_list)
print(merged_comp_list_gen)

print(merged_comp_dist)
print(merged_comp_dist_gen)

print(merged_comp_dist - merged_comp_dist_gen)
