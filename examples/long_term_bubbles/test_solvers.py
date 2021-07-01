'''This is a minimal example we can use to test the solvers for the growth rate
estimation.'''
from numpy import diag, log, where, zeros
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from random import uniform
from scipy.integrate import solve_ivp
from scipy.sparse import identity as spidentity
from scipy.sparse.linalg import spsolve
from time import time as get_time
from model.preprocessing import ( build_support_bubbles, SEPIRInput, HouseholdPopulation,
                    make_initial_condition)
from model.specs import TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations, sparse
from model.imports import NoImportModel
# pylint: disable=invalid-name

MAX_ADULTS = 1 # In this example we assume only single-adult households can join bubbles
MAX_BUBBLE_SIZE = 8
SPEC = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
DOUBLING_TIME = 3
X0 = log(2) / DOUBLING_TIME

# List of observed household compositions
composition_list = read_csv(
    'inputs/eng_and_wales_adult_child_composition_list.csv',
    header=0).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/eng_and_wales_adult_child_composition_dist.csv',
    header=0).to_numpy().squeeze()

basic_mixed_comp_list, basic_mixed_comp_dist = build_support_bubbles(
        composition_list,
        comp_dist,
        MAX_ADULTS,
        MAX_BUBBLE_SIZE,
        1)

basic_mixed_comp_dist = basic_mixed_comp_dist/sum(basic_mixed_comp_dist)
basic_bubbled_input = SEPIRInput(SPEC, basic_mixed_comp_list, basic_mixed_comp_dist)

bubbled_household_population = HouseholdPopulation(
    basic_mixed_comp_list, basic_mixed_comp_dist, basic_bubbled_input)

rhs = SEPIRRateEquations(basic_bubbled_input, bubbled_household_population, NoImportModel(5,2))

r = X0 * (0.5 + uniform(0,1)) # Random r value close to true value

reverse_comp_dist = diag(bubbled_household_population.composition_distribution).dot(bubbled_household_population.composition_list)
reverse_comp_dist = reverse_comp_dist.dot(diag(1/reverse_comp_dist.sum(0)))

Q_int = rhs.Q_int
FOI_by_state = zeros((Q_int.shape[0],bubbled_household_population.no_risk_groups))
for ic in range(rhs.no_inf_compartments):
            states_inf_only =  rhs.inf_by_state_list[ic]
            FOI_by_state += (rhs.ext_matrix_list[ic].dot(
                    rhs.epsilon * states_inf_only.T)).T
index_states = where(
((rhs.states_exp_only.sum(axis=1)==1) *
((rhs.states_pro_only + rhs.states_inf_only + rhs.states_rec_only).sum(axis=1)==0)))[0]

no_index_states = len(index_states)
comp_by_index_state = bubbled_household_population.which_composition[index_states]

index_prob = zeros((bubbled_household_population.no_risk_groups,no_index_states))
for i in range(no_index_states):
    index_class = where(rhs.states_exp_only[index_states[i],:]==1)[0]
    index_prob[index_class,i] = reverse_comp_dist[comp_by_index_state[i], index_class]

multiplier = sparse((no_index_states, no_index_states))
discount_matrix = r * spidentity(Q_int.shape[0]) - Q_int
reward_mat = FOI_by_state.dot(index_prob)
mult_start = get_time()
for i, index_state in enumerate(index_states):
    col = spsolve(discount_matrix, reward_mat[:, i])
    multiplier += sparse((col[index_states], (range(no_index_states), no_index_states * [i] )), shape=(no_index_states, no_index_states))

print('Calculations took',get_time()-mult_start,'seconds.')
