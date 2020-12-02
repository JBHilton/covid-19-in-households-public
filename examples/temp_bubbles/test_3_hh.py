'''This runs the UK-like model with a single set of parameters for 100 days
'''
from numpy import absolute, arange, array, atleast_2d, hstack, true_divide, vstack, where, zeros
from os.path import isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from time import time
from model.common import my_int
from model.preprocessing import (
        ModelInput, HouseholdPopulation)
from model.specs import DEFAULT_SPEC
from examples.temp_bubbles.common import ( DataObject,
        build_mixed_compositions, SINGLE_AGE_CLASS_SEIR_SPEC, SingleClassSEIRInput,
        MergedSEIRInput, merged_initial_condition, merged_initial_condition_alt, demerged_initial_condition,
        build_mixed_compositions_pairwise, pairwise_merged_initial_condition,
        pairwise_demerged_initial_condition,
        make_initial_condition_with_recovereds, within_household_SEPIR,RateEquations,
        within_household_SEIR, SEIRRateEquations)
# pylint: disable=invalid-name

unmerged_input = SingleClassSEIRInput(SINGLE_AGE_CLASS_SEIR_SPEC)

hh_to_merge = 2
mixing_strength = 0.5
merged_input = MergedSEIRInput(SINGLE_AGE_CLASS_SEIR_SPEC,hh_to_merge,mixing_strength)

comp_dist = read_csv(
    'inputs/england_hh_size_dist.csv',
    header=0).to_numpy().squeeze()
comp_dist = comp_dist[:5]
comp_dist = comp_dist/sum(comp_dist)
max_hh_size = len(comp_dist)
composition_list = atleast_2d(arange(1,max_hh_size+1)).T

if isfile('bubble-vars-3-12.pkl') is True:
    with open('bubble-vars-3-12.pkl', 'rb') as f:
        unmerged_population, merged_population, hh_dimension, max_hh_size, pairings, mc_index_vector, mc_reverse_prod = load(f)
else:
    unmerged_population = HouseholdPopulation(
       composition_list,
       comp_dist,
       unmerged_input,
       within_household_SEIR,
       4,
       True)

    merged_comp_list, merged_comp_dist, hh_dimension, pairings, mc_index_vector, mc_reverse_prod = \
      build_mixed_compositions(composition_list, comp_dist, 3, 10)
    merged_population = HouseholdPopulation(
      merged_comp_list,
      merged_comp_dist,
      merged_input,
      within_household_SEIR,
      4,
      True)

    with open('bubble-vars-3-12.pkl', 'wb') as f:
        dump((unmerged_population,
             merged_population,
             hh_dimension,
             max_hh_size,
             pairings,
             mc_index_vector,
             mc_reverse_prod),
             f)

rhs_unmerged = SEIRRateEquations(unmerged_input, unmerged_population)
rhs_merged = SEIRRateEquations(merged_input, merged_population)

ave_hh_size = unmerged_population.composition_distribution.dot(atleast_2d(unmerged_population.composition_list))

def simulate_merge(duration_list,switches,premerge_H0,t0,t_end):
    for switch in range(switches+1):
        print('Initialising merge number',switch,'...')
        H0 = merged_initial_condition_alt(premerge_H0,
                                    unmerged_population,
                                    merged_population,
                                    hh_dimension,
                                    mc_index_vector,
                                    mc_reverse_prod,
                                    pairings,
                                    hh_to_merge,
                                    4)
        tspan = (t0, t0 + duration_list[switch])
        print('Integrating over merge period number',switch,'...')
        t_mer = time()
        solution = solve_ivp(rhs_merged, tspan, H0, first_step=0.001, atol=1e-16)
        print('Integration over merge period took',time()-t_mer,'seconds.')
        temp_time = solution.t
        temp_H = solution.y
        t0 = t0 + duration_list[switch]
        premerge_H0 = demerged_initial_condition(temp_H[:,-1],
                                                        unmerged_population,
                                                        merged_population,
                                                        hh_dimension,
                                                        pairings,
                                                        hh_to_merge,
                                                        4)
        if switch==0:
            merge_time = temp_time
            merge_H = temp_H
        else:
            merge_time = hstack((merge_time,temp_time))
            merge_H = hstack((merge_H,temp_H))

    print('Initialising post-merge period...')
    H0 = demerged_initial_condition(merge_H[:,-1],
                                    unmerged_population,
                                    merged_population,
                                    hh_dimension,
                                    pairings,
                                    hh_to_merge,
                                    4)
    tspan = (t0, t_end)
    print('Integrating over post-merge period...')
    t_post = time()
    solution = solve_ivp(rhs_unmerged, tspan, H0, first_step=0.001)
    print('Integration over post-merge period took',time()-t_post,'seconds.')
    postmerge_time = solution.t
    postmerge_H = solution.y

    return merge_time, merge_H, postmerge_time, postmerge_H



output_data = DataObject(0) # Define data as object with attributed

merge_start = 359.0
merge_duration = 5.0
postmerge_dur = 90

print('Initialising pre-merge period...')
H0 = make_initial_condition_with_recovereds(unmerged_population, rhs_unmerged)

print('Antibody prevalence on September 28th is',
    H0.T.dot(unmerged_population.states[:,3])/ave_hh_size)

rec_numbers = unmerged_population.states[:,3]
visited = where(rec_numbers>0)[0]

attack_rate = zeros((len(visited),))
for i in range(len(visited)):
    attack_rate[i] = rec_numbers[visited[i]] / composition_list[unmerged_population.which_composition[visited[i]]]


print('Household attack rate by September 28th is',
    H0[visited].T.dot(attack_rate) / sum(H0[visited]))
print('Proportion of households visited is',sum(H0[visited]))

output_data.call_prev = H0.T.dot(unmerged_population.states[:,3])/ave_hh_size

tspan = (271.0, merge_start)
solution = solve_ivp(rhs_unmerged, tspan, H0, first_step=0.001, atol=1e-16)
baseline_time = solution.t
baseline_H = solution.y

output_data.premerge_time = solution.t
output_data.premerge_H = solution.y

tspan = (merge_start, merge_start + merge_duration + postmerge_dur)
solution = solve_ivp(rhs_unmerged, tspan, baseline_H[:,-1], first_step=0.001)
output_data.baseline_time = hstack((baseline_time,solution.t))
output_data.baseline_H = hstack((baseline_H,solution.y))

# print('Initialising merge period...')
# H0 = pairwise_merged_initial_condition(premerge_H[:,-1],
#                             unmerged_population,
#                             merged_population,
#                             hh_dimension,
#                             pairings)
# print(sum(H0))
# tspan = (merge_start, merge_start + merge_duration)
# print('Integrating over merge period...')
# t_mer = time()
# solution = solve_ivp(rhs_merged, tspan, H0/sum(H0), first_step=0.001)
# print('Integration over merge period took',time()-t_mer,'seconds.')
# merge_time = solution.t
# merge_H = solution.y

# H0_test = merged_initial_condition(output_data.premerge_H[:,-1],
#                             unmerged_population,
#                             merged_population,
#                             hh_dimension,
#                             pairings,
#                             hh_to_merge,
#                             4)
H0_test_alt = merged_initial_condition_alt(output_data.premerge_H[:,-1],
                            unmerged_population,
                            merged_population,
                            hh_dimension,
                            mc_index_vector,
                            mc_reverse_prod,
                            pairings,
                            hh_to_merge,
                            4)
print('sum(H0_test_alt)=',sum(H0_test_alt))

output_data.merge_t_5day, output_data.merge_H_5day, output_data.postmerge_t_5day, output_data.postmerge_H_5day = \
    simulate_merge([5],0,output_data.premerge_H[:,-1],merge_start, 455)

output_data.merge_S_5day = (1/hh_to_merge) * output_data.merge_H_5day.T.dot(
    merged_population.states[:, 0] + merged_population.states[:, 4] +
    merged_population.states[:, 8])/ave_hh_size
output_data.postmerge_S_5day = output_data.postmerge_H_5day.T.dot(unmerged_population.states[:, 0])/ave_hh_size
output_data.merge_E_5day = (1/hh_to_merge) * output_data.merge_H_5day.T.dot(
    merged_population.states[:, 1] + merged_population.states[:, 5] +
    merged_population.states[:, 9])/ave_hh_size
output_data.postmerge_E_5day = output_data.postmerge_H_5day.T.dot(unmerged_population.states[:, 1])/ave_hh_size
output_data.merge_I_5day = (1/hh_to_merge) * output_data.merge_H_5day.T.dot(
    merged_population.states[:, 2] + merged_population.states[:, 6] +
    merged_population.states[:, 10])/ave_hh_size
output_data.postmerge_I_5day = output_data.postmerge_H_5day.T.dot(unmerged_population.states[:, 2])/ave_hh_size
output_data.merge_R_5day = (1/hh_to_merge) * output_data.merge_H_5day.T.dot(
    merged_population.states[:, 3] + merged_population.states[:, 7] +
    merged_population.states[:,11])/ave_hh_size
output_data.postmerge_R_5day = output_data.postmerge_H_5day.T.dot(unmerged_population.states[:, 3])/ave_hh_size

merge_t_switching = []
merge_H_switching = []
postmerge_t_switching = []
postmerge_H_switching = []

merge_S_switching = []
merge_E_switching = []
merge_I_switching = []
merge_R_switching = []
postmerge_S_switching = []
postmerge_E_switching = []
postmerge_P_switching = []
postmerge_I_switching = []
postmerge_R_switching = []

for days in range(1):
    merge_t_temp, merge_H_temp, postmerge_t_temp, postmerge_H_temp = \
        simulate_merge((days+1)*[1],days,output_data.premerge_H[:,-1],merge_start, 455)
    merge_t_switching.append(merge_t_temp)
    merge_H_switching.append(merge_H_temp)
    postmerge_t_switching.append(postmerge_t_temp)
    postmerge_H_switching.append(postmerge_H_temp)

    merge_S_switching.append((1/hh_to_merge) * merge_H_temp.T.dot(
        merged_population.states[:, 0] + merged_population.states[:, 4] +
        merged_population.states[:, 8])/ave_hh_size)
    postmerge_S_switching.append(postmerge_H_temp.T.dot(unmerged_population.states[:, 0])/ave_hh_size)
    merge_E_switching.append((1/hh_to_merge) * merge_H_temp.T.dot(
        merged_population.states[:, 1] + merged_population.states[:, 5] +
        merged_population.states[:, 9])/ave_hh_size)
    postmerge_E_switching.append(postmerge_H_temp.T.dot(unmerged_population.states[:, 1])/ave_hh_size)
    merge_I_switching.append((1/hh_to_merge) * merge_H_temp.T.dot(
        merged_population.states[:, 2] + merged_population.states[:, 6] +
        merged_population.states[:, 10])/ave_hh_size)
    postmerge_I_switching.append(postmerge_H_temp.T.dot(unmerged_population.states[:, 2])/ave_hh_size)
    merge_R_switching.append((1/hh_to_merge) * merge_H_temp.T.dot(
        merged_population.states[:, 3] + merged_population.states[:, 7] +
        merged_population.states[:, 11])/ave_hh_size)
    postmerge_R_switching.append(postmerge_H_temp.T.dot(unmerged_population.states[:, 3])/ave_hh_size)

output_data.merge_t_switching = merge_t_switching
output_data.merge_H_switching = merge_H_switching
output_data.postmerge_t_switching = postmerge_t_switching
output_data.postmerge_H_switching = postmerge_H_switching
output_data.merge_S_switching = merge_S_switching
output_data.merge_E_switching = merge_E_switching
output_data.merge_I_switching = merge_I_switching
output_data.merge_R_switching = merge_R_switching
output_data.postmerge_S_switching = postmerge_S_switching
output_data.postmerge_E_switching = postmerge_E_switching
output_data.postmerge_I_switching = postmerge_I_switching
output_data.postmerge_R_switching = postmerge_R_switching

# print(absolute(merge_time-merge_time_alt).max())
# print(absolute(merge_H-merge_H_alt).max())
#
# print('Initialising post-merge period...')
# H0 = pairwise_demerged_initial_condition(merge_H[:,-1],
#                                 unmerged_population,
#                                 merged_population,
#                                 hh_dimension,
#                                 pairings)
# tspan = (merge_start + merge_duration, merge_start + merge_duration + postmerge_dur)
# print('Integrating over post-merge period...')
# t_post = time()
# solution = solve_ivp(rhs_unmerged, tspan, H0, first_step=0.001)
# print('Integration over post-merge period took',time()-t_post,'seconds.')
# postmerge_time = solution.t
# postmerge_H = solution.y
#
# print(absolute(postmerge_time-postmerge_time_alt).max())
# print(absolute(postmerge_H-postmerge_H_alt).max())

output_data.baseline_S = output_data.baseline_H.T.dot(unmerged_population.states[:,0])/ave_hh_size
output_data.baseline_E = output_data.baseline_H.T.dot(unmerged_population.states[:,1])/ave_hh_size
output_data.baseline_I = output_data.baseline_H.T.dot(unmerged_population.states[:,2])/ave_hh_size
output_data.baseline_R = output_data.baseline_H.T.dot(unmerged_population.states[:,3])/ave_hh_size

with open('bubble_results_3_12.pkl', 'wb') as f:
    dump(output_data,
     f)
