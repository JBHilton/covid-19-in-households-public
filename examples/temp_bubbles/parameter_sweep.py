from argparse import ArgumentParser
from numpy import arange, array, atleast_2d, hstack
from os.path import isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from time import time
from model.preprocessing import (
        SEIRInput, HouseholdPopulation, make_initial_condition_with_recovereds)
from model.specs import SINGLE_AGE_UK_SPEC, SINGLE_AGE_SEIR_SPEC
from model.common import SEIRRateEquations
from model.imports import NoImportModel
from examples.temp_bubbles.common import (
        DataObject,
        MergedSEIRInput,
        demerged_initial_condition,
        build_mixed_compositions_pairwise,
        pairwise_merged_initial_condition,
        pairwise_demerged_initial_condition,
        match_merged_states_to_unmerged,
        build_mixed_compositions_threewise,
        initialise_merged_system_threewise)

SPEC = {**SINGLE_AGE_UK_SPEC, **SINGLE_AGE_SEIR_SPEC}

# NUMERICAL SETTINGS
ATOL = 1e-16

comp_dist = read_csv(
    'inputs/england_hh_size_dist.csv',
    header=0).to_numpy().squeeze()
comp_dist = comp_dist[:4]
comp_dist = comp_dist/sum(comp_dist)
max_hh_size = len(comp_dist)
composition_list = atleast_2d(arange(1, max_hh_size+1)).T

ave_hh_size = comp_dist.dot(composition_list)

# Number of households we merge
hh_to_merge = 3
# This is left over from an earlier formulation, just keep it at 1
mixing_strength = 1

unmerged_expo_range = array([0.0, 0.5, 1.0])
merged_expo_range = array([0.0, 0.5, 1.0])

unmerged_expo_len = len(unmerged_expo_range)
merged_expo_len = len(merged_expo_range)


def simulate_merge2(
        unmerged_population,
        merged_population,
        rhs_unmerged,
        rhs_merged,
        hh_dimension,
        pairings,
        duration_list,
        switches,
        premerge_H0,
        t0,
        t_end):
    for switch in range(switches+1):
        # print('Initialising merge number',switch,'...')
        H0 = pairwise_merged_initial_condition(
            premerge_H0,
            unmerged_population,
            merged_population,
            hh_dimension,
            pairings,
            4)
        tspan = (t0, t0 + duration_list[switch])
        # print('Integrating over merge period number',switch,'...')
        solution = solve_ivp(
            rhs_merged,
            tspan,
            H0/sum(H0),
            first_step=0.001,
            atol=ATOL)
        # print('Integration over merge period took',time()-t_mer,'seconds.')
        temp_time = solution.t
        temp_H = solution.y
        t0 = t0 + duration_list[switch]
        premerge_H0 = pairwise_demerged_initial_condition(
            temp_H[:, -1],
            unmerged_population,
            merged_population,
            hh_dimension,
            pairings,
            4)
        if switch == 0:
            merge_time = temp_time
            merge_H = temp_H
        else:
            merge_time = hstack((merge_time, temp_time))
            merge_H = hstack((merge_H, temp_H))

    # print('Initialising post-merge period...')
    H0 = pairwise_demerged_initial_condition(
        merge_H[:, -1],
        unmerged_population,
        merged_population,
        hh_dimension,
        pairings,
        4)
    tspan = (t0, t_end)
    # print('Integrating over post-merge period...')
    solution = solve_ivp(
        rhs_unmerged,
        tspan,
        H0,
        first_step=0.001,
        atol=ATOL)
    # print('Integration over post-merge period took',time()-t_post,'seconds.')
    postmerge_time = solution.t
    postmerge_H = solution.y

    return merge_time, merge_H, postmerge_time, postmerge_H


def simulate_merge3(
        unmerged_population,
        merged_population,
        rhs_unmerged,
        rhs_merged,
        hh_dimension,
        pairings,
        state_match,
        duration_list,
        switches,
        premerge_H0,
        t0,
        t_end):
    for switch in range(switches+1):
        # print('Initialising merge number',switch,'...')
        H0 = initialise_merged_system_threewise(
            premerge_H0,
            unmerged_population,
            merged_population,
            state_match)
        tspan = (t0, t0 + duration_list[switch])
        # print('Integrating over merge period number',switch,'...')
        # t_mer = time()
        solution = solve_ivp(
            rhs_merged,
            tspan,
            H0/sum(H0),
            first_step=0.001,
            atol=ATOL)
        # print('Integration over merge period took',time()-t_mer,'seconds.')
        temp_time = solution.t
        temp_H = solution.y
        t0 = t0 + duration_list[switch]
        premerge_H0 = demerged_initial_condition(
            temp_H[:, -1],
            unmerged_population,
            merged_population,
            hh_dimension,
            pairings,
            hh_to_merge,
            4)
        if switch == 0:
            merge_time = temp_time
            merge_H = temp_H
        else:
            merge_time = hstack((
                merge_time,
                temp_time))
            merge_H = hstack((merge_H, temp_H))

    # print('Initialising post-merge period...')
    H0 = demerged_initial_condition(
        merge_H[:, -1],
        unmerged_population,
        merged_population,
        hh_dimension,
        pairings,
        hh_to_merge,
        4)
    tspan = (t0, t_end)
    # print('Integrating over post-merge period...')
    # t_post = time()
    solution = solve_ivp(
        rhs_unmerged,
        tspan,
        H0,
        first_step=0.001,
        atol=ATOL)
    # print('Integration over post-merge period took',time()-t_post,'seconds.')
    postmerge_time = solution.t
    postmerge_H = solution.y

    return merge_time, merge_H, postmerge_time, postmerge_H


def main(no_of_workers):
    if isfile('threewise_merge_comps.pkl') is True:
        with open('threewise_merge_comps.pkl', 'rb') as f:
            print('Loading merged composition distribution...')
            merged_comp_list_2, \
                merged_comp_dist_2, \
                pairings_2, \
                merged_comp_list_3, \
                merged_comp_dist_3, \
                hh_dimension, \
                pairings_3 = load(f)
    else:
        print('Building merged composition distribution...')
        merged_comp_list_2, \
            merged_comp_dist_2, \
            hh_dimension, \
            pairings_2 = build_mixed_compositions_pairwise(
                composition_list, comp_dist)
        merged_comp_list_3, \
            merged_comp_dist_3, \
            hh_dimension, \
            pairings_3 = build_mixed_compositions_threewise(
                composition_list, comp_dist, 10)

        with open('threewise_merge_comps.pkl', 'wb') as f:
            dump(
                (
                    merged_comp_list_2,
                    merged_comp_dist_2,
                    pairings_2,
                    merged_comp_list_3,
                    merged_comp_dist_3,
                    hh_dimension,
                    pairings_3
                ),
                f)
    # December 1st
    t0 = 335.0
    # December 25th
    merge_start = 359.0
    # Continue running projections to end of Jan
    t_end = 396

    for i in range(unmerged_expo_len):
        filename_stem = 'sweep_results_' + str(i)
        unmerged_input = SEIRInput(SPEC, composition_list, comp_dist)
        unmerged_input.density_expo = unmerged_expo_range[i]
        unmerged_input.k_home = (
            ave_hh_size ** unmerged_expo_range[i]) * unmerged_input.k_home

        unmerged_population = HouseholdPopulation(
           composition_list,
           comp_dist,
           unmerged_input,
           True)

        rhs_unmerged = SEIRRateEquations(
            unmerged_input,
            unmerged_population,
            NoImportModel(4, 1))

        H0 = make_initial_condition_with_recovereds(unmerged_population, rhs_unmerged)
        tspan = (t0, merge_start)
        solution = solve_ivp(
            rhs_unmerged,
            tspan,
            H0,
            first_step=0.001,
            atol=ATOL)
        baseline_time = solution.t
        baseline_H = solution.y

        baseline_premerge_H0 = baseline_H[:, -1]

        tspan = (merge_start, t_end)
        solution = solve_ivp(
            rhs_unmerged,
            tspan,
            baseline_H[:, -1],
            first_step=0.001,
            atol=ATOL)
        baseline_time = hstack((baseline_time, solution.t))
        baseline_H = hstack((baseline_H, solution.y))

        baseline_S = baseline_H.T.dot(
            unmerged_population.states[:, 0])/ave_hh_size
        baseline_E = baseline_H.T.dot(
            unmerged_population.states[:, 1])/ave_hh_size
        baseline_I = baseline_H.T.dot(
            unmerged_population.states[:, 2])/ave_hh_size
        baseline_R = baseline_H.T.dot(
            unmerged_population.states[:, 3])/ave_hh_size

        with open(filename_stem + '.pkl', 'wb') as f:
            dump(
                (
                    unmerged_population,
                    baseline_H,
                    baseline_time,
                    baseline_S,
                    baseline_E,
                    baseline_I,
                    baseline_R
                ),
                f)

        for j in range(merged_expo_len):
            print('i={0} j={1}.'.format(i, j))
            print('Pre-merge density exponent is', unmerged_expo_range[i], '.')
            print('Merged density exponent is', merged_expo_range[j], '.')

            this_iteration_start = time()

            filename = filename_stem + str(j)

            merged_input2 = MergedSEIRInput(
                            SPEC, composition_list, comp_dist, 2, 1)
            merged_input2.density_expo = merged_expo_range[j]
            # merged_input2.k_home = (
            # ave_hh_size ** unmerged_expo_range[i]) * merged_input2.k_home
            merged_population2 = HouseholdPopulation(
                merged_comp_list_2,
                merged_comp_dist_2,
                merged_input2,
                True)
            rhs_merged2 = SEIRRateEquations(
                merged_input2,
                merged_population2,
                NoImportModel(4, 2))

            merged_input3 = MergedSEIRInput(
                SPEC, composition_list, comp_dist, 3, 1)
            merged_input3.density_expo = merged_expo_range[j]
            # merged_input3.k_home = (ave_hh_size ** unmerged_expo_range[i])
            # * merged_input3.k_home
            merged_population3 = HouseholdPopulation(
              merged_comp_list_3,
              merged_comp_dist_3,
              merged_input3,
              True)
            # We only need to generate this thing once, it is very time
            # consuming
            if i == 0 and j == 0:
                state_match = match_merged_states_to_unmerged(
                    unmerged_population,
                    merged_population3,
                    pairings_3,
                    hh_to_merge,
                    4)

            rhs_merged3 = SEIRRateEquations(
                merged_input3,
                merged_population3,
                NoImportModel(4, 3))
            # This is just a class I made up to store the results - very hacky!
            merge_results = DataObject(0)

            # STRATEGIES: 1 is form triangles for 2 days, 2 is form pair on
            # 25th and again on 26th, 3 is 1 plus pair on new year's, 4 is 2
            # plus pair on new year's

            merge_results.t_merge_1, \
                merge_results.H_merge_1, \
                merge_results.t_postmerge_1, \
                merge_results.H_postmerge_1 = \
                simulate_merge3(
                    unmerged_population,
                    merged_population3,
                    rhs_unmerged,
                    rhs_merged3,
                    hh_dimension,
                    pairings_3,
                    state_match,
                    [2],
                    0,
                    baseline_premerge_H0,
                    merge_start,
                    365)
            merge_results.t_merge_3, merge_results.H_merge_3, merge_results.t_postmerge_3, merge_results.H_postmerge_3 = \
                simulate_merge2(
                    unmerged_population,
                    merged_population2,
                    rhs_unmerged,
                    rhs_merged2,
                    hh_dimension,
                    pairings_2,
                    [1],
                    0,
                    merge_results.H_postmerge_1[:, -1],
                    365,
                    t_end)

            solution = solve_ivp(rhs_unmerged, (365, t_end), merge_results.H_postmerge_1[:,-1], first_step=0.001, atol=1e-16)
            merge_results.t_postmerge_1 = hstack(
                (merge_results.t_postmerge_1, solution.t))
            merge_results.H_postmerge_1 = hstack(
                (merge_results.H_postmerge_1, solution.y))

            merge_results.t_merge_2, merge_results.H_merge_2, merge_results.t_postmerge_2, merge_results.H_postmerge_2 = \
                simulate_merge2(
                    unmerged_population,
                    merged_population2,
                    rhs_unmerged,
                    rhs_merged2,
                    hh_dimension,
                    pairings_2,
                    [1, 1],
                    1,
                    baseline_premerge_H0,
                    merge_start,
                    365)
            merge_results.t_merge_4, merge_results.H_merge_4, merge_results.t_postmerge_4, merge_results.H_postmerge_4 = \
                simulate_merge2(
                    unmerged_population,
                    merged_population2,
                    rhs_unmerged,
                    rhs_merged2,
                    hh_dimension,
                    pairings_2,
                    [1],
                    0,
                    merge_results.H_postmerge_2[:,-1],
                    365,
                    t_end)

            solution = solve_ivp(rhs_unmerged, (365, t_end), merge_results.H_postmerge_2[:,-1], first_step=0.001, atol=1e-16)
            merge_results.t_postmerge_2 = hstack((merge_results.t_postmerge_2,solution.t))
            merge_results.H_postmerge_2 = hstack((merge_results.H_postmerge_2,solution.y))

            with open(filename + '.pkl', 'wb') as f:
                dump(
                    (
                        merged_population2,
                        merged_population3,
                        merge_results
                    ),
                    f)

            print('Iteration took {0} seconds'.format(
                time() - this_iteration_start))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=2)
    args = parser.parse_args()
    main(args.no_of_workers)
