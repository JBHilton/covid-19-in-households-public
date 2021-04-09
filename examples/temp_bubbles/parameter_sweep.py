from argparse import ArgumentParser
from numpy import arange, array, atleast_2d, hstack
from os.path import isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from time import time
from multiprocessing import Pool
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


def create_unmerged_context(p):
    return UnmergedContext(*p)


class UnmergedContext:
    def __init__(self, i, density_exponent, t_start, t_end, merge_start):
        unmerged_input = SEIRInput(SPEC, composition_list, comp_dist)
        unmerged_input.density_expo = density_exponent
        unmerged_input.k_home = (
            ave_hh_size ** density_exponent) * unmerged_input.k_home

        self.population = HouseholdPopulation(
           composition_list,
           comp_dist,
           unmerged_input,
           True)

        self.rhs = SEIRRateEquations(
            unmerged_input,
            self.population,
            NoImportModel(4, 1))

        self.H0 = make_initial_condition_with_recovereds(
            self.population, self.rhs)
        self.tspan = (t_start, merge_start)
        solution = solve_ivp(
            self.rhs,
            self.tspan,
            self.H0,
            first_step=0.001,
            atol=ATOL)
        baseline_time = solution.t
        self.baseline_H = solution.y

        self.baseline_H0 = self.baseline_H[:, -1]

        tspan = (merge_start, t_end)
        solution = solve_ivp(
            self.rhs,
            tspan,
            self.baseline_H[:, -1],
            first_step=0.001,
            atol=ATOL)
        baseline_time = hstack((baseline_time, solution.t))
        baseline_H = hstack((self.baseline_H, solution.y))

        baseline_S = self.baseline_H.T.dot(
            self.population.states[:, 0])/ave_hh_size
        baseline_E = self.baseline_H.T.dot(
            self.population.states[:, 1])/ave_hh_size
        baseline_I = self.baseline_H.T.dot(
            self.population.states[:, 2])/ave_hh_size
        baseline_R = self.baseline_H.T.dot(
            self.population.states[:, 3])/ave_hh_size

        self.filename_stem = 'sweep_results_' + str(i)
        with open(self.filename_stem + '.pkl', 'wb') as f:
            dump(
                (
                    self.population,
                    baseline_H,
                    baseline_time,
                    baseline_S,
                    baseline_E,
                    baseline_I,
                    baseline_R
                ),
                f)


def unpack_paramas_and_run_merge(p):
    '''This function enables multi-argument parallel run.'''
    return run_merge(*p)


def run_merge(
        i,
        unmerged_exp,
        j,
        merged_exp,
        merged_comp_list_2,
        merged_comp_dist_2,
        merged_comp_list_3,
        merged_comp_dist_3,
        pairings_2,
        pairings_3,
        hh_dimension,
        unmerged,
        t_start,
        t_end,
        merge_start,
        merge_end
        ):
    '''This function runs a merge simulation for specified parameters.'''
    print('i={0} j={1}.'.format(i, j))
    print('Pre-merge density exponent is', unmerged_exp, '.')
    print('Merged density exponent is', merged_exp, '.')

    this_iteration_start = time()

    merged_input2 = MergedSEIRInput(
                    SPEC, composition_list, comp_dist, 2, 1)
    merged_input2.density_expo = merged_exp
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
    merged_input3.density_expo = merged_exp
    # merged_input3.k_home = (ave_hh_size ** unmerged_expo_range[i])
    # * merged_input3.k_home
    merged_population3 = HouseholdPopulation(
        merged_comp_list_3,
        merged_comp_dist_3,
        merged_input3,
        True)
    # We only need to generate this thing once, it is very time
    # consuming
    state_match = match_merged_states_to_unmerged(
        unmerged.population,
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
            unmerged.population,
            merged_population3,
            unmerged.rhs,
            rhs_merged3,
            hh_dimension,
            pairings_3,
            state_match,
            [2],
            0,
            unmerged.baseline_H0,
            merge_start,
            merge_end)
    merge_results.t_merge_3, \
        merge_results.H_merge_3, \
        merge_results.t_postmerge_3, \
        merge_results.H_postmerge_3 = \
        simulate_merge2(
            unmerged.population,
            merged_population2,
            unmerged.rhs,
            rhs_merged2,
            hh_dimension,
            pairings_2,
            [1],
            0,
            merge_results.H_postmerge_1[:, -1],
            merge_end,
            t_end)

    solution = solve_ivp(
        unmerged.rhs,
        (merge_end, t_end),
        merge_results.H_postmerge_1[:, -1],
        first_step=0.001,
        atol=ATOL)
    merge_results.t_postmerge_1 = hstack(
        (merge_results.t_postmerge_1, solution.t))
    merge_results.H_postmerge_1 = hstack(
        (merge_results.H_postmerge_1, solution.y))

    merge_results.t_merge_2, \
        merge_results.H_merge_2, \
        merge_results.t_postmerge_2, \
        merge_results.H_postmerge_2 = \
        simulate_merge2(
            unmerged.population,
            merged_population2,
            unmerged.rhs,
            rhs_merged2,
            hh_dimension,
            pairings_2,
            [1, 1],
            1,
            unmerged.baseline_H0,
            merge_start,
            merge_end)
    merge_results.t_merge_4, \
        merge_results.H_merge_4, \
        merge_results.t_postmerge_4, \
        merge_results.H_postmerge_4 = \
        simulate_merge2(
            unmerged.population,
            merged_population2,
            unmerged.rhs,
            rhs_merged2,
            hh_dimension,
            pairings_2,
            [1],
            0,
            merge_results.H_postmerge_2[:, -1],
            merge_end,
            t_end)

    solution = solve_ivp(
        unmerged.rhs,
        (merge_end, t_end),
        merge_results.H_postmerge_2[:, -1],
        first_step=0.001,
        atol=ATOL)
    merge_results.t_postmerge_2 = hstack((
        merge_results.t_postmerge_2, solution.t))
    merge_results.H_postmerge_2 = hstack((
        merge_results.H_postmerge_2, solution.y))

    filename = unmerged.filename_stem + str(j)
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

unmerged_exponents = array([0.0, 0.5, 1.0])
merged_exponents = array([0.0, 0.5, 1.0])


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
    merge_end = 365.0
    # Continue running projections to end of Jan
    t_end = 396

    params = []
    for i, e in enumerate(unmerged_exponents):
        params.append((i, e, t0, t_end, merge_start))
    with Pool(no_of_workers) as pool:
        unmerged_results = pool.map(create_unmerged_context, params)
    params = []
    for i, ei in enumerate(unmerged_exponents):
        for j, ej in enumerate(merged_exponents):
            params.append((
                i, ei, j, ej,
                merged_comp_list_2, merged_comp_dist_2,
                merged_comp_list_3, merged_comp_dist_3,
                pairings_2, pairings_3,
                hh_dimension,
                unmerged_results[i],
                t0, t_end, merge_start, merge_end))
    with Pool(no_of_workers) as pool:
        pool.map(unpack_paramas_and_run_merge, params)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=2)
    args = parser.parse_args()
    main(args.no_of_workers)
