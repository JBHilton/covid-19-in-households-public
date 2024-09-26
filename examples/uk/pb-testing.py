from copy import deepcopy
from matplotlib.pyplot import semilogx, subplots
from numpy import arange, array, log, where, zeros
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
import time
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEPIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations, MatrixImportSEPIRRateEquations, UnloopedSEPIRRateEquations
from model.imports import FixedImportModel, NoImportModel, ExponentialImportModel

# pylint: disable=invalid-name

if isdir('outputs/uk') is False:
    mkdir('outputs/uk')

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


SPEC = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
model_input_to_fit = SEPIRInput(SPEC, composition_list, comp_dist)
household_population_to_fit = HouseholdPopulation(
    composition_list, comp_dist, model_input_to_fit)
rhs_to_fit = SEPIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(5,2))
beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
model_input = deepcopy(model_input_to_fit)
model_input.k_ext *= beta_ext
print('Estimated beta is',beta_ext)

# With the parameters chosen, we calculate Q_int:
household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

no_imports = NoImportModel(5, 2)

# Set up background population for models with nonzero imports
base_rhs = SEPIRRateEquations(model_input, household_population, no_imports)
base_H0 = make_initial_condition_by_eigenvector(growth_rate,
                                                model_input,
                                                household_population,
                                                base_rhs,
                                                1e-3,
                                                0.0,
                                                False,
                                                3)
x0 = base_H0.T.dot(household_population.states[:, 1::5])

exp_imports = ExponentialImportModel(5,
                                     2,
                                     base_rhs,
                                     growth_rate,
                                     x0)

rhs_U = UnloopedSEPIRRateEquations(model_input, household_population, exp_imports, sources="IMPORT")

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs_U, 0., 0.0,False,3)
tspan = (0.0, 365)
u_start = time.time()
solution_U = solve_ivp(rhs_U, tspan, H0, first_step=0.001, atol=1e-16, t_eval = arange(0., 365., 1.))
print("U took", time.time() - u_start)

H_U = solution_U.y
R_U = H_U.T.dot(household_population.states[:, 4::5])

import numpy as np
import scipy.sparse as sp
import math

## h is time step; X0 is initial value
def main_(r, t_start, t_final, A1, A2, X0, h=1):
    start_time = time.time()

    matrix_size = A1.shape[0]
    I = sp.eye(matrix_size, format='csr')  # Sparse identity matrix

    # Precompute factorial values and sparse matrix powers for efficiency
    factorials = np.array([math.factorial(i) for i in range(5)])

    A1_powers = [sp.csr_matrix(I)]
    A2_powers = [sp.csr_matrix(I)]

    for i in range(1, 5):
        A1_powers.append(A1_powers[-1].dot(A1))  # A1^i
        A2_powers.append(A2_powers[-1].dot(A2))  # A2^i

    t_powers = np.array([h ** i for i in range(5)])  ## (t(i+1)-t(i))^i

    r_powers = np.array([(1 / r) ** i for i in range(5)])

    exp_rh = np.exp(r * h)
    exp_2rh = np.exp(2 * r * h)
    exp_3rh = np.exp(3 * r * h)

    cons_9 = (exp_rh - r * h - 1) * r_powers[2]
    cons_10 = (exp_rh * (r * h - 1) + 1) * r_powers[2]
    cons_11 = (2 * exp_rh - (r ** 2) * (h ** 2) - 2 * r * h - 2) / 2
    cons_12 = exp_rh * (r * h - 2) + r * h + 2
    cons_13 = ((exp_rh - 2) ** 2 + 2 * r * h - 1) / 4
    cons_14 = (exp_rh * ((r ** 2) * (h ** 2) - 2 * r * h + 2) - 2) / 2
    cons_15 = exp_2rh / (2 * r) - exp_rh * h - 1 / (2 * r)
    cons_16 = (exp_2rh * (2 * r * h - 3) + 4 * exp_rh - 1) / 4

    cons_17 = (6 * exp_rh - 6 - (r ** 3) * (h ** 3) - 3 * (r ** 2) * (h ** 2) - 6 * r * h) / 6
    cons_18 = exp_rh * (r * h - 3) + (h ** 2) * (r ** 2) / 2 + 2 * r * h + 3
    cons_19 = ((exp_rh - 4) ** 2 + 2 * (h ** 2) * (r ** 2) + 6 * r * h - 9) / 8
    cons_20 = exp_rh * ((r ** 2) * (h ** 2) - 4 * r * h + 6) / 2 - (h * r + 3)
    cons_21 = exp_2rh / 4 + exp_rh * (1 - r * h) - (2 * r * h + 5) / 4
    cons_22 = (exp_2rh * (r * h - 2) + 4 * exp_rh - r * h - 2) / 4
    cons_23 = exp_3rh / 18 - exp_2rh / 4 + exp_rh / 2 - (6 * r * h + 11) / 36
    cons_24 = 1 + exp_rh * ((r ** 3) * (h ** 3) - 3 * (r ** 2) * (h ** 2) + 6 * h * r - 6) / 6
    cons_25 = (((exp_rh - 1) ** 2) - exp_rh * (h ** 2) * (r ** 2)) / 2
    cons_26 = exp_2rh * (2 * r * h - 5) / 4 + exp_rh * (1 + r * h) + 1 / 4
    cons_27 = exp_3rh / 12 - exp_2rh / 2 + exp_rh * (1 + 2 * r * h) / 4 + 1 / 6
    cons_28 = exp_2rh * (2 * (r ** 2) * (h ** 2) - 6 * h * r + 7) / 8 - exp_rh + 1 / 8
    cons_29 = exp_3rh / 6 + exp_2rh * (1 - 2 * r * h) / 4 - exp_rh / 2 + 1 / 12
    cons_30 = exp_3rh * (6 * r * h - 11) / 36 + exp_2rh / 2 - exp_rh / 4 + 1 / 18

    A21 = A2.dot(A1)
    A12 = A1.dot(A2)
    A1_1 = A1.dot(A1)
    A2_2 = A2.dot(A2)

    A112 = (A1_1.dot(A2)) * cons_11 * r_powers[3]
    A121 = (A1.dot(A21)) * cons_12 * r_powers[3]
    A122 = (A12.dot(A2)) * cons_13 * r_powers[3]
    A211 = (A21.dot(A1)) * cons_14 * r_powers[3]
    A212 = (A21.dot(A2)) * cons_15 * r_powers[2]
    A221 = (A2.dot(A21)) * cons_16 * r_powers[3]

    A11_12 = (A1_1.dot(A12)) * cons_17 * r_powers[4]

    A11_21 = (A1_1.dot(A21)) * cons_18 * r_powers[4]

    A11_22 = (A1_1.dot(A2_2)) * cons_19 * r_powers[4]

    A12_11 = (A12.dot(A1_1)) * cons_20 * r_powers[4]

    A12_12 = (A12.dot(A12)) * cons_21 * r_powers[4]

    A12_21 = (A12.dot(A21)) * cons_22 * r_powers[4]

    A12_22 = (A12.dot(A2_2)) * cons_23 * r_powers[4]

    A21_11 = (A21.dot(A1_1)) * cons_24 * r_powers[4]

    A21_12 = (A21.dot(A12)) * cons_25 * r_powers[4]

    A21_21 = (A21.dot(A21)) * cons_26 * r_powers[4]

    A21_22 = (A21.dot(A2_2)) * cons_27 * r_powers[4]

    A22_11 = (A2_2.dot(A1_1)) * cons_28 * r_powers[4]

    A22_12 = (A2_2.dot(A12)) * cons_29 * r_powers[4]

    A22_21 = (A2_2.dot(A21)) * cons_30 * r_powers[4]

    ## r_1 is sum of the terms [1],[3],[7],[15]
    r_1 = sum([t_powers[i + 1] * A1_powers[i + 1] / factorials[i + 1] for i in range(4)])

    r_2_1 = np.array([(exp_rh - 1) ** i for i in range(5)])

    multiplier_1 = (A12 * cons_9 +
                           A21 * cons_10 +
                           A112 +
                           A121 +
                           A211 +
                           A11_12 +
                           A11_21 +
                           A12_11 +
                           A21_11)

    multiplier_2 = (A122 +
                          A212 +
                          A221 +
                          A11_22 +
                          A12_12 +
                          A12_21 +
                          A21_12 +
                          A21_21 +
                          A22_11)

    multiplier_3 = (A12_22 +
                            A21_22 +
                            A22_12 +
                            A22_21)

    time_steps = np.arange(t_start, t_final + h, h)

    exp_rt0 = np.exp(r * time_steps)
    exp_2rt0 = np.exp(2 * r * time_steps)
    exp_3rt0 = np.exp(3 * r * time_steps)

    # term_start = time.time()
    # term = [(I + r_1 + exp_rt0[t] * multiplier_1 +
    #         exp_2rt0[t] * multiplier_2 +
    #         exp_3rt0[t] * multiplier_3) for t in range(len(time_steps))]
    #
    # print(f"Term calculation time: {time.time() - term_start} seconds")

    def compute_result_matrix_(t, X0):
        exp_rh_power = np.array([exp_rt0[t] ** i for i in range(5)])

        ## r_2 is the sum of the terms [2], [6], [14], [30]
        r_2 = sum([exp_rh_power[i + 1] * r_2_1[i + 1] * A2_powers[i + 1] * r_powers[i + 1] / factorials[i + 1] for i in
                   range(4)])

        X0 = (I + r_1 + r_2 + exp_rt0[t] * multiplier_1 +
                 exp_2rt0[t] * multiplier_2 +
                 exp_3rt0[t] * multiplier_3).dot(X0)


        # dot_start = time.time()
        # print(f"Matrix multiplication time: {time.time() - dot_start} seconds")

        return X0

    print(f"Precalculation time: {time.time() - start_time} seconds")

    loop_start = time.time()

    for t in range(len(time_steps[0:])):
        # print('t', t)
        # print('X0', X0)
        X0 = compute_result_matrix_(t, X0)
        # X0 = np.maximum(X0, 0)  ## put zero for values<0

    print(f"Loop time: {time.time() - loop_start} seconds")

    ex_time = time.time() - start_time
    print(f"Execution time: {ex_time} seconds")

    return X0, ex_time


import pickle
import os
from numpy import log
from scipy.sparse import load_npz

DOUBLING_TIME = 3
r = log(2) / DOUBLING_TIME


A2 =exp_imports.base_matrix

A1 = household_population.Q_int

X0 = H0

## I think here, with Joe's matrices, the transpose of A1 and A2 should be considered.
A1 = A1.T
A2 = A2.T
#####################

t_start = 0
t_final = 10

# Note_1 : If you want to change the step size in the code, you need two steps;
# First: change the h in the code "Peano-Baker", Second: change the h in the code "Run_Peano_Baker".

result, ex_time = main_(r, t_start, t_final, A1, A2, H0, h=.1)
print("Final Result at t_final:", result)

H_U_t = H_U[:, t_final]
print("Max relative error for H>1e-9 is", max(abs((result-H_U_t))[H_U_t>1e-9]/H_U_t[H_U_t>1e-9]))

# Try doing error as a function of step size:

step_sizes = array([1., 1e-1, 1e-2, 1e-3])
rel_err = zeros((len(step_sizes),))
ex_times = zeros((len(step_sizes),))

for i in range(len(step_sizes)):
    result, ex_times[i] = main_(r, t_start, t_final, A1, A2, H0, h=step_sizes[i])
    rel_err[i] = max(abs((result-H_U_t))[H_U_t>1e-9]/H_U_t[H_U_t>1e-9])

# Rough estimate of how many extra decimal points of precision you get for each step
# spent calculating, as a function of step size
decimals_per_second = -np.log10(rel_err/rel_err[0])[1:] / (ex_times[1:]-ex_times[0])

errplt, (logax, ratioax) = subplots(1,
                                  2,
                                  figsize = (10.5, 5))
errplt.subplots_adjust(wspace=0.5)

logax.loglog(step_sizes,
          rel_err,
          'ko:')
logax.set_xlabel("Step size")
logax.set_ylabel("Relative Error")
logax2 = logax.twinx()
logax2.loglog(step_sizes,
          ex_times,
          'rx:')
logax2.set_ylabel("Execution time (s)")

ratioax.loglog(ex_times,
          rel_err,
          'ko:')
ratioax.set_xlabel("Execution time (s)")
ratioax.set_ylabel("Relative Error")

errplt.show()

errplt.savefig('plots/uk/pb_ex_time.png', bbox_inches='tight', dpi=300)