
from model.preprocessing import ( estimate_beta_ext, SEIRInput, HouseholdPopulation)
from model.specs import SINGLE_AGE_SEIR_SPEC_FOR_FITTING, SINGLE_AGE_UK_SPEC
from model.common import SEIRRateEquations, UnloopedSEIRRateEquations
from model.imports import ExponentialImportModel, FixedImportModel, NoImportModel
from numpy import arange, array, log, where, zeros
import time
import numpy as np
import scipy.sparse as sp
import math

growth_rate = (1/11) * log((29/2343) / (73/2812))

hh_size = 8

composition_list = array([[hh_size]])

comp_dist = array([1.])


SPEC = {**SINGLE_AGE_SEIR_SPEC_FOR_FITTING, **SINGLE_AGE_UK_SPEC}
model_input_to_fit = SEIRInput(SPEC, composition_list, comp_dist)
household_population_to_fit = HouseholdPopulation(
    composition_list, comp_dist, model_input_to_fit)
rhs_to_fit = SEIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(4,1))
beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
model_input = model_input_to_fit
model_input.k_ext *= beta_ext
print('Estimated beta is',beta_ext)



household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

no_imports = NoImportModel(4, 1)
base_rhs = SEIRRateEquations(model_input, household_population, no_imports)
fixed_imports = FixedImportModel(4,1, base_rhs, array([.1]))
exp_imports = ExponentialImportModel(4,
                                     1,
                                     base_rhs,
                                     growth_rate,
                                     array([1e-5]))
rhs = UnloopedSEIRRateEquations(model_input, household_population, exp_imports, sources="IMPORT")

Q_int = rhs.Q_int
Q_ext = exp_imports.base_matrix

init_cases = 1
S0 = hh_size - init_cases
E0 = 0
I0 = init_cases
R0 = 0
x0 = zeros(Q_int.shape[0])
init_state = where((rhs.states_sus_only==S0)&
                    (rhs.states_exp_only==E0)&
                    (rhs.states_inf_only==I0)&
                    (rhs.states_rec_only==R0))[0]

x0[init_state] = 1.

t_start = 0
t_final = 7
delta_t_pb = 0.5
delta_t = 1

r = growth_rate

A2 = Q_ext

A1 = Q_int

A1 = A1.T
A2 = A2.T
# ######################
## Peanno Baker method

matrix_size_pb = A1.shape[0]
I = sp.eye(matrix_size_pb, format='csr')
factorials_pb = np.array([math.factorial(i) for i in range(4)])
A1_powers_pb = [sp.csr_matrix(I)]
A2_powers_pb = [sp.csr_matrix(I)]

for i in range(1, 5):
    A1_powers_pb.append(A1_powers_pb[-1].dot(A1))  # A1^i
    A2_powers_pb.append(A2_powers_pb[-1].dot(A2))  # A2^i

t_powers_pb = np.array([delta_t_pb ** i for i in range(4)])

r_powers_pb = np.array([(1 / r) ** i for i in range(4)])


exp_rh_pb = np.exp(r * delta_t_pb)
exp_2rh_pb = np.exp(2 * r * delta_t_pb)


cons_9 = (exp_rh_pb - r * delta_t_pb - 1) * r_powers_pb[2]
cons_10 = (exp_rh_pb * (r * delta_t_pb - 1) + 1) * r_powers_pb[2]
cons_11 = (2 * exp_rh_pb - (r ** 2) * t_powers_pb[2] - 2 * r * delta_t_pb - 2) / 2
cons_12 = exp_rh_pb * (r * delta_t_pb - 2) + r * delta_t_pb + 2
cons_13 = ((exp_rh_pb - 2) ** 2 + 2 * r * delta_t_pb - 1) / 4
cons_14 = (exp_rh_pb * ((r ** 2) * t_powers_pb[2] - 2 * r * delta_t_pb + 2) - 2) / 2
cons_15 = exp_2rh_pb / (2 * r) - exp_rh_pb * delta_t_pb - 1 / (2 * r)
cons_16 = (exp_2rh_pb * (2 * r * delta_t_pb - 3) + 4 * exp_rh_pb - 1) / 4

A21_pb = A2.dot(A1)
A12_pb = A1.dot(A2)
A1_1_pb = A1.dot(A1)


A112_pb = (A1_1_pb.dot(A2)) * cons_11 * r_powers_pb[3]
A121_pb = (A1.dot(A21_pb)) * cons_12 * r_powers_pb[3]
A122_pb = (A12_pb.dot(A2)) * cons_13 * r_powers_pb[3]
A211_pb = (A21_pb.dot(A1)) * cons_14 * r_powers_pb[3]
A212_pb = (A21_pb.dot(A2)) * cons_15 * r_powers_pb[2]
A221_pb = (A2.dot(A21_pb)) * cons_16 * r_powers_pb[3]

r_1_pb = sum([t_powers_pb[i + 1] * A1_powers_pb[i + 1] / factorials_pb[i + 1] for i in range(3)])

r_1_1_pb = r_1_pb + I

r_2_1_pb= np.array([(exp_rh_pb - 1) ** i for i in range(4)])

time_steps_pb = np.arange(t_start, t_final + delta_t_pb, delta_t_pb)


multiplier_1_pb = (A12_pb * cons_9 +
                A21_pb * cons_10 +
                A112_pb +
                A121_pb +
                A211_pb)

multiplier_2_pb = (A122_pb +
                A212_pb +
                A221_pb)

def compute_result_matrix_(t0, X0):

    exp_rt0_pb = np.exp(r * t0)

    exp_rh_power_pb = np.array([exp_rt0_pb ** i for i in range(4)])

    r_2_pb = sum([exp_rh_power_pb[i + 1] * r_2_1_pb[i + 1] * A2_powers_pb[i + 1] * r_powers_pb[i + 1] / factorials_pb[i + 1] for i in
                   range(3)])

    X0 = (r_1_1_pb + r_2_pb + exp_rh_power_pb[1] * multiplier_1_pb + exp_rh_power_pb[2] * multiplier_2_pb).dot(X0)

    return X0

def main_(X0):

    time_vector_pb = [time_steps_pb[0]]
    for t in time_steps_pb[1:]:
        X0 = compute_result_matrix_(t, X0)
        X0 = np.maximum(X0, 0)

        time_vector_pb.append(t)  # Store t


    return X0, np.array(time_vector_pb)


#####################
execution_times_pb = []

for _ in range(1000):
    start_time = time.time()
    result, time_vector_pb = main_(x0)
    end_time = time.time()
    execution_times_pb.append(end_time - start_time)

average_time = sum(execution_times_pb) / len(execution_times_pb)
print('Average execution time_PB:', average_time)
