
from model.preprocessing import ( estimate_beta_ext, SEIRInput, HouseholdPopulation)
from model.specs import SINGLE_AGE_SEIR_SPEC_FOR_FITTING, SINGLE_AGE_UK_SPEC
from model.common import SEIRRateEquations, UnloopedSEIRRateEquations
from model.imports import ExponentialImportModel, FixedImportModel, NoImportModel
from numpy import arange, array, log, where, zeros
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

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
print("len(init_state) =",len(init_state))
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
############################
## Magnus

times = np.arange(t_start, t_final + delta_t, delta_t)
def main_magnus(r, A1, A2, X0, h):

    A1 = A1.tocsr()
    A2 = A2.tocsr()

    A21 = A2 @ A1
    A12 = A1 @ A2
    A21_12 = A21 - A12
    A2112_212 = -2 * A21 @ A2 + A12 @ A2 + A2 @ A21
    A121_1221 = 2 * A12 @ A1 - A1 @ A12 - A21 @ A1


    exp_rh = np.exp(r * h)
    exp_2rh = np.exp(2 * r * h)


    term_1 = A2112_212 * ((r * h - 3) * exp_2rh + 4 * r * h * exp_rh + h * r + 3) / (12 * r**3)
    term_21 = A121_1221 * (-((h**2) * (r**2) - 6 * r * h + 12) * exp_rh + (h**2) * (r**2) + 6 * r * h + 12) / (12 * r**3)
    term_22 = A21_12 * ((r * h - 2) * exp_rh + r * h + 2) / (2 * r**2) + A2 * (exp_rh - 1) / r
    term_2 = term_21 + term_22
    term_3 = A1 * h
    I = sp.identity(A1.shape[0], format="csr")


    def expm_approx(A, X):

        sol = expm_multiply(A,X)

        return sol


    def compute_result_matrix_(t, X):

        exp_rt = np.exp(r * t)
        exp_2rt = np.exp(2 * r * t)

        sum_ = term_3 + exp_rt * term_2 + exp_2rt * term_1

        X = expm_approx(sum_, X)

        return np.maximum(X, 0)


    X_values = [x0.copy()]
    t_values = [times[0]]

    for t in times[1:]:
        X0 = compute_result_matrix_(t, X0)
        X_values.append(X0.copy())
        t_values.append(t)


    return X0, np.array(t_values)
#####################
execution_times = []

for _ in range(1000):
    start_time = time.time()
    vector_fractional, time_method = main_magnus(r, A1, A2, x0, delta_t)
    end_time = time.time()
    execution_times.append(end_time - start_time)

average_time = sum(execution_times) / len(execution_times)
print('Average execution time_Magnus:', average_time)


