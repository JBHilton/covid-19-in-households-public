# This file contains functions necessary for performing inference from longitudinal testing data, including likelihood
# calculations.

from copy import deepcopy
from numpy import abs, array, log, sum, where
from scipy.integrate import solve_ivp
from model.preprocessing import make_initial_condition_by_eigenvector

def one_step_household_llh(hh_data,
                  test_times,
                  tau,
                  lam,
                  base_rhs,
                 growth_rate,
                 init_prev,
                 R_comp):
    '''This function calculates the log likelihood of parameters tau and lam given some data with test results at two
    time points for a single household. This assumes data comes as number of positive tests at start and end time point,
    and that each positive test corresponds to exactly one individual in the I compartment of the model.'''
    rhs = deepcopy(base_rhs)
    rhs.int_rate *= tau
    rhs.ext_rate *= lam

    H0 = make_initial_condition_by_eigenvector(growth_rate,
                                               rhs.model_input,
                                               rhs.household_population,
                                               rhs,
                                               init_prev,
                                               0.0,
                                               False,
                                               R_comp)
    Ht = 0 * H0 # Set up initial condition vector
    possible_states = where(abs(sum(rhs.states_inf_only, 1) - hh_data[0]) < 1e-1)[0]
    # Set Ht equal to eigenvector initial condition, conditioned on test results
    Ht[possible_states, ] = H0[possible_states, ] / sum(H0[possible_states, ])
    llh = 0

    tspan = (test_times[0], test_times[1])
    solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-16)
    T = solution.t
    H = solution.y
    I = hh_data[1]
    possible_states = where(abs(sum(rhs.states_inf_only,1)-I)<1e-1)[0]
    llh += log(sum(H[possible_states, -1]))
    return(llh)

def one_step_population_likelihood(test_data,
                  test_times,
                  tau,
                  lam,
                  base_rhs,
                 growth_rate,
                 init_prev,
                 R_comp):
    '''This is a wrapper for one_step_household_llh function, which allows for the calculation of the joint likelihood
    of an entire population's worth of independent one-step testing samples in a single function call.'''
    return (sum(array([one_step_household_llh(data_i,
                  test_times,
                  tau,
                  lam,
                  base_rhs,
                 growth_rate,
                 init_prev,
                 R_comp) for data_i in test_data])))

# Irene's version of what one_step_population_likelihood could look like.
def one_step_population_likelihood(test_data,
                                   test_times,
                                   tau,
                                   lam,
                                   base_rhs,
                                   growth_rate,
                                   init_prev=1e-2,
                                   R_comp=3):
    '''This is a wrapper for one_step_household_llh function, which allows for the calculation of the joint likelihood
    of an entire population's worth of independent one-step testing samples in a single function call.'''
    total_llh = sum(
        one_step_household_llh(hh_data,
                               test_times,
                               tau,
                               lam,
                               base_rhs,
                               growth_rate,
                               init_prev,
                               R_comp)
        for hh_data in test_data
    )
    return total_llh

#Other stuff we might need later

def run_one_step_inference(test_data,
                           test_times,
                           base_rhs,
                           growth_rate,
                           init_prev=1e-2,
                           R_comp=3,
                           tau_init=0.09,
                           lam_init=3.0,
                           bounds=((0.0, 0.15), (2.0, 5.0))):

    def neg_log_likelihood(params):
        tau, lam = params
        return one_step_population_likelihood(test_data,
                                               test_times,
                                               tau,
                                               lam,
                                               base_rhs,
                                               growth_rate,
                                               init_prev,
                                               R_comp)

    result = minimize(neg_log_likelihood, [tau_init, lam_init], bounds=bounds)
    tau_hat, lam_hat = result.x
    return tau_hat, lam_hat
