'''This sets up and runs a simple system which is low-dimensional enough to
do locally'''
from os.path import isfile
from numpy import (
        arange, array, atleast_2d, concatenate, copy, empty, ix_, ones, where, diag, zeros)
from numpy import int64 as my_int
from numpy.random import rand
from time import time as get_time
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix as sparse
# import ode
from model.preprocessing import (
    CareHomeInput, HouseholdPopulation, initialise_carehome)
from model.specs import CAREHOME_SPEC
from model.common import CareHomeRateEquations, build_state_matrix
from model.imports import CareHomeImportModel
from pickle import load, dump
# pylint: disable=invalid-name


class CareHomePopulation(HouseholdPopulation):
    def __init__(
            self,
            composition_list,
            composition_distribution,
            model_input,
            print_progress=True):
        super().__init__(
            composition_list,
            composition_distribution,
            model_input,
            no_compartments=6,
            print_progress=print_progress)

    def _create_subsystems(self, household_spec):
        '''Assuming frequency-dependent homogeneous within-household mixing
        composition[i] is the number of individuals in age-class i inside the
        household'''
        # susceptibility indexed by patients and carers
        sus = self.model_input.sus
        # reduction in infectivity during prodrome
        tau = self.model_input.tau
        # internal mixing matrix
        K_home = model_input.k_home
        # E->P progression rate
        alpha_1 = model_input.alpha_1
        # P->I progression rate
        alpha_2 = model_input.alpha_2
        # Recovery (I->R) rate
        gamma = model_input.gamma
        # Disease-specific mortality rate
        mu_cov = model_input.mu_cov
        # Background bed clearance rate
        mu = model_input.mu
        # Arrival rate into empty beds
        b = model_input.b

        # Set of individuals actually present here
        class_idx = household_spec.class_indexes
        no_compartments = self.num_of_epidemiological_compartments

        K_home = K_home[ix_(class_idx, class_idx)]
        sus = sus[class_idx]
        tau = tau[class_idx]
        r_home = atleast_2d(diag(sus).dot(K_home))

        states, \
            reverse_prod, \
            index_vector, \
            rows = build_state_matrix(household_spec)
        # Location of prodromals in state vector
        p_pos = 2 + no_compartments * arange(len(class_idx))
        # Location of infected in state vector
        i_pos = 3 + no_compartments * arange(len(class_idx))
        # Location of empty beds in state vector
        d_pos = 5 + no_compartments * arange(len(class_idx))

        # This is number of people actually present, given some beds are empty
        empty_adjusted_comp = household_spec.composition[class_idx] - states[:, d_pos]
        # Replace zeros with ones - we only ever use this as a denominator whose
        # numerator will be zero anyway if it should be zero
        empty_adjusted_comp[empty_adjusted_comp == 0] = 1

        Q_int = sparse(household_spec.matrix_shape)
        inf_event_row = array([], dtype=my_int)
        inf_event_col = array([], dtype=my_int)
        inf_event_class = array([], dtype=my_int)

        # Add events for each age class
        for i in range(len(class_idx)):

            s_present = where(states[:, 6*i] > 0)[0]
            e_present = where(states[:, 6*i+1] > 0)[0]
            p_present = where(states[:, 6*i+2] > 0)[0]
            i_present = where(states[:, 6*i+3] > 0)[0]
            r_present = where(states[:, 6*i+4] > 0)[0]
            d_present = where(states[:, 6*i+5] > 0)[0]

            # First do infection events
            inf_to = zeros(len(s_present), dtype=my_int)
            inf_rate = zeros(len(s_present))
            for k in range(len(s_present)):
                old_state = copy(states[s_present[k], :])
                # tau is prodromal reduction in infectivity
                inf_rate[k] = old_state[6*i] * (
                    r_home[i, :].dot(
                        (old_state[i_pos] / empty_adjusted_comp[k])
                        + tau * (old_state[p_pos] / empty_adjusted_comp[k])))
                new_state = old_state.copy()
                new_state[6*i] -= 1
                new_state[6*i + 1] += 1
                inf_to[k] = index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (inf_rate, (s_present, inf_to)),
                shape=household_spec.matrix_shape)
            inf_event_row = concatenate((inf_event_row, s_present))
            inf_event_col = concatenate((inf_event_col, inf_to))
            inf_event_class = concatenate(
                (inf_event_class, class_idx[i]*ones((len(s_present)))))
            # # disp('Infection events done')

            # # Now do exposure to prodromal
            inc_to = zeros(len(e_present), dtype=my_int)
            inc_rate = zeros(len(e_present))
            for k in range(len(e_present)):
                # First do detected
                old_state = copy(states[e_present[k], :])
                inc_rate[k] = alpha_1 * old_state[6*i+1]
                new_state = copy(old_state)
                new_state[6*i + 1] -= 1
                new_state[6*i + 2] += 1
                inc_to[k] = index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]

            Q_int += sparse(
                (inc_rate, (e_present, inc_to)),
                shape=household_spec.matrix_shape)
            # # disp('Incubaion events done')

            # # Now do prodromal to infectious
            dev_to = zeros(len(p_present), dtype=my_int)
            dev_rate = zeros(len(p_present))
            for k in range(len(p_present)):
                # First do detected
                old_state = copy(states[p_present[k], :])
                dev_rate[k] = alpha_2 * old_state[6*i+2]
                new_state = copy(old_state)
                new_state[6*i + 2] -= 1
                new_state[6*i + 3] += 1
                dev_to[k] = index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]

            Q_int += sparse(
                (dev_rate, (p_present, dev_to)),
                shape=household_spec.matrix_shape)

            # Now do recovery of infectious cases
            rec_to = zeros(len(i_present), dtype=my_int)
            rec_rate = zeros(len(i_present))
            for k in range(len(i_present)):
                old_state = copy(states[i_present[k], :])
                rec_rate[k] = gamma * old_state[6*i+3]
                new_state = copy(old_state)
                new_state[6*i+3] -= 1
                new_state[6*i+4] += 1
                rec_to[k] = index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (rec_rate, (i_present, rec_to)),
                shape=household_spec.matrix_shape)
            # disp('Recovery events done')

            # Now do emptying of beds
            emp_to = zeros(len(s_present), dtype=my_int)
            emp_rate = zeros(len(s_present))
            for k in range(len(s_present)):
                old_state = copy(states[s_present[k], :])
                emp_rate[k] = mu[i] * old_state[6*i]
                new_state = copy(old_state)
                new_state[6*i] -= 1
                new_state[6*i+5] += 1
                emp_to[k] - index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (emp_rate, (s_present, emp_to)),
                shape=household_spec.matrix_shape)

            emp_to = zeros(len(e_present), dtype=my_int)
            emp_rate = zeros(len(e_present))
            for k in range(len(e_present)):
                old_state = copy(states[e_present[k], :])
                emp_rate[k] = mu[i] * old_state[6*i+1]
                new_state = copy(old_state)
                new_state[6*i+1] -= 1
                new_state[6*i+5] += 1
                emp_to[k] - index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (emp_rate, (e_present, emp_to)),
                shape=household_spec.matrix_shape)

            emp_to = zeros(len(p_present), dtype=my_int)
            emp_rate = zeros(len(p_present))
            for k in range(len(p_present)):
                old_state = copy(states[p_present[k], :])
                emp_rate[k] = mu[i] * old_state[6*i+2]
                new_state = copy(old_state)
                new_state[6*i+2] -= 1
                new_state[6*i+5] += 1
                emp_to[k] - index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (emp_rate, (p_present, emp_to)),
                shape=household_spec.matrix_shape)

            emp_to = zeros(len(i_present), dtype=my_int)
            emp_rate = zeros(len(i_present))
            for k in range(len(s_present)):
                old_state = copy(states[s_present[k], :])
                emp_rate[k] = (mu[i] + mu_cov[i]) * old_state[6*i+3]
                new_state = copy(old_state)
                new_state[6*i+3] -= 1
                new_state[6*i+5] += 1
                emp_to[k] - index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (emp_rate, (i_present, emp_to)),
                shape=household_spec.matrix_shape)

            emp_to = zeros(len(r_present), dtype=my_int)
            emp_rate = zeros(len(r_present))
            for k in range(len(r_present)):
                old_state = copy(states[s_present[k], :])
                emp_rate[k] = mu[i] * old_state[6*i+4]
                new_state = copy(old_state)
                new_state[6*i+4] -= 1
                new_state[6*i+5] += 1
                emp_to[k] - index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (emp_rate, (r_present, emp_to)),
                shape=household_spec.matrix_shape)

            # Now do arrival of new patients into empty beds
            arr_to = zeros(len(d_present), dtype=my_int)
            arr_rate = zeros(len(d_present))
            for k in range(len(d_present)):
                old_state = copy(states[d_present[k], :])
                arr_rate[k] = b[i] * old_state[6*i+5]
                new_state = copy(old_state)
                new_state[6*i+5] -= 1
                new_state[6*i] += 1 # We assume new arrivals are susceptible!
                arr_to[k] - index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (arr_rate, (d_present, arr_to)),
                shape=household_spec.matrix_shape)

        S = Q_int.sum(axis=1).getA().squeeze()
        Q_int += sparse((
            -S,
            (
                arange(household_spec.total_size),
                arange(household_spec.total_size))))
        return tuple((
            Q_int,
            states,
            array(inf_event_row, dtype=my_int, ndmin=1),
            array(inf_event_col, dtype=my_int, ndmin=1),
            array(inf_event_class, dtype=my_int, ndmin=1)))



model_input = CareHomeInput(CAREHOME_SPEC)

# Model a single care home type:
composition_list = array([[10,0,0]])
carehome_size = sum(sum(composition_list))
comp_dist = array([1])

if isfile('carehome_pop.pkl') is True:
    with open('carehome_pop.pkl', 'rb') as f:
        carehome_population = load(f)
else:
    pop_start = get_time()
    carehome_population = CareHomePopulation(
        composition_list,
        comp_dist,
        model_input)
    with open('carehome_pop.pkl', 'wb') as f:
        dump(carehome_population, f)
    print('Within-carehome event matrix generated and saved in',
        get_time()-pop_start,'seconds.')

no_days = 100
import_times = arange(7,no_days+7,7) # Weekly imports covering simulation time
prodromal_prev = 1e-1*(
    (1/model_input.alpha_2)/(1/model_input.alpha_2 + 1/model_input.gamma)
    ) * rand(len(import_times))
infected_prev = 1e-1*(
    (1/model_input.gamma)/(1/model_input.alpha_2 + 1/model_input.gamma)
    ) * rand(len(import_times))

import_model = CareHomeImportModel(
    import_times,
    prodromal_prev,
    infected_prev)

rhs = CareHomeRateEquations(
    model_input,
    carehome_population,
    import_model)

initial_presence = array([[1,0,0]])

H0 = initialise_carehome(
    carehome_population, rhs, initial_presence)

print('Initial conditions set up.')

print('Now solving...')

tspan = (0.0, no_days)
sol_start = get_time()
solution = solve_ivp(rhs, tspan, H0, first_step=0.001)
#solution = ode.backwardeuler(rhs, H0, tspan, 0.001)
#solution = ode.backwardeuler(rhs, H0, tspan, timestep=0.001)
sol_end = get_time()

#ode.BackwardEuler(dfun=rhs, xzero=H0, timerange=tspan,
 #   timestep=0.001, convergencethreshold=1e-10, maxiterations=1000

#time = solution[0]
#H = solution[1]
time=solution.t
H = solution.y
P = H.T.dot(carehome_population.states[:, 2::5])
I = H.T.dot(carehome_population.states[:, 3::5])

print(
    'Solution took ',
    sol_end-sol_start,
    ' seconds.')

Prob = empty(shape=(carehome_size,len(time)), dtype = object)

for i in range(carehome_size):
    for j in range(len(time)):
        Prob[i,j] = sum(H[where(carehome_population.states[:,3]==i)].T[j])

with open('carehome_results.pkl','wb') as f:
    dump((time, carehome_size, H,P,I,Prob),f)
