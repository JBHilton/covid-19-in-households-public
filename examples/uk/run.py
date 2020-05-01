'''This runs the UK-like model with a single set of parameters for 100 days
''' 
from pickle import load, dump
from numpy import array, arange, concatenate, diag, linspace, ones, where, zeros
from pandas import read_excel, read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import (
    make_aggregator, aggregate_contact_matrix,
    aggregate_vector_quantities, build_household_population)
from model.common import sparse, hh_ODE_rates

k_home = read_excel(
    'inputs/MUestimates_home_2.xlsx',
    sheet_name='United Kingdom of Great Britain',
    header=None).to_numpy()
k_all = read_excel(
    'inputs/MUestimates_all_locations_2.xlsx',
    sheet_name='United Kingdom of Great Britain',
    header=None).to_numpy()

# Because we want 80 to be included as well.
fine_bds = arange(0, 81, 5)
coarse_bds = concatenate((fine_bds[:6], fine_bds[12:]))

pop_pyramid = read_csv(
    'inputs/United Kingdom-2019.csv', index_col=0)
pop_pyramid = (pop_pyramid['F'] + pop_pyramid['M']).to_numpy()

k_home = aggregate_contact_matrix(k_home, fine_bds, coarse_bds, pop_pyramid)
k_all= aggregate_contact_matrix(k_all, fine_bds, coarse_bds, pop_pyramid)
k_ext = k_all - k_home

# This is in ten year blocks
rho = read_csv(
    'inputs/rho_estimate_cdc.csv', header=None).to_numpy().flatten()

cdc_bds = arange(0, 81, 10)
aggregator = make_aggregator(cdc_bds, fine_bds)

# This is in five year blocks
rho = sparse(
    (rho[aggregator], (arange(len(aggregator)), [0]*len(aggregator)))) 

R0=2.4
gamma = 0.5
rho = gamma * R0 * aggregate_vector_quantities(
    rho, fine_bds, coarse_bds, pop_pyramid).toarray().squeeze()
det = (0.9/max(rho)) * rho
sigma = rho / det
# No asymptomatic transmission
tau = 0.0 * ones(len(rho))
alpha = 1.0 / 5.0

# List of observed household compositions
composition_list = read_csv(
    'inputs/uk_composition_list.csv',
    header=None).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/uk_composition_dist.csv',
    header=None).to_numpy().squeeze()

from os.path import isfile
if isfile('vars.pkl') is True:
    with open('vars.pkl', 'rb') as f:
        Q_int, states, which_composition, \
                system_sizes, cum_sizes, \
                inf_event_row, inf_event_col \
            = load(f)
else: 
    # With the parameters chosen, we calculate Q_int:
    Q_int, states, which_composition, \
            system_sizes, cum_sizes, \
            inf_event_row, inf_event_col \
        = build_household_population(
            composition_list,
            sigma,
            det,
            tau,
            k_home,
            alpha,
            gamma)
    with open('vars.pkl', 'wb') as f:
        dump(
            (Q_int, states, which_composition, system_sizes,
            cum_sizes, inf_event_row, inf_event_col),
            f)

total_size = len(which_composition)

# To define external mixing we need to set up the transmission matrices:
det_trans_matrix = diag(sigma).dot(k_ext) # Scale rows of contact matrix by
                                          # age-specific susceptibilities
# Scale columns by asymptomatic reduction in transmission
undet_trans_matrix = diag(sigma).dot(k_ext.dot(diag(tau)))
# This stores number in each age class by household
composition_by_state = composition_list[which_composition,:]
states_sus_only = states[:,::5] # ::5 gives columns corresponding to
                                # susceptible cases in each age class in
                                # each state
s_present = where(states_sus_only.sum(axis=1) > 0)[0]

# Our starting state H is the composition distribution with a small amount of
# infection present:
states_det_only = states[:,2::5] # 2::5 gives columns corresponding to
                                 # detected cases in each age class in each
                                 # state
states_undet_only = states[:,3::5] # 4:5:end gives columns corresponding to
                                   # undetected cases in each age class in
                                   # each state
fully_sus = where(states_sus_only.sum(axis=1) == states.sum(axis=1))[0]
i_is_one = where((states_det_only + states_undet_only).sum(axis=1) == 1)[0]

H0 = zeros(len(which_composition))
# Assign probability of 1e-5 to each member of each composition being sole infectious person in hh
H0[i_is_one] = (1.0e-5) * comp_dist[which_composition[i_is_one]]
# Assign rest of probability to there being no infection in the household
H0[fully_sus] = (1 - 1e-5 * sum(comp_dist[which_composition[i_is_one]])) * comp_dist

def RHS(t, p):
    print(t)
    return hh_ODE_rates(
        t,
        p,
        Q_int,
        states,
        composition_by_state,
        states_sus_only,
        states_det_only,
        states_undet_only,
        det_trans_matrix,
        undet_trans_matrix,
        inf_event_row,
        inf_event_col,
        total_size)

tspan = (0.0, 100)
solution = solve_ivp(RHS, tspan, H0, first_step=0.001)

time = solution.t
H = solution.y
D = H.T.dot(states[:,2::5])
U = H.T.dot(states[:,3::5])

with open('uk_like.pkl', 'wb') as f:
    dump((time, H, D, U, coarse_bds), f)
