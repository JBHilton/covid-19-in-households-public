'''This script constructs the internal transmission matrix for a UK-like
population and a single instance of the external importation matrix.
'''
from numpy import array, arange, concatenate, diag, ones, where, zeros
from pandas import read_excel, read_csv
from model.preprocessing import (
    make_aggregator, aggregate_contact_matrix,
    aggregate_vector_quantities, build_household_population)
from model.common import get_FOI_by_class, build_external_import_matrix
from model.common import sparse
from model.defineparameters import params

fine_bds = arange(0, 81, 5)  # Because we want 80 to be included as well.

coarse_bds = concatenate((fine_bds[:6], fine_bds[12:]))

pop_pyramid = read_csv(
    'inputs/United Kingdom-2019.csv', index_col=0)

pop_pyramid = (pop_pyramid['F'] + pop_pyramid['M']).to_numpy()

# This is in ten year blocks
rho = read_csv(
    'inputs/rho_estimate_cdc.csv', header=None).to_numpy().flatten()

cdc_bds = arange(0, 81, 10)
aggregator = make_aggregator(cdc_bds, fine_bds)

# This is in five year blocks
rho = sparse(
    (rho[aggregator], (arange(len(aggregator)), [0]*len(aggregator))))

rho = aggregate_vector_quantities(
    rho, fine_bds, coarse_bds, pop_pyramid).toarray().squeeze()

params['det'] = 0.2 * ones(rho.shape)
params['tau'] = 0.5 * ones(rho.shape)
params['sigma'] = rho / params['det']

# List of observed household compositions
composition_list = read_csv(
    'inputs/uk_composition_list.csv',
    header=None).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/uk_composition_dist.csv',
    header=None).to_numpy().squeeze()

# With the parameters chosen, we calculate Q_int:
Q_int, states, which_composition, \
        system_sizes, cum_sizes, \
        inf_event_row, inf_event_col \
        = build_household_population(
    composition_list,
    params['sigma'],
    params['det'],
    params['tau'],
    params['k_home'],
    params['alpha'],
    params['gamma'])

# To define external mixing we need to set up the transmission matrices:
det_trans_matrix = diag(params['sigma']) * params['k_ext'] # Scale rows of contact matrix by
                                       # age-specific susceptibilities
# Scale columns by asymptomatic reduction in transmission
undet_trans_matrix = diag(params['sigma']).dot(params['k_ext'].dot(diag(params['tau'])))
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
H = zeros(len(which_composition))
# Assign probability of 1e-5 to each member of each composition being sole infectious person in hh
H[i_is_one] = (1e-5) * comp_dist[which_composition[i_is_one]]
# Assign rest of probability to there being no infection in the household
H[fully_sus] = (1 - 1e-5 * sum(comp_dist[which_composition[i_is_one]])) * comp_dist

# Calculate force of infection on each state
FOI_det,FOI_undet = get_FOI_by_class(
    H,
    composition_by_state,
    states_sus_only,
    states_det_only,
    states_undet_only,
    det_trans_matrix,
    undet_trans_matrix)
# Now calculate the external infection components of the transmission
# matrix:
Q_ext_det, Q_ext_undet = build_external_import_matrix(
    states,
    inf_event_row,
    inf_event_col,
    FOI_det,
    FOI_undet,
    len(which_composition))

def read_test(name, M):
    A = read_csv(
        'matlab_src/{}.mat'.format(name),
        skiprows=6,
        header=None,
        delimiter=' ').to_numpy()
    return sparse((
        A[:, 2], (A[:, 0]-1, A[:, 1]-1)),
        shape=M.shape)

def compare(A, B):
    return abs(A - B).sum()
