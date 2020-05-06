'''Defines the parameters we will use in the model'''

from numpy import array, arange, concatenate, diag, linspace, ones, where, zeros
from pandas import read_excel, read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import (
    make_aggregator, aggregate_contact_matrix,
    aggregate_vector_quantities)
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

gamma = 1.0/2.0
R0 = 2.4

rho = gamma * R0 * aggregate_vector_quantities(
    rho, fine_bds, coarse_bds, pop_pyramid).toarray().squeeze()

det = (0.9/max(rho)) * rho

params = {'R0' : R0,
    'gamma' : gamma,
    'alpha' : 1.0/5.0,
    'det' : det,
    'sigma' : rho / det,
    'tau' : 0.0 * ones(len(rho)),
    'k_home' : k_home,
    'k_ext' : k_ext,
    'coarse_bds' : coarse_bds
    }
