#Add the packages we need for the inferring process
import numpy as np

from numpy.random import choice
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from pickle import dump, load
import scipy as sp
from scipy.integrate import solve_ivp
import time
from tqdm import tqdm
from scipy.optimize import minimize
import scipy as sp
from scipy.integrate import solve_ivp
from numpy import linalg as LA

import pickle

from numpy.linalg import eig

#Make the model work on a Windows

import os
if 'inference-from-testing' in os.getcwd():
    os.chdir("../..")
os.getcwd()

#Add the necessary packages for the model

from copy import deepcopy
from numpy import arange, array, atleast_2d, log
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import SINGLE_AGE_SEIR_SPEC_FOR_FITTING, SINGLE_AGE_UK_SPEC
from model.common import SEIRRateEquations
from model.imports import FixedImportModel

# Use a random seed for reproducibility

np.random.seed(637)


DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

comp_dist = read_csv(
    'inputs/england_hh_size_dist.csv',
    header=0).to_numpy().squeeze()
comp_dist = comp_dist[:3]
comp_dist[:2] *= 0
comp_dist = comp_dist/sum(comp_dist)
max_hh_size = len(comp_dist)
composition_list = np.atleast_2d(arange(1, max_hh_size+1)).T

#comp_dist = array([1.])
#composition_list = array([[1]])

if isdir('outputs') is False:
    mkdir('outputs')
if isdir('outputs/inference-from-testing') is False:
    mkdir('outputs/inference-from-testing')

SPEC = {**SINGLE_AGE_SEIR_SPEC_FOR_FITTING, **SINGLE_AGE_UK_SPEC}
base_sitp = SPEC["SITP"]
SPEC["SITP"] = 1 - (1-base_sitp)**3
model_input_to_fit = SEIRInput(SPEC, composition_list, comp_dist)
household_population_to_fit = HouseholdPopulation(
    composition_list, comp_dist, model_input_to_fit)
beta_ext = 0.
model_input = deepcopy(model_input_to_fit)
model_input.k_ext *= beta_ext

true_density_expo = model_input.density_expo


# With the parameters chosen, we calculate Q_int:
household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

true_lam = 3.
pop_prev = 1e-2
rhs = SEIRRateEquations(model_input, household_population, FixedImportModel(4,1, np.array([true_lam * pop_prev]), 0))

H0 = np.zeros((household_population.total_size),)
all_sus = np.where(np.sum(rhs.states_exp_only + rhs.states_inf_only + rhs.states_rec_only, 1)<1e-1)[0]
one_inf = np.where((np.abs(np.sum(rhs.states_inf_only, 1) - 1)<1e-1) & (np.sum(rhs.states_exp_only + rhs.states_rec_only, 1)<1e-1))[0]
H0[all_sus] = 0.99 * comp_dist
H0[one_inf] = 0.01 * comp_dist
S0 = H0.T.dot(household_population.states[:, ::4])
E0 = H0.T.dot(household_population.states[:, 1::4])
I0 = H0.T.dot(household_population.states[:, 2::4])
R0 = H0.T.dot(household_population.states[:, 3::4])
start_state = (1/model_input.ave_hh_size) * array([S0.sum(),
                                                   E0.sum(),
                                                   I0.sum(),
                                                   R0.sum()])
tspan = (0.0, 365)
print("(S,E,I,R)(0)=",start_state)
