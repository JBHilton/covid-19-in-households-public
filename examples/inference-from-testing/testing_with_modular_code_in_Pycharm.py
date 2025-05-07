# The packages we ll need for the inferring process


true_lam = 3.

import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, savefig
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

#Get the code running on Windows

import os

if 'inference-from-testing' in os.getcwd():
    os.chdir("../..")
os.getcwd()

# Packages we need from Joe's code

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
from model.common import UnloopedSEIRRateEquations, UnloopedSEPIRRateEquations
from model.imports import FixedImportModel
from model.imports import NoImportModel


## Use a random seed for reproducibility

np.random.seed(637)

##

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
model_input = deepcopy(model_input_to_fit)
model_input.k_ext *= true_lam

true_density_expo = model_input.density_expo

# With the parameters chosen, we calculate Q_int:
household_population = HouseholdPopulation(
    composition_list, comp_dist, model_input)

pop_prev = 1e-2



no_imports = NoImportModel(4, 1)

base_rhs = UnloopedSEIRRateEquations(model_input, household_population, no_imports)
base_H0 = make_initial_condition_by_eigenvector(growth_rate,
                                                model_input,
                                                household_population,
                                                base_rhs,
                                                1e-1,
                                                0.0,
                                                False,
                                                3)
x0 = base_H0.T.dot(household_population.states[:, 1::4])

fixed_imports = FixedImportModel(4,
                                     1,
                                     base_rhs,
                                     x0)

rhs = UnloopedSEIRRateEquations(model_input, household_population, fixed_imports, sources="IMPORT")

H0 = np.zeros((household_population.total_size),)
all_sus = np.where(np.sum(rhs.states_exp_only + rhs.states_inf_only + rhs.states_rec_only, 1)<1e-1)[0]
one_inf = np.where((np.abs(np.sum(rhs.states_inf_only, 1) - 1)<1e-1) & (np.sum(rhs.states_exp_only + rhs.states_rec_only, 1)<1e-1))[0]
H0[all_sus] = comp_dist
# H0[one_inf] = 0.01 * comp_dist
S0 = H0.T.dot(household_population.states[:, ::4])
E0 = H0.T.dot(household_population.states[:, 1::4])
I0 = H0.T.dot(household_population.states[:, 2::4])
R0 = H0.T.dot(household_population.states[:, 3::4])
start_state = (1/model_input.ave_hh_size) * array([S0.sum(),
                                                   E0.sum(),
                                                   I0.sum(),
                                                   R0.sum()])

print("(S,E,I,R)(0)=",start_state)

## Now set up evaluation time points and solve system:

# New time at which we evaluate the infection
trange = np.arange(0,7*5,7) # Evaluate for 12 weeks
tspan = array([trange[0], trange[-1]])

# Solve:
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16, t_eval=trange)

T = solution.t
H = solution.y
S = H.T.dot(household_population.states[:, 0])
E = H.T.dot(household_population.states[:, 1])
I = H.T.dot(household_population.states[:, 2])
R = H.T.dot(household_population.states[:, 3])

##

data_list = [S/model_input.ave_hh_by_class,
    E/model_input.ave_hh_by_class,
    I/model_input.ave_hh_by_class,
    R/model_input.ave_hh_by_class]

lgd=['S','E','I','R']

fig, (axis) = subplots(1,1, sharex=True)

from matplotlib import cm
cmap = cm.get_cmap('tab20')
alpha = 0.5
for i in range(len(data_list)):
    axis.plot(
        T, data_list[i], label=lgd[i],
        color=cmap(i/len(data_list)), alpha=alpha)
axis.set_ylabel('Proportion of population')
axis.legend(ncol=1, bbox_to_anchor=(1,0.50))
plt.savefig('outputs/Trajectories')

H[H<0.0] = 0.0

#Make test data for single-household trajectories and calculate LLH

from numpy.random import choice

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, 1e-1, 0.0,False,3)
test_times = np.arange(7,7*5,7)
def generate_single_hh_test_data(test_times):
    Ht = deepcopy(H0)
    test_data = np.zeros((len(test_times),))
    for i in range(len(test_times)-1):
        tspan = (test_times[i], test_times[i+1])
        solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-16)
        T = solution.t
        H = solution.y
        state = choice(range(len(H[:, -1])), 1, p=H[:, -1]/sum(H[:, -1]))
        test_data[i] = rhs.states_inf_only[state]
        Ht *= 0
        Ht[state] = 1
    return(test_data)
sample_data = generate_single_hh_test_data(test_times)
print(sample_data)

## Log likelihood if we can only measure infecteds
# This is for test results from one household, you´ll need to adapt to multiple

def llh_from_test_data(test_data, test_ts, rhs, H0):
    total_sol_time = 0
    Ht = deepcopy(H0)
    llh = 0
    for i in range(len(test_times)-1):
        if i==0:
            start_time = 0
        else:
            start_time = test_times[i-1]
        tspan = (start_time, test_times[i])
        pre_sol = time.time()
        solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-16)
        total_sol_time += time.time()-pre_sol
        T = solution.t
        H = solution.y
        I = test_data[i]
        possible_states = np.where(np.abs(np.sum(rhs.states_inf_only,1)-I)<1e-1)[0]
        llh += np.log(np.sum(H[possible_states, -1]))
        Ht *= 0
        Ht[possible_states] = H[possible_states, -1]
    #print(total_sol_time)
    return(llh)

start_time = time.time()
llh_from_test_data(sample_data, test_times, rhs, H0)
print(time.time() - start_time)

## Log likelihood calculation but return probability trajectory

def llh_with_traj(test_data, test_times, rhs, H0):
    Ht = deepcopy(H0)
    H_all = np.atleast_2d(deepcopy(H0)).T
    t_all = np.array(0)
    llh = 0
    for i in range(len(test_times)):
        if i==0:
            start_time = 0
        else:
            start_time = test_times[i-1]
        tspan = (start_time, test_times[i])
        solution = solve_ivp(rhs, tspan, Ht, first_step=0.001, atol=1e-16)
        T = solution.t
        H = solution.y
        I = test_data[i]
        possible_states = np.where(np.abs(np.sum(rhs.states_inf_only,1)-I)<1e-1)[0]
        llh += np.log(np.sum(H[possible_states, -1]))
        Ht *= 0
        Ht[possible_states] = H[possible_states, -1]
        Ht = Ht/sum(Ht)
        H_all = np.hstack((H_all, H))
        t_all = np.hstack((t_all, T))
    return(llh, H_all, t_all)

llh, H_all, t_all = llh_with_traj(sample_data, test_times, rhs, H0)

##Plot

S = H_all.T.dot(household_population.states[:, ::4])
E = H_all.T.dot(household_population.states[:, 1::4])
I = H_all.T.dot(household_population.states[:, 2::4])
R = H_all.T.dot(household_population.states[:, 3::4])

data_list = [S/model_input.ave_hh_by_class,
    E/model_input.ave_hh_by_class,
    I/model_input.ave_hh_by_class,
    R/model_input.ave_hh_by_class]

lgd=['S','E','I','R', "Test data"]

fig, (axis) = subplots(1,1, sharex=True)

cmap = get_cmap('tab20')
alpha = 0.5
for i in range(len(data_list)):
    axis.plot(t_all,
        data_list[i], label=lgd[i],
        color=cmap(i/len(data_list)), alpha=alpha)
axis.plot(test_times, sample_data/model_input.ave_hh_by_class, marker=".", ls="", ms=20, label=lgd[-1])
axis.set_ylabel('Proportion of population')
axis.legend(ncol=1, bbox_to_anchor=(1,0.50))
plt.savefig('outputs/LLh with Trajectories')

## Now do multiple households

# Generate test data:
n_hh = 100
multi_hh_data = [generate_single_hh_test_data(test_times) for i in range(n_hh)]

# Option: Only include houses with at least one +ve
# multi_hh_data = [data for data in multi_hh_data if np.sum(data)>0]

#if isfile('outputs/inference-from-testing/synthetic-testing-data.pkl') is True:
#    with open('outputs/inference-from-testing/synthetic-testing-data.pkl', 'rb') as f:
#        multi_hh_data = load(f)
#else:
#    n_hh = 1000
#    multi_hh_data = [generate_single_hh_test_data(test_times) for i in range(n_hh)]
#
#    with open('outputs/inference-from-testing/synthetic-testing-data.pkl', 'wb') as f:
#        dump((multi_hh_data), f)

##Sample m households
#np.random.seed(42)# Setting a seed for reproducibility
#m = 100
#sample_idx = np.random.choice(range(len(multi_hh_data)), m)
#sampled_households = [multi_hh_data[i] for i in sample_idx]

#sampled_households

## Try a single likelihood calculation and see how long it takes to evaluate

start_time = time.time()
sampled_households_llh = sum(array([llh_from_test_data(sample_data, test_times, rhs, H0) for sample_data in sampled_households]))
#print("Single calculation takes", time.time() - start_time,"seconds.")

## At this point, we should be able to write a function taking parameters as input and calculating llh for those parameters.

def llh_from_pars(data, test_times, tau, lam):
    pre_hh_time = time.time()
    model_input = SEIRInput(SPEC, composition_list, comp_dist, print_ests=False)
    model_input.k_home = (tau / model_input.beta_int) * model_input.k_home
    model_input.k_ext = 0 * model_input.k_ext
    model_input.density_expo = true_density_expo
    # With the parameters chosen, we calculate Q_int:
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input)
    # lam goes in as arg for rate equations
    rhs = SEIRRateEquations(model_input, household_population, fixed_imports)
    H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, 1e-2, 0.0,False,3)
    tspan = (0.0, 365)
    #print(time.time() - pre_hh_time)
    return(sum(array([llh_from_test_data(data, test_times, rhs, H0) for data in sampled_households])))

llh_from_pars(sampled_households, test_times, 0.015, 2.75)

## Root finder approach

def get_tau_lam_mles(data,test_times,tau_0,lam_0):
    def f(params):
        tau = params[0]
        lam = params[1]
        return -llh_from_pars(data, test_times, tau, lam)

    mle=sp.optimize.minimize(f,[tau_0,lam_0],bounds=((0.005, 0.15),(2., 5.)))
    return mle

## Get MlE

start_time = time.time()
mle = get_tau_lam_mles(sampled_households, test_times, 0.02, 2.5)
end_time = time.time()
tau_hat, lam_hat = mle.x[0], mle.x[1]
print("Parameter estimation takes", (end_time - start_time)/60,"minutes.")
print("Optimised in", mle.nit, "iterations.")
print("MLE of tau=", tau_hat)
print("MLE of lam=", lam_hat)

# with open('outputs/inference-from-testing/synth-data-par-ests.pkl', 'wb') as f:
#     dump((tau_vals, lam_vals, llh_fixed_lam, llh_fixed_tau, tau_hat, lam_hat), f)

##Graphs ahead

tau_vals = arange(0.050, 0.18, 0.01)
lam_vals = arange(2., 4.0, 0.25)

llh_fixed_lam = np.zeros((len(tau_vals),))
for i in range(len(tau_vals)):
    #print("tau=",tau_vals[i])
    llh_fixed_lam[i] = llh_from_pars(sampled_households, test_times, tau_vals[i], true_lam)

llh_fixed_tau = np.zeros((len(lam_vals),))
for i in range(len(lam_vals)):
    #print("lam=",lam_vals[i])
    llh_fixed_tau[i] = llh_from_pars(sampled_households, test_times, model_input.beta_int, lam_vals[i])

## 1D graph

fig, (ax1, ax2) = subplots(1, 2, sharey=True)
ax1.plot(tau_vals, llh_fixed_lam, label='Log likelihood vs tau')
ax1.set_xlabel("tau")
ax1.set_ylabel("log likelihood")
ax1.plot(array(([model_input.beta_int, model_input.beta_int])), array(([llh_fixed_lam.min(), llh_fixed_lam.max()])), "k--", label='true value for lambda')
ax1.plot(array(([tau_hat, tau_hat])), array(([llh_fixed_lam.min(), llh_fixed_lam.max()])), "r--", label='estimated value for lambda')
ax1.legend(loc='lower right')
ax2.plot(lam_vals, llh_fixed_tau, label='Log likelihood vs lambda')
ax2.set_xlabel("lambda")
ax2.set_ylabel("log likelihood")
ax2.plot(array(([3., 3.])), array(([llh_fixed_tau.min(), llh_fixed_tau.max()])), "k--", label='true value for tau')
ax2.plot(array(([lam_hat, lam_hat])), array(([llh_fixed_tau.min(), llh_fixed_tau.max()])), "r--", label='estimated value for tau')
ax2.legend(loc='lower right')
plt.savefig('outputs/1D llh')  # Save as PNG file

## 2D LLH

tau_vals = arange(0.060, 0.105, 0.01)
lam_vals = arange(2., 4.0, 0.25)
llh_vals = np.zeros((len(tau_vals), len(lam_vals)))
for i in range(len(tau_vals)):
    for j in range(len(lam_vals)):
        llh_vals[i,j] = llh_from_pars(sampled_households, test_times, tau_vals[i], lam_vals[j])
        print("tau=",tau_vals[i],", lam=",lam_vals[j], ", llh[tau, lam]=", llh_vals[i, j])
with open('outputs/inference-from-testing/synth-data-gridded-par-ests.pkl', 'wb') as f:
 dump((tau_vals, lam_vals, llh_vals), f)

 ##plot

 fig, ax = subplots(1, 1)
 lam_inc = lam_vals[1] - lam_vals[0]
 lam_max = lam_vals[-1] + (lam_vals[1]-lam_vals[0])
 lam_range = lam_max - lam_vals[0]
 tau_inc = tau_vals[1] - tau_vals[0]
 tau_max = tau_vals[-1] + (tau_vals[1]-tau_vals[0])
 tau_range = tau_max - tau_vals[0]
 ax.imshow(llh_vals,
             origin='lower',
             extent=(lam_vals[0]-0.5*lam_inc,lam_max-0.5*lam_inc,tau_vals[0]-0.5*tau_inc,tau_max-0.5*tau_inc),
          aspect=lam_range/tau_range)
 ax.set_xlabel("tau")
 ax.set_ylabel("lambda")
 ax.plot([true_lam],
         [model_input.beta_int],
          marker=".",
          ms=20)
 plt.savefig('outputs/2D llh')  # Save as PNG file
