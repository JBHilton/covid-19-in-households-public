'''This sets up and runs a simple system which is low-dimensional enough to
do locally'''
from os.path import isfile
import numpy
from numpy import arange, array, where
from numpy.random import rand
from time import time as get_time
from scipy.integrate import solve_ivp
import ode
from model.preprocessing import (
    CareHomeInput, HouseholdPopulation, initialise_carehome)
from model.specs import CAREHOME_SPEC
from model.common import CareHomeRateEquations, within_carehome_system
from model.imports import CareHomeImportModel
from pickle import load, dump
# pylint: disable=invalid-name

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
    carehome_population = HouseholdPopulation(
        composition_list, comp_dist, model_input, within_carehome_system, 6)
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
    
Prob = numpy.empty(shape=(carehome_size,len(time)), dtype = object)

for i in range(carehome_size):
    for j in range(len(time)):
        Prob[i,j] = sum(H[where(carehome_population.states[:,3]==i)].T[j])

with open('carehome_results.pkl','wb') as f:
    dump((time, carehome_size, H,P,I,Prob),f)
