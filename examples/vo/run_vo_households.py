''' In this example we import the age-stratified testing data for Vo and run
each household as an independent example with exponential imports'''

from os.path import isfile
from pickle import load, dump
from numpy import arange, array, histogram, ones, prod, shape, sum, where, zeros
from pandas import read_csv
from scipy.integrate import solve_ivp
from scipy.special import binom
from model.preprocessing import VoInput
from model.specs import VO_SPEC
from model.common import ExpImportRateEquations, my_int, within_household_spread

''' In the following function, t0 is the time point from which we run the master
equations, start_date is the first day in the Vo data. This function assumes the
ages are in numeric form, i.e. 0th age class, 1st age class etc.'''
def get_test_probability(rhs,tests,symptoms,ages,states,composition,H0,t0,start_date):
    nn, tt = tests.shape
    tests[tests!=tests]=-1000 # Replace NaNs with a large negative value
    test_days = (start_date-t0) + where(tests>-1000)[1] # Find all locations where we have a test result

    H_start = H0
    t_start = start_date
    result_prob = []
    for t in test_days:
        solution = solve_ivp(rhs, (t_start,t), H_start, first_step=0.001)
        H = solution.y
        H_end = H[:,-1] # Get end state

        ''' Now calculate total positives and negatives by age class on this
        date'''
        pos_samples = (tests[:,t]==1)
        symp_ages = ages[symptoms[pos_samples]=='yes']
        asymp_ages = ages[symptoms[pos_samples]=='no']
        neg_ages = ages[tests[:,t]==0]

        min_symps_by_age = zeros((1,nn))
        for i in symp_ages:
            min_symps_by_age[i] += 1
        min_asymps_by_age = zeros((1,nn))
        for i in asymp_ages:
            min_asymps_by_age[i] +=1
        max_inf_by_age = composition
        for i in neg_ages:
            max_inf_by_age[i] -=1

        valid_locs = where(states[arange(2,-1,5)]>=min_symps_by_age & states[arange(3,-1,5)]>=min_asymps_by_age & states[arange(2,-1,5)]+states[arange(3,-1,5)]<=max_inf_by_age)
        result_prob.append = sum(H_end[valid_locs])
        H_start = 0*H_end
        H_start[valid_locs] = (1/result_prob[-1])*H_end[valid_locs]
        t_start = t

    return prod(result_prob)

model_input = VoInput(VO_SPEC)

with open('inputs/vo-testing-data.pkl','rb') as f:
    hh_tests, hh_ages, hh_symptoms = load(f)

hh_age_dict = {'00-10': 1, \
        '11-20': 2,
        '21-30': 3,
        '31-40': 4,
        '41-50': 5,
        '51-60': 6,
        '61-70': 7,
        '71-80': 8,
        '81-90': 9,
        '91+': 10}

composition_list = []
for h in hh_ages:
    for i in arange(len(h)):
        h[i] = hh_age_dict[h[i]]
    composition_list.append(histogram(h,bins=arange(0,12))[0])

test_prob = 1
epsilon = 0

sus = model_input.sigma
det = model_input.det
tau = model_input.tau
k_home = model_input.k_home
alpha = model_input.alpha
gamma = model_input.gamma


for i in range(len(composition_list)):

    composition = composition_list[i]
    tests = hh_tests[i]
    symptoms  = hh_symptoms[i]
    ages = hh_ages[i]

    system_size = 1
    for j in where(composition>0)[0]:
        system_size *= binom(
            composition[j] + 5 - 1,
            5 - 1)
    system_size = int(system_size)

    Q_int, states, inf_event_row, inf_event_col, inf_event_class \
        = within_household_spread(
            composition,
            sus,
            det,
            tau,
            k_home,
            alpha,
            gamma)

    which_composition = zeros(system_size, dtype=my_int)

    def exponential_import_model(t,H):
        rhs = ExpImportRateEquations(
        t,
        model_input,
        Q_int,
        composition,
        which_composition,
        states,
        inf_event_row,
        inf_event_col,
        inf_event_class,
        epsilon,
        det_profile,
        undet_profile,
        r)
        return rhs(t,H)
    H0 = zeros((50,))
    det_profile = det
    undet_profile = ones((9,))-det
    r = model_input.gamma*VO_SPEC['R0'] - model_input.gamma
    for i in range(10):
        H0[5*i] = composition[i]/sum(composition)
    t0 = 0
    start_date = 30

    composition = array([composition])
    this_prob = get_test_probability(exponential_import_model,tests,symptoms,ages,states,composition,H0,t0,start_date)
    test_prob = test_prob*this_prob

print('Likelihood of input parameters given data is ',test_prob,'.')
