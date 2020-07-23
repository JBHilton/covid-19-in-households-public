''' In this example we import the age-stratified testing data for Vo and run
each household as an independent example with exponential imports'''

from os.path import isfile
from pickle import load, dump
from numpy import arange, array, exp, histogram, log, ones, prod, shape, sum, where, zeros
from pandas import read_csv
from scipy.integrate import solve_ivp
from scipy.special import binom
from model.preprocessing import VoInput, build_household_population
from model.specs import VO_SPEC
from model.common import ExpImportRateEquations, my_int, get_symptoms_by_test

''' In the following function, t0 is the time point from which we run the master
equations, start_date is the first day in the Vo data. This function assumes the
ages are in numeric form, i.e. 0th age class, 1st age class etc.'''
def get_test_probability(rhs,tests,symptoms,ages,states,composition,H0,t0,start_date):

    nn, tt = tests.shape
    tests[tests!=tests]=-1000 # Replace NaNs with a large negative value
    test_days = (start_date-t0) + where(tests>-1000)[1] # Find all locations where we have a test result
    test_days = list(set(test_days)) # get unique days

    H_start = H0
    t_start = start_date
    result_prob = []
    for t in test_days:
        solution = solve_ivp(rhs, (t_start,t), H_start,first_step=0.001)
        H = solution.y
        H_end = H[:,-1] # Get end state
        # print('H=',H_end)
        # print('After initial run sum(H_end)=',sum(H_end))

        ''' Now calculate total positives and negatives by age class on this
        date'''

        symp_locs, asymp_locs = get_symptoms_by_test(tests[:,t-(start_date-t0)],symptoms)
        symp_ages = ages[symp_locs]
        # print('symp_ages=',symp_ages)
        asymp_ages = ages[asymp_locs]
        # print('asym_ages=',asymp_ages)
        neg_ages = ages[tests[:,t-(start_date-t0)]==0]
        # print('neg_ages=',neg_ages)
        # print('ages=',ages)

        min_symps_by_age = zeros((10,))
        for i in symp_ages:
            min_symps_by_age[i] += 1
        min_asymps_by_age = zeros((10,))
        for i in asymp_ages:
            min_asymps_by_age[i] +=1
        max_inf_by_age = list(composition[0]) # Index is needed because composition is inside an array
        # print(composition)
        for i in neg_ages:
            max_inf_by_age[i] -=1
        # print('min_symps=',min_symps_by_age)
        # print('min_asymps_by_age=',min_asymps_by_age)
        # print('max_inf_by_age=',max_inf_by_age)

        # The next line finds all the lines consistent with the min/max number of infecteds by multiplying truth values for each comparision along the rows
        valid_locs = where(prod(states[:,2::5]>=min_symps_by_age,axis=1)*(prod(states[:,3::5]>=min_asymps_by_age,axis=1)*prod(states[:,2::5]+states[:,3::5]<=max_inf_by_age,axis=1)))[0]
        # print('Number of valid locations is ',len(valid_locs))
        # print('Numbre of invalid locations is ',len(H_end)-len(valid_locs))
        # print('Valid probability is ',sum(H_end[valid_locs]))
        # print('Invalid probability is ',sum(H_end[where(1-prod(states[:,2::5]>=min_symps_by_age,axis=1)*(prod(states[:,3::5]>=min_asymps_by_age,axis=1)*prod(states[:,2::5]+states[:,3::5]<=max_inf_by_age,axis=1)))[0]]))
        result_prob.append(sum(H_end[valid_locs]))
        H_start = 0*H_end
        H_start[valid_locs] = (1/result_prob[-1])*H_end[valid_locs]
        # print('New starting probability is ',sum(H_start))
        t_start = t

    return sum(log((result_prob)))

model_input = VoInput(VO_SPEC)

with open('inputs/vo-testing-data.pkl','rb') as f:
    hh_tests, hh_ages, hh_symptoms = load(f)

hh_age_dict = {'00-10': 0, \
        '11-20': 1,
        '21-30': 2,
        '31-40': 3,
        '41-50': 4,
        '51-60': 5,
        '61-70': 6,
        '71-80': 7,
        '81-90': 8,
        '91+': 9}

composition_list = []
for h in hh_ages:
    for i in arange(len(h)):
        h[i] = hh_age_dict[h[i]]
    composition_list.append(histogram(h,bins=arange(0,11))[0])

test_prob = []
epsilon = 0

sus = model_input.sigma
det = model_input.det
tau = model_input.tau
k_home = model_input.k_home
alpha = model_input.alpha
gamma = model_input.gamma


for i in range(10):
    if sum(composition_list[i])<10: # There are a few huge households which we skip
        print('Now doing household ',i+1,' of ',len(composition_list))
        composition = composition_list[i]
        tests = hh_tests[i]
        symptoms  = hh_symptoms[i]
        ages = hh_ages[i]

        Q_int, states, which_composition, \
                system_sizes, cum_sizes, \
                inf_event_row, inf_event_col, inf_event_class \
            = build_household_population(array([composition]), model_input)

        # print(composition)
        # print(shape(composition))
        # print(shape(states))

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
        H0 = zeros((system_sizes[0],))
        states_sus_only = states[:,::5]
        fully_sus = where(states_sus_only.sum(axis=1) == states.sum(axis=1))[0]
        H0[fully_sus] = 1
        det_profile = det
        undet_profile = ones((10,))-det
        r = model_input.gamma*VO_SPEC['R0'] - model_input.gamma
        t0 = 0
        start_date = 30

        composition = array([composition])
        this_prob = get_test_probability(exponential_import_model,tests,symptoms,ages,states,composition,H0,t0,start_date)
        test_prob.append(this_prob)

with open('testing-probabilities.pkl','wb') as f:
    dump((test_prob),f)
print('Likelihood of input parameters given data is ',exp(sum(test_prob)),'.')
print('test_prob=',test_prob)
