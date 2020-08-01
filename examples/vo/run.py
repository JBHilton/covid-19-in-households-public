''' In this example we import the age-stratified testing data for Vo and run
each household as an independent example with exponential imports'''

from os.path import isfile
from pickle import load, dump
from numpy import (
        arange, array, exp, histogram, log, ones, prod, sum,
        where, zeros)
from pandas import isnull, read_csv, unique
from scipy.integrate import solve_ivp
from model.preprocessing import VoInput, HouseholdPopulation
from model.specs import VO_SPEC
from model.common import RateEquations
from model.imports import ExponentialImportModel

def get_symptoms_by_test(tests, symptoms):
    symp_locs = []
    asym_locs = []
    for i in range(len(tests)):
        if tests[i]==1:
            if symptoms[i]=='yes':
                symp_locs.append(i)
            elif symptoms[i]=='no':
                asym_locs.append(i)
    return symp_locs, asym_locs


def get_test_probability(
        tests, symptoms, ages, states, composition, H0, t0, start_date):
    '''In the following function, t0 is the time point from which we run the
    master equations, start_date is the first day in the Vo data. This function
    assumes the ages are in numeric form, i.e. 0th age class, 1st age class
    etc.'''

    nn, tt = tests.shape
    # Find all locations where we have a test result
    test_days = unique(start_date-t0 + where(~isnull(tests))[1])

    H_start = H0
    t_start = start_date
    result_prob = []
    for t in test_days:
        solution = solve_ivp(rhs, (t_start,t), H_start,first_step=0.001,atol=1e-9)
        H = solution.y
        H_end = H[:,-1] # Get end state
        print('H=',H_end)
        # print('After initial run sum(H_end)=',sum(H_end))
        # H_end[H_end<0]=0
        # print('Throwing away bad values, sum(H_end)=',sum(H_end))

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
        print('Number of valid states is ',len(valid_locs))
        print('Number of invalid states is ',len(H_end)-len(valid_locs))
        print('Valid probability is ',sum(H_end[valid_locs]))
        print('Invalid probability is ',sum(H_end[where(1-prod(states[:,2::5]>=min_symps_by_age,axis=1)*(prod(states[:,3::5]>=min_asymps_by_age,axis=1)*prod(states[:,2::5]+states[:,3::5]<=max_inf_by_age,axis=1)))[0]]))
        result_prob.append(sum(H_end[valid_locs]))
        H_start = 0*H_start
        H_start[valid_locs] = H_end[valid_locs]
        H_start = H_start/sum(H_start)
        print('New starting probability is ',sum(H_start))
        print('H_start = ',H_start)
        t_start = t

    return sum(log((result_prob)))

model_input = VoInput(VO_SPEC)

if isfile('vo-testing-data.pkl') is True:
    with open('vo-testing-data.pkl','rb') as f:
        hh_tests, hh_ages, hh_symptoms = load(f)
else:
    df = read_csv(
        'examples/vo/vo_data.csv',
        dtype={
            'household_id': str})
    hcol = df.household_id.values
    hhids = unique(df.household_id)
    testday_indices = range(104, 123)
    hh_tests = []
    hh_ages = []
    hh_symptoms = []
    for hid in hhids:
        dfh = df[df.household_id == hid]
        tests = dfh.iloc[:, testday_indices].values
        aa = dfh.iloc[:,2].values
        ss = dfh.symptomatic.values
        tests[tests=='Neg'] = 0
        tests[tests=='Pos'] = 1
        hh_tests.append(tests.astype(float))
        hh_ages.append(aa)
        hh_symptoms.append(ss)
    with open('vo-testing-data.pkl','wb') as f:
        dump((hh_tests,hh_ages,hh_symptoms),f)

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

det_profile = 1e-5*det
undet_profile = 1e-5*ones((10,))-det
r = 1e-2
exponential_importation = ExponentialImportModel(r, det_profile, undet_profile)

for i in range(10):
    if sum(composition_list[i])<10: # There are a few huge households which we skip
        print('Now doing household ',i+1,' of ',len(composition_list))
        composition = composition_list[i]
        tests = hh_tests[i]
        symptoms  = hh_symptoms[i]
        ages = hh_ages[i]

        household_population = HouseholdPopulation(
            array([composition]), array([1]), model_input)

        # print(composition)
        # print(shape(composition))
        # print(shape(states))

        H0 = zeros((household_population.system_sizes[0],))
        states_sus_only = household_population.states[:,::5]
        fully_sus = where(states_sus_only.sum(axis=1) == household_population.states.sum(axis=1))[0]
        H0[fully_sus] = 1
        rhs = RateEquations(
            model_input,
            household_population,
            exponential_importation,
            epsilon)
        t0 = 0
        start_date = 30

        composition = array([composition])
        this_prob = get_test_probability(
            tests,
            symptoms,
            ages,
            household_population.states,
            composition,
            H0,
            t0,
            start_date)
        test_prob.append(this_prob)

with open('testing-probabilities.pkl','wb') as f:
    dump((test_prob),f)
print('Likelihood of input parameters given data is ',exp(sum(test_prob)),'.')
print('Likelihoods are ',exp(test_prob),'.')
