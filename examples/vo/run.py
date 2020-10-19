'''In this example we import the age-stratified testing data for Vo and run
each household as an independent example with exponential imports'''
from os.path import isfile
from pickle import load, dump
from numpy import (
        arange, array, exp, histogram, isnan, log, ones, prod, where, zeros)
from pandas import read_csv, unique
from scipy.integrate import solve_ivp
from model.preprocessing import VoInput, HouseholdPopulation
from model.specs import VO_SPEC
from model.common import RateEquations
from model.imports import ExponentialImportModel

AGE_GROUPS = [
    '00-10', '11-20', '21-30', '31-40', '41-50',
    '51-60', '61-70', '71-80', '81-90', '91+']
HH_AGE_DICT = {
    ag: iag for iag, ag in enumerate(AGE_GROUPS)}


def get_symptoms_by_test(test, symptoms):
    '''
    Returns time indexes for symptomatic and asymptomatic tests.

        Parameters:
            test (array): array of tested individuals within the household.
            symptoms (list): a yes/no list for symptoms in invididauls of the
                household.
        Returns:
            symp_locs
            asym_locs
    '''
    symp_locs = []
    asym_locs = []
    for i in range(len(test)):
        if test[i] == 1:
            if symptoms[i] == 'yes':
                symp_locs.append(i)
            elif symptoms[i] == 'no':
                asym_locs.append(i)
    return symp_locs, asym_locs


def get_test_probability(
        tests, symptoms, ages, states, composition, H0, t0, start_date, rhs):
    '''In the following function, t0 is the time point from which we run the
    master equations, start_date is the first day in the Vo data. This function
    assumes the ages are in numeric form (zero-based numbering), i.e. index 0
    for first age class, index 1 for second age class etc.'''

    nn, tt = tests.shape
    # Find all locations where we have a test result
    test_days = unique(start_date - t0 + where(~isnan(tests))[1])

    H_start = H0
    t_start = start_date
    result_prob = []
    for t in test_days:
        solution = solve_ivp(
            rhs, (t_start, t), H_start, first_step=1e-9, atol=1e-12)
        H = solution.y
        H_end = H[:, -1]  # Get end state
        # print('H = ', H_end)
        # print('After initial run sum(H_end)=',sum(H_end))
        # H_end[H_end<0]=0
        # print('Throwing away bad values, sum(H_end)=',sum(H_end))

        # Now calculate total positives and negatives by age class on this date
        symp_locs, asym_locs = get_symptoms_by_test(
            tests[:, t - (start_date - t0)], symptoms)
        symp_ages = ages[symp_locs]
        # print('symp_ages=', symp_ages)
        asym_ages = ages[asym_locs]
        # print('asym_ages=',asymp_ages)
        neg_ages = ages[tests[:, t-(start_date-t0)] == 0]

        min_symps_by_age = zeros((len(AGE_GROUPS),))
        for i in symp_ages:
            min_symps_by_age[i] += 1
        min_asyms_by_age = zeros((len(AGE_GROUPS),))
        # Index is needed because composition is inside an array
        for i in asym_ages:
            min_asyms_by_age[i] += 1
        max_inf_by_age = list(composition[0])
        # print(composition)
        for i in neg_ages:
            max_inf_by_age[i] -= 1
        # print('min_symps=',min_symps_by_age)
        # print('min_asymps_by_age=',min_asymps_by_age)
        # print('max_inf_by_age=',max_inf_by_age)

        # The next line finds all the lines consistent with the min/max number
        # of infecteds by multiplying truth values for each comparision along
        # the rows
        mask = array(
            prod(states[:, 2::5] >= min_symps_by_age, axis=1)
            * prod(states[:, 3::5] >= min_asyms_by_age, axis=1)
            * prod(states[:, 2::5] + states[:, 3::5] <= max_inf_by_age, axis=1),
            dtype=bool)
        # print('Number of valid states is ', len(valid_locs))
        # print('Number of invalid states is ', len(H_end) - len(valid_locs))
        # print('Valid probability is ', H_end[mask].sum())
        # print('Invalid probability is ', H_end[~mask].sum())
        result_prob.append(H_end[mask].sum())
        H_start = 0.0 * H_start
        H_start[mask] = H_end[mask]
        H_start = H_start / H_start.sum()
        # print('New starting probability is ', H_start.sum())
        # print('H_start = ', H_start)
        t_start = t
    return sum(log((result_prob)))


class SerialLikelihoodCalculation:
    def __init__(self):
        self.model_input = VoInput(VO_SPEC)

        if isfile('vo-testing-data.pkl') is True:
            with open('vo-testing-data.pkl', 'rb') as f:
                self.tests, self.ages, self.symptoms = load(f)
        else:
            df = read_csv(
                'examples/vo/vo_data.csv',
                dtype={
                    'household_id': str})
            hcol = df.household_id.values
            hhids = unique(df.household_id)
            testday_columns = range(104, 123)
            self.tests = []
            self.ages = []
            self.symptoms = []
            for hid in hhids:
                dfh = df[df.household_id == hid]
                tests = dfh.iloc[:, testday_columns].values
                age_groups = dfh.age_group.values
                ss = dfh.symptomatic.values
                tests[tests == 'Neg'] = 0
                tests[tests == 'Pos'] = 1
                self.tests.append(tests.astype(float))
                self.ages.append(
                    array([HH_AGE_DICT[ag] for ag in age_groups]))
                self.symptoms.append(ss)
            with open('vo-testing-data.pkl', 'wb') as f:
                dump((self.tests, self.ages, self.symptoms), f)

        self.composition_list = [
            histogram(h, bins=arange(0, len(AGE_GROUPS)+1))[0]
            for h in self.ages]

    def __call__(self, r):
        test_prob = []
        epsilon = 0.0

        sus = self.model_input.sigma
        det = self.model_input.det

        det_profile = 1e-5*det
        undet_profile = 1e-5*(ones((10,))-det)
        exponential_importation = ExponentialImportModel(
            r, det_profile, undet_profile)

        for i in range(1):
            # There are a few huge households which we skip
            if sum(self.composition_list[i]) < 10:
                print('Processing household {0} of {1}'.format(
                    i+1, len(self.composition_list)))
                composition = self.composition_list[i]
                tests = self.tests[i]
                symptoms = self.symptoms[i]
                ages = self.ages[i]

                household_population = HouseholdPopulation(
                    array([composition]), array([1.0]), self.model_input)

                # print(composition)
                # print(shape(composition))
                # print(shape(states))

                H0 = zeros((household_population.system_sizes[0],))
                states_sus_only = household_population.states[:, ::5]
                fully_sus = where(
                    states_sus_only.sum(axis=1)
                    ==
                    household_population.states.sum(axis=1))[0]
                H0[fully_sus] = 1
                rhs = RateEquations(
                    self.model_input,
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
                    start_date,
                    rhs)
                test_prob.append(this_prob)

        # with open('testing-probabilities.pkl', 'wb') as f:
            # dump((test_prob), f)

        # print('Likelihood of input parameters given data is {0}.'.format(
        #    exp(sum(test_prob))))
        # print('Likelihoods are {0}.'.format(exp(test_prob)))
        return exp(test_prob)


if __name__ == '__main__':
    calculator = SerialLikelihoodCalculation()
    print(calculator(1e-2))
