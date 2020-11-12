from os.path import isfile
from copy import deepcopy
from pickle import load, dump
from numpy import (
        arange, array, histogram, isnan, log, ones, prod, where, zeros)
from scipy.integrate import solve_ivp
from pandas import read_csv, unique
from model.specs import VO_SPEC
from model.preprocessing import VoInput, HouseholdPopulation
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


class HouseholdTestData:
    def __init__(self, ages, symptoms, tests):
        self.ages = ages
        self.symptoms = symptoms
        self.tests = tests
        # Convert the age information into composition data
        self.composition = histogram(
            ages, bins=arange(0, len(AGE_GROUPS)+1))[0]

    def get_test_probability(self, rhs, H0, test_start_date, t_end):
        '''In the following function, t0 is the time point from which we run
        the master equations, start_date is the first day in the Vo data. This
        function assumes the ages are in numeric form (zero-based numbering),
        i.e. index 0 for first age class, index 1 for second age class etc.'''

        nn, tt = self.tests.shape
        # Find all locations where we have a test result
        test_days = unique(
            test_start_date + where(~isnan(self.tests))[1])

        H_start = H0
        t_start = 0.0
        result_prob = []
        test_days.sort()

        for t_next_test in test_days:
            solution = solve_ivp(
                rhs,
                (t_start, t_next_test),
                H_start,
                first_step=1e-9,
                atol=1e-12)
            H = solution.y
            H_end = H[:, -1]
            # print('H = ', H_end)
            # print('After initial run sum(H_end)=',sum(H_end))
            # H_end[H_end<0]=0
            # print('Throwing away bad values, sum(H_end)=',sum(H_end))

            # Now calculate total positives and negatives by age class on this
            # date
            symp_locs, asym_locs = get_symptoms_by_test(
                self.tests[:, int(t_next_test) - test_start_date],
                self.symptoms)
            symp_ages = self.ages[symp_locs]
            # print('symp_ages=', symp_ages)
            asym_ages = self.ages[asym_locs]
            # print('asym_ages=',asymp_ages)
            neg_ages = self.ages[
                self.tests[:, int(t_next_test) - test_start_date] == 0]

            min_symps_by_age = zeros((len(AGE_GROUPS),))
            for i in symp_ages:
                min_symps_by_age[i] += 1
            min_asyms_by_age = zeros((len(AGE_GROUPS),))
            # Index is needed because composition is inside an array
            for i in asym_ages:
                min_asyms_by_age[i] += 1
            max_inf_by_age = list(self.composition)
            # print(composition)
            for i in neg_ages:
                max_inf_by_age[i] -= 1
            # print('min_symps=',min_symps_by_age)
            # print('min_asymps_by_age=',min_asymps_by_age)
            # print('max_inf_by_age=',max_inf_by_age)

            # The next line finds all the lines consistent with the min/max
            # number of infecteds by multiplying truth values for each
            # comparision along the rows

            states = rhs.household_population.states
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
            if H_start.sum() == 0.0:
                raise ValueError
            H_start = H_start / H_start.sum()
            # print('New starting probability is ', H_start.sum())
            # print('H_start = ', H_start)
            t_start = t_next_test
        return sum(log((result_prob)))

    def plot2file(self, file_name):
        from matplotlib.pyplot import figure
        from re import findall
        from numpy import argsort
        fig = figure()
        ax = fig.gca()
        left_bounds = [
            int(findall('[0-9]+', AGE_GROUPS[a])[0]) for a in self.ages]
        index = argsort(left_bounds)
        for i, timeline in enumerate(self.tests[index, :]):
            for j, d in enumerate(timeline):
                if d == 0:
                    ax.plot(j, i, marker='o', color='b')
                elif d == 1:
                    ax.plot(j, i, marker='+', color='r')
                else:
                    ax.plot(j, i, marker='.', c='k')

        ax.set_yticks(arange(len(self.ages)))
        ax.set_yticklabels([AGE_GROUPS[a] for a in self.ages[index]])
        fig.savefig(file_name, dpi=300)


class LikelihoodCalculation:
    ''''''
    def __init__(self):
        if isfile('vo-testing-data.pkl') is True:
            with open('vo-testing-data.pkl', 'rb') as f:
                tests, ages, symptoms = load(f)
        else:
            df = read_csv(
                'examples/vo/vo_data.csv',
                dtype={
                    'household_id': str})
            hhids = unique(df.household_id)
            testday_columns = range(104, 123)
            tests = []
            ages = []
            symptoms = []
            for hid in hhids:
                dfh = df[df.household_id == hid]
                test = dfh.iloc[:, testday_columns].values
                age_groups = dfh.age_group.values
                ss = dfh.symptomatic.values
                test[test == 'Neg'] = 0
                test[test == 'Pos'] = 1
                tests.append(test.astype(float))
                ages.append(
                    array([HH_AGE_DICT[ag] for ag in age_groups]))
                symptoms.append(ss)
            with open('vo-testing-data.pkl', 'wb') as f:
                dump((tests, ages, symptoms), f)

        self.households = [
            HouseholdTestData(a, s, t)
            for a, s, t in zip(ages, symptoms, tests)]
        self.t_first_test = 14
        self.t_end = self.t_first_test + 20.0
        self.epsilon = 0.0

    def __call__(self, r, a):
        spec = deepcopy(VO_SPEC)
        spec['external_importation']['exponent'] = r
        spec['external_importation']['alpha'] = a
        self.model_input = VoInput(spec)
        unfiltered_probabilities = self._process_households()
        probabilities = [
            prob for prob in unfiltered_probabilities
            if prob is not None]
        return sum(probabilities)

    def compute_probability(self, household):
        # There are a few huge households which we skip
        household_is_acceptable = (
            (household.composition.sum() < 8)
            and
            (household.composition.sum() == household.tests.shape[0]))
        try:
            if household_is_acceptable:
                return self._compute_probability_for_valid(household)
            else:
                return None
        except ValueError:
            return None

    def _initialize(self, household):
        household_population = HouseholdPopulation(
            array([household.composition]),
            array([1.0]),
            self.model_input,
            print_progress=False)

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
            self.epsilon)
        return H0, rhs

    def _compute_probability_for_valid(self, household):
        H0, rhs = self._initialize(household)
        return household.get_test_probability(
            rhs,
            H0,
            self.t_first_test,
            self.t_end)
