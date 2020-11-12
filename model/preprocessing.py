'''Various functions and classes that help build the model'''
from abc import ABC
from copy import copy, deepcopy
from numpy import (
        append, arange, around, array, cumsum, ones, ones_like, where,
        zeros, concatenate, vstack, identity, tile, hstack)
from numpy.linalg import eig
from scipy.sparse import block_diag
from scipy.special import binom as binom_coeff
from scipy.stats import binom
from pandas import read_excel, read_csv
from tqdm import tqdm
from model.common import within_household_spread, sparse, my_int
from model.imports import import_model_from_spec, NoImportModel


def initialise_carehome(
        household_population,
        rhs,
        initial_presence):
    '''TODO: docstring'''
    initial_absence = household_population.composition_list - initial_presence

    # Starting state is one where total difference between S and initial
    # presence and total difference between E and initial absence are both zero
    starting_states = where((
        abs(rhs.states_sus_only - initial_presence).sum(axis=1) +
        abs(rhs.states_emp_only - initial_absence).sum(axis=1)) == 0)[0]

    H0 = zeros(len(household_population.which_composition))
    H0[starting_states] = household_population.composition_distribution
    return H0


def make_initial_condition(
        household_population,
        rhs,
        alpha=1.0e-5):
    '''TODO: docstring'''
    fully_sus = where(
        rhs.states_sus_only.sum(axis=1)
        ==
        household_population.states.sum(axis=1))[0]
    i_is_one = where(
        (rhs.states_det_only + rhs.states_undet_only).sum(axis=1) == 1)[0]
    H0 = zeros(len(household_population.which_composition))
    x = household_population.composition_distribution[
        household_population.which_composition[i_is_one]]
    H0[i_is_one] = alpha * x
    H0[fully_sus] = (1.0 - alpha * sum(x)) \
        * household_population.composition_distribution
    return H0


def make_initial_SEPIRQ_condition(
        household_population,
        rhs,
        prev=1.0e-5,
        seroprev=6e-2,
        AR=0.78):
    '''TODO: docstring'''
    fully_sus = where(
        rhs.states_sus_only.sum(axis=1)
        ==
        household_population.states.sum(axis=1))[0]
    already_visited = where(
        (rhs.states_rec_only.sum(axis=1)
            == around(AR*household_population.states.sum(axis=1)).astype(int)
            & ((rhs.states_sus_only + rhs.states_rec_only).sum(axis=1)
                == household_population.states.sum(axis=1)))
        & ((rhs.states_rec_only).sum(axis=1) > 0))[0]
    # This last condition is needed to make sure we don't include any fully
    # susceptible states
    i_is_one = where(
        ((rhs.states_inf_only).sum(axis=1) == 1)
        & ((
            rhs.states_sus_only+rhs.states_inf_only).sum(axis=1)
            ==
            household_population.states.sum(axis=1))
    )[0]
    ave_hh_size = sum(
        household_population.composition_distribution.T.dot(
            household_population.composition_list))
    H0 = zeros(len(household_population.which_composition))
    inf_comps = household_population.which_composition[i_is_one]
    x = array([])
    for state in i_is_one:
        x = append(
            x,
            (1/len(inf_comps == household_population.which_composition[state]))
            * household_population.composition_distribution[
                household_population.which_composition[state]])
        # base_comp_dist[household_population.which_composition[state]]-=x[-1]
    visited_comps = household_population.which_composition[already_visited]
    y = array([])
    for state in already_visited:
        y = append(
            y,
            (1/len(
                visited_comps
                == household_population.which_composition[state]))
            * household_population.composition_distribution[
                household_population.which_composition[state]])
        # base_comp_dist[household_population.which_composition[state]]-=y[-1]
    # y = household_population.composition_distribution[
    #     household_population.which_composition[already_visited]]
    H0[i_is_one] = ave_hh_size*(prev/sum(x)) * x
    H0[already_visited] = ave_hh_size*((seroprev/AR)/sum(y)) * y
    H0[fully_sus] = (1-sum(H0)) * household_population.composition_distribution

    return H0


def make_aggregator(coarse_bounds, fine_bounds):
    '''Construct a matrix that stores where each class in finer structure is
    in coarser structure'''
    return array([
        where(coarse_bounds <= fine_bounds[i])[0][-1]
        for i in range(len(fine_bounds) - 1)])


def aggregate_contact_matrix(k_fine, fine_bds, coarse_bds, pyramid):
    '''Aggregates an age-structured contact matrice to return the corresponding
    transmission matrix under a finer age structure.'''

    aggregator = make_aggregator(coarse_bds, fine_bds)

    # Prem et al. estimates cut off at 80, so we bundle all >75 year olds into
    # one class for consistency with these estimates:
    pyramid[len(fine_bds) - 1] = pyramid[len(fine_bds) - 1:].sum()
    pyramid = pyramid[:len(fine_bds) - 1]
    # Normalise to give proportions
    pyramid = pyramid / pyramid.sum()

    # sparse matrix defined here just splits pyramid into rows corresponding to
    # coarse boundaries, then summing each row gives aggregated pyramid
    row_cols = (aggregator, arange(len(aggregator)))
    # getA is necessary to convert numpy.matrix to numpy.array. The former is
    # deprecated and should disappear soon but scipy still returns.
    agg_pop_pyramid = sparse(
        (pyramid, row_cols)).sum(axis=1).getA().squeeze()

    rel_weights = pyramid / agg_pop_pyramid[aggregator]

    # Now define contact matrix with age classes from Li et al data
    pop_weight_matrix = sparse((rel_weights, row_cols))
    pop_no_weight = sparse((ones_like(aggregator), row_cols))

    return pop_weight_matrix * k_fine * pop_no_weight.T


def aggregate_vector_quantities(v_fine, fine_bds, coarse_bds, pyramid):
    '''Aggregates an age-structured contact matrice to return the corresponding
    transmission matrix under a finer age structure.'''

    aggregator = make_aggregator(coarse_bds, fine_bds)

    # The Prem et al. estimates cut off at 80, so we all >75 year olds into one
    # class for consistency with these estimates:
    pyramid[len(fine_bds) - 1] = sum(pyramid[len(fine_bds)-1:])
    pyramid = pyramid[:len(fine_bds) - 1]
    # Normalise to give proportions
    pyramid = pyramid / pyramid.sum()

    # sparse matrix defines here just splits pyramid into rows corresponding to
    # coarse boundaries, then summing each row gives aggregated pyramid
    row_cols = (aggregator, arange(len(aggregator)))
    agg_pop_pyramid = sparse(
        (pyramid, row_cols)).sum(axis=1).getA().squeeze()

    rel_weights = pyramid / agg_pop_pyramid[aggregator]

    # Now define contact matrix with age classes from Li et al data
    pop_weight_matrix = sparse((rel_weights, row_cols))

    return pop_weight_matrix * v_fine


def add_vulnerable_hh_members(
        composition_list, composition_distribution, vuln_prop):
    '''Create a version of the adult-child composition list and distribution
    which distinguishes between vulnerable and non-vulnerable adutls. Note that
    as written it is assuming only two age classes, with the second one being
    the one we divide by vulnerability.'''

    new_comp_list = copy(composition_list)
    new_comp_list = hstack((
        composition_list,
        zeros((len(composition_list), 1), type=my_int)))
    new_comp_dist = copy(composition_distribution)
    for comp_no in range(len(composition_list)):
        comp = composition_list[comp_no]
        if comp[1] > 0:
            new_comp_dist[comp_no] = \
                composition_distribution[comp_no] \
                * binom.pmf(0, comp[1], vuln_prop)
            for i in range(1, comp[1]+1):
                new_comp_list = vstack(
                    (new_comp_list, [comp[0], comp[1]-i, i]))
                prob = \
                    composition_distribution[comp_no] \
                    * binom.pmf(i, comp[1], vuln_prop)
                new_comp_dist.append(prob)
    return new_comp_list, new_comp_dist


class HouseholdPopulation:
    def __init__(
            self,
            composition_list,
            composition_distribution,
            model_input,
            build_function=within_household_spread,
            no_compartments=5,
            print_progress=True):
        '''This builds internal mixing matrix for entire system of
        age-structured households.'''

        self.composition_list = composition_list
        self.composition_distribution = composition_distribution

        # If the compositions include household size at the beginning, we
        # throw it away here. While we would expect to see some households
        # with equal numbers in age class 1 and all others combined, we should
        # not see it everywhere and so this is a safe way to check.
        # condition = max(abs(
        #    composition_list[:, 0] - composition_list[:, 1:].sum(axis=1)))
        # if condition == 0:
        #    composition_list = composition_list[:, 1:]

        no_types, no_classes = composition_list.shape

        # This is an array of logicals telling you which classes are present in
        # each composition
        classes_present = composition_list > 0

        system_sizes = ones(no_types, dtype=my_int)
        for i, _ in enumerate(system_sizes):
            for j in where(classes_present[i, :])[0]:
                system_sizes[i] *= binom_coeff(
                    composition_list[i, j] + no_compartments - 1,
                    no_compartments - 1)

        # This is useful for placing blocks of system states
        cum_sizes = cumsum(system_sizes)
        # Total size is sum of the sizes of each composition-system, because
        # we are considering a single household which can be in any one
        # composition
        total_size = cum_sizes[-1]
        # Stores list of (S,E,D,U,R)_a states for each composition
        states = zeros((total_size, no_compartments * no_classes), dtype=my_int)
        which_composition = zeros(total_size, dtype=my_int)

        # Initialise matrix of internal process by doing the first block
        which_composition[:system_sizes[0]] = zeros(system_sizes[0], dtype=my_int)
        Q_temp, states_temp, inf_event_row, inf_event_col, inf_event_class \
            = build_function(
                composition_list[0, :],
                model_input)
        Q_int = sparse(Q_temp)
        class_list = where(classes_present[0, :])[0]
        for j in range(len(class_list)):
            this_class = class_list[j]
            states[:system_sizes[0], no_compartments*this_class:no_compartments*(this_class+1)] = \
                states_temp[:, no_compartments*j:no_compartments*(j+1)]

        if print_progress:
            progress_bar = tqdm(
                range(1, no_types),
                desc='Building within-household transmission matrix')
        else:
            progress_bar = range(1, no_types)
        # NOTE: The way I do this loop is very wasteful, I'm making lots of
        # arrays which I'm overwriting with different sizes
        for i in progress_bar:
            # print('Processing {}/{}'.format(i, no_types))
            which_composition[cum_sizes[i-1]:cum_sizes[i]] = i * ones(
                system_sizes[i], dtype=my_int)
            Q_temp, states_temp, inf_temp_row, inf_temp_col, inf_temp_class \
                = build_function(
                    composition_list[i, :],
                    model_input)
            Q_int = block_diag((Q_int, Q_temp), format='csc')
            Q_int.eliminate_zeros()
            class_list = where(classes_present[i,:])[0]
            for j in range(len(class_list)):
                this_class = class_list[j]
                states[
                    cum_sizes[i-1]:cum_sizes[i],
                    no_compartments*this_class:no_compartments*(this_class+1)] = states_temp[:, no_compartments*j:no_compartments*(j+1)]

            inf_event_row = concatenate(
                (inf_event_row, cum_sizes[i-1] + inf_temp_row))
            inf_event_col = concatenate(
                (inf_event_col, cum_sizes[i-1] + inf_temp_col))
            inf_event_class = concatenate((inf_event_class, inf_temp_class))

        self.Q_int = Q_int
        self.states = states
        self.which_composition = which_composition
        self.system_sizes = system_sizes
        self.cum_sizes = cum_sizes
        self.inf_event_row = inf_event_row
        self.inf_event_col = inf_event_col
        self.inf_event_class = inf_event_class

    @property
    def composition_by_state(self):
        return self.composition_list[self.which_composition, :]


class ConstantDetModel:
    '''This is a profile of detected individuals'''
    def __init__(self, spec):
        self.constant = spec['constant']

    def __call__(self, rho):
        return self.constant * ones(rho.shape)


class ScaledDetModel:
    '''TODO: add docstring'''
    def __init__(self, spec):
        self.max_det = spec['max_det_fraction']

    def __call__(self, rho):
        return (self.max_det / rho.max()) * rho


def det_from_spec(spec):
    text_to_type = {
        'constant': ConstantDetModel,
        'scaled': ScaledDetModel,
    }
    return text_to_type[spec['det_model']['type']](spec['det_model'])


class ModelInput(ABC):
    def __init__(self, spec, header=None):
        self.spec = deepcopy(spec)
        self.k_home = read_excel(
            spec['k_home']['file_name'],
            sheet_name=spec['k_home']['sheet_name'],
            header=header).to_numpy()
        self.k_all = read_excel(
            spec['k_all']['file_name'],
            sheet_name=spec['k_all']['sheet_name'],
            header=header).to_numpy()

    @property
    def alpha(self):
        return self.spec['alpha']

    @property
    def gamma(self):
        return self.spec['gamma']


class StandardModelInput(ModelInput):
    '''TODO: add docstring'''
    def __init__(self, spec):
        super().__init__(spec)

        # Because we want 80 to be included as well.
        fine_bds = arange(0, 81, 5)
        self.coarse_bds = concatenate((fine_bds[:6], fine_bds[12:]))

        pop_pyramid = read_csv(
            spec['pop_pyramid_file_name'], index_col=0)
        pop_pyramid = (pop_pyramid['F'] + pop_pyramid['M']).to_numpy()

        self.k_home = aggregate_contact_matrix(
            self.k_home, fine_bds, self.coarse_bds, pop_pyramid)
        self.k_all = aggregate_contact_matrix(
            self.k_all, fine_bds, self.coarse_bds, pop_pyramid)
        self.k_ext = self.k_all - self.k_home

        # This is in ten year blocks
        rho = read_csv(
            spec['rho_file_name'], header=None).to_numpy().flatten()

        cdc_bds = arange(0, 81, 10)
        aggregator = make_aggregator(cdc_bds, fine_bds)

        # This is in five year blocks
        rho = sparse((
            rho[aggregator],
            (arange(len(aggregator)), [0]*len(aggregator))))

        rho = spec['recovery_rate'] * spec['R0'] * aggregate_vector_quantities(
            rho, fine_bds, self.coarse_bds, pop_pyramid).toarray().squeeze()

        det_model = det_from_spec(self.spec)
        # self.det = (0.9/max(rho)) * rho
        self.det = det_model(rho)
        self.tau = spec['asymp_trans_scaling'] * ones(rho.shape)
        self.sigma = rho / self.det

    @property
    def alpha(self):
        return self.spec['incubation_rate']

    @property
    def gamma(self):
        return self.spec['recovery_rate']


class TwoAgeModelInput(ModelInput):
    '''TODO: add docstring'''
    def __init__(self, spec):
        super().__init__(spec)

        fine_bds = arange(0, 81, 5)
        self.coarse_bds = array([0, 20])

        # pop_pyramid = read_csv(
        #     'inputs/United Kingdom-2019.csv', index_col=0)
        pop_pyramid = read_csv(
            spec['pop_pyramid_file_name'], index_col=0)
        pop_pyramid = (pop_pyramid['F'] + pop_pyramid['M']).to_numpy()

        self.k_home = aggregate_contact_matrix(
            self.k_home, fine_bds, self.coarse_bds, pop_pyramid)
        self.k_all = aggregate_contact_matrix(
            self.k_all, fine_bds, self.coarse_bds, pop_pyramid)
        self.k_ext = self.k_all - self.k_home

        # This is in ten year blocks
        rho = read_csv(
            spec['rho_file_name'], header=None).to_numpy().flatten()

        # This is in ten year blocks
        # rho = read_csv(
        #     'inputs/rho_estimate_cdc.csv', header=None).to_numpy().flatten()

        cdc_bds = arange(0, 81, 10)
        aggregator = make_aggregator(cdc_bds, fine_bds)

        # This is in five year blocks
        rho = sparse((
            rho[aggregator],
            (arange(len(aggregator)), [0]*len(aggregator))))

        rho = spec['recovery_rate'] * spec['R0'] * aggregate_vector_quantities(
            rho, fine_bds, self.coarse_bds, pop_pyramid).toarray().squeeze()

        det_model = det_from_spec(self.spec)
        # self.det = (0.9/max(rho)) * rho
        self.det = det_model(rho)
        self.tau = spec['asymp_trans_scaling'] * ones(rho.shape)
        self.sus = rho / self.det
        self.import_model = NoImportModel()

    @property
    def alpha(self):
        return self.spec['incubation_rate']

    @property
    def gamma(self):
        return self.spec['recovery_rate']


class VoInput(ModelInput):
    '''TODO: add docstring'''
    def __init__(self, spec):
        super().__init__(spec, header=0)
        fine_bds = arange(0, 96, 5)
        self.coarse_bds = arange(0, 96, 10)

        pop_pyramid = read_csv(
            spec['pop_pyramid_file_name'], index_col=0)
        pop_pyramid = (pop_pyramid['F'] + pop_pyramid['M']).to_numpy()

        '''We need to add an extra row to contact matrix to split 75+ class
        into 75-90 and 90+'''
        proportions_75_plus = append(
            pop_pyramid[15:18],
            sum(pop_pyramid[18:]))
        proportions_75_plus = proportions_75_plus/sum(proportions_75_plus)

        premultiplier = vstack((
            identity(16),
            tile(identity(16)[15, :], (3, 1))))
        postmultiplier = hstack((
            identity(16),
            zeros((16, 3))))
        postmultiplier[15, 15:] = proportions_75_plus

        k_home = (premultiplier.dot(self.k_home)).dot(postmultiplier)
        k_all = (premultiplier.dot(self.k_all)).dot(postmultiplier)

        self.k_home = aggregate_contact_matrix(
            k_home, fine_bds, self.coarse_bds, pop_pyramid)
        self.k_all = aggregate_contact_matrix(
            k_all, fine_bds, self.coarse_bds, pop_pyramid)
        self.k_ext = self.k_all - self.k_home

        no_age_classes = self.k_home.shape[0]

        # Now construct a matrix to map the age-stratified quantities from the
        # specs to the age boundaries used in the model.
        self.age_quant_bounds = spec['age_quant_bounds']
        age_quant_map = []
        min_now = 0
        for i in range(len(self.age_quant_bounds)):
            max_now = where(self.coarse_bds>self.age_quant_bounds[i])[0][0]
            # The additions in the expression below are list additions, not
            # array additions. We convert to an array after construction
            age_quant_map.append(
                [0] * min_now
                + [1] * (max_now - min_now)
                + [0] * (no_age_classes-max_now))
            min_now = max_now
        age_quant_map.append([0]*min_now + [1]*(no_age_classes - min_now))
        age_quant_map = array(age_quant_map)

        self.det = array(spec['symptom_prob']).dot(age_quant_map)
        self.tau = array(spec['asymp_trans_scaling']).dot(age_quant_map)
        self.sus = array(spec['sus']).dot(age_quant_map) 

        self.import_model = import_model_from_spec(spec, self.det)

    @property
    def alpha(self):
        return self.spec['incubation_rate']

    @property
    def gamma(self):
        return self.spec['recovery_rate']


class TwoAgeWithVulnerableInput:
    '''TODO: add docstring'''
    def __init__(self, spec):
        self.spec = deepcopy(spec)

        self.epsilon = spec['external_trans_scaling']

        self.vuln_prop = spec['vuln_prop']

        left_expander = vstack((
            identity(2),
            [0, 1]))
        # Add copy of bottom row - vulnerables behave identically to adults
        right_expander = array([
            [1, 0, 0],
            [0, 1-self.vuln_prop, self.vuln_prop]
        ])
        # Add copy of right row, scaled by vulnerables, and scale adult column
        # by non-vuln proportion
        k_home = read_excel(
            spec['k_home']['file_name'],
            sheet_name=spec['k_home']['sheet_name'],
            header=None).to_numpy()
        k_all = read_excel(
            spec['k_all']['file_name'],
            sheet_name=spec['k_all']['sheet_name'],
            header=None).to_numpy()

        fine_bds = arange(0, 81, 5)
        self.coarse_bds = array([0, 20])

        # pop_pyramid = read_csv(
        #     'inputs/United Kingdom-2019.csv', index_col=0)
        pop_pyramid = read_csv(
            spec['pop_pyramid_file_name'], index_col=0)
        pop_pyramid = (pop_pyramid['F'] + pop_pyramid['M']).to_numpy()

        self.k_home = aggregate_contact_matrix(
            k_home, fine_bds, self.coarse_bds, pop_pyramid)
        self.k_all = aggregate_contact_matrix(
            k_all, fine_bds, self.coarse_bds, pop_pyramid)
        self.k_ext = self.k_all - self.k_home

        self.k_home = left_expander.dot(self.k_home.dot(right_expander))
        self.k_all = left_expander.dot(self.k_all.dot(right_expander))
        self.k_ext = left_expander.dot(self.k_ext.dot(right_expander))

        self.sus = spec['sus']
        self.tau = spec['prodromal_trans_scaling']

        eigenvalue = max(eig(

            self.sus * ((1/spec['recovery_rate']) *
             (self.k_home + self.epsilon * self.k_ext) + \
            (1/spec['symp_onset_rate']) *
            (self.k_home + self.epsilon * self.k_ext) * self.tau)

            )[0])

        self.k_home = (spec['R0']/eigenvalue)*self.k_home
        self.k_all = (spec['R0']/eigenvalue)*self.k_all
        self.k_ext = (spec['R0']/eigenvalue)*self.k_ext

        self.k_ext[2, :] = 0 * self.k_ext[2, :]

    @property
    def alpha_1(self):
        return self.spec['incubation_rate']

    @property
    def alpha_2(self):
        return self.spec['symp_onset_rate']

    @property
    def gamma(self):
        return self.spec['recovery_rate']


class CareHomeInput(ModelInput):
    '''TODO: add docstring'''
    def __init__(self, spec):
        # We do not call super constructor as array are constructed manually.
        self.spec = deepcopy(spec)

        # Within-home contact matrix for patients and carers (full time and
        # agency)
        self.k_home = array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])
        # Contact matrix with other care homes - agency staff may work more
        # than one home
        self.k_ext = array([
            [0, 0, 0],
            [0, 0.01, 0.01],
            [0.5, 0.5, 0.5]])

        # Rate of contact with general outside population
        self.import_rate = array([0.5, 0.5, 0.5])

        self.sus = spec['sus']
        self.tau = spec['prodromal_trans_scaling']

        eigenvalue = max(eig(
            self.sus * ((1/spec['recovery_rate']) * (self.k_home) + \
            (1/spec['symp_onset_rate']) * (self.k_home) * self.tau)
            )[0])

        # Scaling below means R0 is the one defined in specs
        self.k_home = (spec['R_carehome']/eigenvalue) * self.k_home
        self.k_ext = self.k_ext

        self.mu = spec['empty_rate']
        self.mu_cov = spec['covid_mortality_rate']
        self.b = spec['refill_rate']
        self.epsilon = spec['inter_home_coupling']

    @property
    def alpha_1(self):
        return self.spec['incubation_rate']

    @property
    def alpha_2(self):
        return self.spec['symp_onset_rate']

    @property
    def gamma(self):
        return self.spec['recovery_rate']
