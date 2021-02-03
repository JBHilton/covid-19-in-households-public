'''Various functions and classes that help build the model'''
from abc import ABC
from copy import copy, deepcopy
from numpy import (
        append, arange, around, array, cumsum, ones, ones_like, where,
        zeros, concatenate, vstack, identity, tile, hstack, prod, ix_,
        atleast_2d, diag)
from numpy.linalg import eig
from scipy.sparse import block_diag
from scipy.special import binom as binom_coeff
from scipy.stats import binom
from pandas import read_excel, read_csv
from tqdm import tqdm
from model.common import (
        within_household_spread, sparse, my_int, build_state_matrix)
from model.imports import import_model_from_spec, NoImportModel
from model.subsystems import subsystem_key


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


class HouseholdSubsystemSpec:
    '''Class to store composition subsystem specification to avoid code
    repetition'''
    def __init__(self, composition, no_compartments):
        self.composition = composition
        # This is an array of logicals telling you which classes are present in
        # each composition
        self.classes_present = composition.ravel() > 0
        self.class_indexes = where(self.classes_present)[0]
        self.system_sizes = array([
            binom_coeff(
                composition[class_index] + no_compartments - 1,
                no_compartments - 1)
            for class_index in self.class_indexes], dtype=my_int)
        self.system_sizes = self.system_sizes.ravel()
        self.total_size = prod(self.system_sizes)
        self.no_compartments = no_compartments

    @property
    def matrix_shape(self):
        return (self.total_size, self.total_size)


class HouseholdPopulation(ABC):
    def __init__(
            self,
            composition_list,
            composition_distribution,
            compartmental_structure,
            model_input,
            print_progress=True):
        '''This builds internal mixing matrix for entire system of
        age-structured households.'''

        self.composition_list = composition_list
        self.composition_distribution = composition_distribution
        self.subsystem_function = subsystem_key[compartmental_structure][0]
        self.num_of_epidemiological_compartments = subsystem_key[compartmental_structure][1]
        self.model_input = model_input

        # If the compositions include household size at the beginning, we
        # throw it away here. While we would expect to see some households
        # with equal numbers in age class 1 and all others combined, we should
        # not see it everywhere and so this is a safe way to check.
        # condition = max(abs(
        #    composition_list[:, 0] - composition_list[:, 1:].sum(axis=1)))
        # if condition == 0:
        #    composition_list = composition_list[:, 1:]

        # TODO: what if composition is given as list?
        self.no_compositions, self.no_risk_groups = composition_list.shape

        household_subsystem_specs = [
            HouseholdSubsystemSpec(c, self.num_of_epidemiological_compartments)
            for c in composition_list]

        # This is to remember mapping between states and household compositions
        self.which_composition = concatenate([
            i * ones(hsh.total_size, dtype=my_int)
            for i, hsh in enumerate(household_subsystem_specs)])

        # List of tuples describing model parts which need to be assembled into
        # a complete system. The derived classes will override the processing
        # function below.
        if print_progress:
            progress_bar = tqdm(
                household_subsystem_specs,
                desc='Building within-household transmission matrix')
        else:
            progress_bar = household_subsystem_specs
        model_parts = [
            self.subsystem_function(self,s)
            for s in progress_bar]

        self._assemble_system(household_subsystem_specs, model_parts)

    def _assemble_system(self, household_subsystem_specs, model_parts):
        # This is useful for placing blocks of system states
        cum_sizes = cumsum(array(
            [s.total_size for s in household_subsystem_specs]))
        total_size = cum_sizes[-1]
        self.Q_int = block_diag(
            [part[0] for part in model_parts],
            format='csc')
        self.Q_int.eliminate_zeros()
        self.offsets = concatenate(([0], cum_sizes))
        self.states = zeros((
            total_size,
            self.num_of_epidemiological_compartments * self.no_risk_groups))
        # self.states = concatenate([part[1] for part in model_parts])
        for i, part in enumerate(model_parts):
            class_list = household_subsystem_specs[i].class_indexes
            for j in range(len(class_list)):
                this_class = class_list[j]
                row_idx = slice(self.offsets[i], self.offsets[i+1])
                dst_col_idx = slice(
                    self.num_of_epidemiological_compartments*this_class,
                    self.num_of_epidemiological_compartments*(this_class+1))
                src_col_idx = slice(
                    self.num_of_epidemiological_compartments*j,
                    self.num_of_epidemiological_compartments*(j+1))
                self.states[row_idx, dst_col_idx] = part[1][:, src_col_idx]
        self.inf_event_row = concatenate([
            part[2] + self.offsets[i]
            for i, part in enumerate(model_parts)])
        self.inf_event_col = concatenate([
            part[3] + self.offsets[i]
            for i, part in enumerate(model_parts)])
        self.inf_event_class = concatenate([part[4] for part in model_parts])
        self.cum_sizes = cum_sizes
        self.system_sizes = array([
            hsh.total_size
            for hsh in household_subsystem_specs])

    @property
    def composition_by_state(self):
        return self.composition_list[self.which_composition, :]


class ConstantDetModel:
    '''This class acts a constant function representing profile of detected
    infections'''
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
        self.sus = rho / self.det

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

        self.density_expo = 1

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

        self.inf_scales = [ones(rho.shape),self.tau]

        self.inf_compartment_list = [1]
        self.no_inf_compartments = len(self.inf_compartment_list)

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

            self.sus * (
                (1.0/spec['recovery_rate'])
                * (self.k_home + self.epsilon * self.k_ext)
                + (1.0/spec['symp_onset_rate']) *
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
