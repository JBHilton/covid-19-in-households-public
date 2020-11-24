from numpy import (
    arange, array, atleast_2d, concatenate, copy, cumprod, diag, isnan, ix_,
    ones, prod, shape, sum, where, zeros)
from numpy import int64 as my_int
from numpy import exp, log
import pdb
from scipy.sparse import csc_matrix as sparse
from scipy.stats import multinomial
from model.preprocessing import ModelInput
from model.common import build_state_matrix, build_external_import_matrix_SEPIRQ
from model.imports import NoImportModel


def build_mixed_compositions_pairwise(composition_list,composition_distribution):

    no_comps = composition_list.shape[0]

    if composition_list.ndim==1:
        hh_dimension = 1
    else:
        hh_dimension = composition_list.shape[1]

    mixed_comp_list = zeros((no_comps**2,2*hh_dimension),dtype=my_int)
    mixed_comp_dist = zeros((no_comps**2,1))

    pairings = [[],[]]

    for hh1 in range(no_comps):
        for hh2 in range(no_comps):
            index = hh1*no_comps + hh2

            pairings[0].append(hh1)
            pairings[1].append(hh2)

            mixed_comp_list[index,:hh_dimension] = \
             composition_list[hh1,]
            mixed_comp_list[index,hh_dimension:] = \
             composition_list[hh2,]

            mixed_comp_dist[index] = \
             composition_distribution[hh1] * composition_distribution[hh2]

    return mixed_comp_list, mixed_comp_dist, hh_dimension, pairings

def pairwise_merged_initial_condition(H_unmerged,
                            unmerged_population,
                            merged_population,
                            hh_dimension,
                            pairings,
                            no_compartments=5):
    H0_len = sum(merged_population.system_sizes)
    H0 = zeros((H0_len,))
    reverse_prod = unmerged_population.reverse_prod
    index_vector_list = unmerged_population.index_vector
    which_composition = merged_population.which_composition
    merged_states = merged_population.states
    unmerged_states = unmerged_population.states

    for i in range(H0_len):
        comp_0 = pairings[0][which_composition[i]]
        comp_1 = pairings[1][which_composition[i]]
        state_0 = merged_states[i,:no_compartments]
        state_1 = merged_states[i,no_compartments:]
        index_vector_0 = index_vector_list[comp_0]
        index_vector_1 = index_vector_list[comp_1]
        index_0 = index_vector_0[
            state_0.dot(reverse_prod[comp_0]) + state_0[-1], 0]
        index_1 = index_vector_1[
            state_1.dot(reverse_prod[comp_1]) + state_1[-1], 0]
        H0[i] = H_unmerged[index_0] * H_unmerged[index_1]

    return H0

def pairwise_demerged_initial_condition(H_merged,
                            unmerged_population,
                            merged_population,
                            hh_dimension,
                            pairings,
                            no_compartments = 5):
    H0_len = sum(unmerged_population.system_sizes)
    H0 = zeros((H0_len,))
    reverse_prod = unmerged_population.reverse_prod
    index_vector_list = unmerged_population.index_vector
    which_composition = merged_population.which_composition
    merged_states = merged_population.states
    unmerged_states = unmerged_population.states

    for i in range(len(H_merged)):
        comp_0 = pairings[0][which_composition[i]]
        comp_1 = pairings[1][which_composition[i]]
        state_0 = merged_states[i,:no_compartments]
        state_1 = merged_states[i,no_compartments:]
        index_vector_0 = index_vector_list[comp_0]
        index_vector_1 = index_vector_list[comp_1]
        index_0 = index_vector_0[
            state_0.dot(reverse_prod[comp_0]) + state_0[-1], 0]
        index_1 = index_vector_1[
            state_1.dot(reverse_prod[comp_1]) + state_1[-1], 0]
        H0[index_0] += 0.5*H_merged[i]
        H0[index_1] += 0.5*H_merged[i]

    return H0


def build_mixed_compositions(composition_list,composition_distribution,no_hh=2):

    no_comps = composition_list.shape[0]

    if composition_list.ndim==1:
        hh_dimension = 1
    else:
        hh_dimension = composition_list.shape[1]

    no_mixed_comps = 0

    mixed_comp_list = []
    mixed_comp_dist = []

    hhi = no_hh*[0]
    pairings = []
    for pairing_index in range(no_hh):
        pairings.append([])

    def comp_iterator(depth, no_hh):
        if depth<no_hh:
            for i in range(hhi[depth-1],no_comps):
                hhi[depth] = i
                comp_iterator(depth+1, no_hh)
        else:
            index = 0
            for hh in range(no_hh):
                index +=  hhi[hh] * no_comps**(no_hh - 1 - hh)
                pairings[hh].append(hhi[hh])
            this_mix_comp = zeros((no_hh*hh_dimension,))
            hist = zeros((no_comps,))
            for hh in range(no_hh):
                this_mix_comp[hh*hh_dimension:(hh+1)*hh_dimension] = \
                 composition_list[hhi[hh],]
                hist[hhi[hh]] += 1
            this_mix_prob = multinomial.pmf(hist, n=no_hh, p=composition_distribution)
            mixed_comp_list.append(this_mix_comp)
            mixed_comp_dist.append(this_mix_prob)

    comp_iterator(0, no_hh)
    mixed_comp_list = array(mixed_comp_list, dtype=my_int)
    mixed_comp_dist = array(mixed_comp_dist)
    print(sum(mixed_comp_dist))
    return mixed_comp_list, mixed_comp_dist, hh_dimension, pairings

def my_multinomial(hist,n,p):
    log_prob = sum(log(arange(1,n+1)))
    for i in range(len(hist)):
        log_prob += hist[i] * log(p[i]) - sum(log(arange(1,hist[i]+1)))
    return exp(log_prob)

def merged_initial_condition(H_unmerged,
                            unmerged_population,
                            merged_population,
                            hh_dimension,
                            pairings,
                            no_hh=2,
                            no_compartments=5):
    H0_len = sum(merged_population.system_sizes)
    H0 = ones((H0_len,))
    reverse_prod = unmerged_population.reverse_prod
    index_vector_list = unmerged_population.index_vector
    which_composition = merged_population.which_composition
    merged_states = merged_population.states
    unmerged_states = unmerged_population.states

    for i in range(H0_len):
        hist = zeros(len(H_unmerged,))
        for hh in range(no_hh):
            comp = pairings[hh][which_composition[i]]
            state = merged_states[i,
                hh * hh_dimension * no_compartments :
                (hh+1) * hh_dimension * no_compartments]
            index_vector = index_vector_list[comp]
            index = index_vector[
                state.dot(reverse_prod[comp]) + state[-1], 0]
            hist[index] += 1
        H0[i] = multinomial.pmf(hist, n=no_hh, p=H_unmerged)

    return H0

def demerged_initial_condition(H_merged,
                            unmerged_population,
                            merged_population,
                            hh_dimension,
                            pairings,
                            no_hh =2,
                            no_compartments = 5):
    H0_len = sum(unmerged_population.system_sizes)
    H0 = zeros((H0_len,))
    reverse_prod = unmerged_population.reverse_prod
    index_vector_list = unmerged_population.index_vector
    which_composition = merged_population.which_composition
    merged_states = merged_population.states
    unmerged_states = unmerged_population.states

    for i in range(len(H_merged)):
        for hh in range(no_hh):
            comp = pairings[hh][which_composition[i]]
            state = merged_states[i,
                hh * hh_dimension * no_compartments :
                (hh+1) * hh_dimension * no_compartments]
            index_vector = index_vector_list[comp]
            index = index_vector[
                state.dot(reverse_prod[comp]) + state[-1], 0]
            H0[index] += (1 / no_hh) * H_merged[i]

    return H0

SINGLE_AGE_CLASS_SPEC = {
    # Interpretable parameters:
    'R_int': 1.1,                      # Within-household reproduction ratio
    'recovery_rate': 1/4,           # Recovery rate
    'incubation_rate': 1/1,         # E->P incubation rate
    'symp_onset_rate': 1/4,         # P->I prodromal to symptomatic rate
    'prodromal_trans_scaling':
     array([0.5]),          # Prodromal transmission intensity relative to full inf transmission
    'sus': array([1]),          # Relative susceptibility by age/vulnerability class
    'external_trans_scaling': 1.5 * 1/24,  # Relative intensity of external compared to internal contacts
    # We don't actually use these two mixing matrices, but we need them to make the abstract base class work
    'k_home': {
        'file_name': 'inputs/MUestimates_home_2.xlsx',
        'sheet_name':'United Kingdom of Great Britain'
    },
    'k_all': {
        'file_name': 'inputs/MUestimates_all_locations_2.xlsx',
        'sheet_name': 'United Kingdom of Great Britain'
    }
}

class DataObject():
    def __init__(self,thing):
        self.thing = thing

class SingleClassInput(ModelInput):
    '''TODO: add docstring'''
    def __init__(self, spec):
        super().__init__(spec)

        self.k_home = array(spec['R_int'], ndmin=2)
        self.k_ext = array(
            spec['R_int'] * spec['external_trans_scaling'], ndmin=2)
        self.tau = spec['prodromal_trans_scaling']
        self.sus = spec['sus']
        self.import_model = NoImportModel()

    @property
    def alpha_1(self):
        return self.spec['incubation_rate']

    @property
    def alpha_2(self):
        return self.spec['symp_onset_rate']

    @property
    def gamma(self):
        return self.spec['recovery_rate']

class MergedInput(ModelInput):
    '''TODO: add docstring'''
    def __init__(self, spec, no_hh, guest_trans_scaling):
        super().__init__(spec)
        self.k_home = \
            diag((1-guest_trans_scaling) * spec['R_int'] * ones((no_hh,))) + \
                    guest_trans_scaling * spec['R_int'] * ones((no_hh,no_hh))
        self.k_ext = \
         spec['R_int'] * spec['external_trans_scaling'] * ones((no_hh,no_hh))
        self.tau = spec['prodromal_trans_scaling'] * ones((no_hh,))
        self.sus = spec['sus'] * ones((no_hh,))
        self.import_model = NoImportModel()

    @property
    def alpha_1(self):
        return self.spec['incubation_rate']

    @property
    def alpha_2(self):
        return self.spec['symp_onset_rate']

    @property
    def gamma(self):
        return self.spec['recovery_rate']

def make_initial_condition(
        household_population,
        rhs,
        prev=1.0e-5):
    '''TODO: docstring'''
    fully_sus = where(
        rhs.states_sus_only.sum(axis=1)
        ==
        household_population.states.sum(axis=1))[0]
    i_is_one = where(
        (rhs.states_inf_only).sum(axis=1) == 1)[0]
    H0 = zeros(len(household_population.which_composition))
    x = household_population.composition_distribution[
            household_population.which_composition[i_is_one]]
    H0[i_is_one] = prev * x
    H0[fully_sus] = (1.0 - prev * sum(x)) \
            * household_population.composition_distribution
    return H0

def within_household_SEPIR(
        composition, model_input):
    '''Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    sus = model_input.sus
    tau = model_input.tau
    K_home = model_input.k_home
    alpha_1 = model_input.alpha_1
    alpha_2 = model_input.alpha_2
    gamma = model_input.gamma

    no_compartments = 5

    # Set of individuals actually present here
    classes_present = where(composition.ravel() > 0)[0]
    K_home = K_home[ix_(classes_present, classes_present)]
    sus = sus[classes_present]
    tau = tau[classes_present]
    r_home = atleast_2d(diag(sus).dot(K_home))

    states, total_size, reverse_prod, index_vector, rows = build_state_matrix(composition, classes_present, no_compartments)

    p_pos = 2 + no_compartments * arange(len(classes_present))
    i_pos = 3 + no_compartments * arange(len(classes_present))

    Q_int = sparse((total_size, total_size))
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)

    # Add events for each age class
    for i in range(len(classes_present)):
        s_present = where(states[:, no_compartments*i] > 0)[0]
        e_present = where(states[:, no_compartments*i+1] > 0)[0]
        p_present = where(states[:, no_compartments*i+2] > 0)[0]
        i_present = where(states[:, no_compartments*i+3] > 0)[0]

        # First do infection events
        inf_to = zeros(len(s_present), dtype=my_int)
        inf_rate = zeros(len(s_present))
        for k in range(len(s_present)):
            old_state = copy(states[s_present[k], :])
            inf_rate[k] = old_state[no_compartments*i] * (
                r_home[i, :].dot(
                    (old_state[i_pos] / composition[classes_present])
                    + (old_state[p_pos] / composition[classes_present]) * tau)) # tau is prodromal reduction
            new_state = old_state.copy()
            new_state[no_compartments*i] -= 1
            new_state[no_compartments*i + 1] += 1
            inf_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]
        Q_int += sparse(
            (inf_rate, (s_present, inf_to)),
            shape=(total_size, total_size))
        inf_event_row = concatenate((inf_event_row, s_present))
        inf_event_col = concatenate((inf_event_col, inf_to))
        inf_event_class = concatenate(
            (inf_event_class, classes_present[i]*ones((len(s_present)))))
        # input('Press enter to continue')
        # # disp('Infection events done')
        # # Now do exposure to prodromal
        inc_to = zeros(len(e_present), dtype=my_int)
        inc_rate = zeros(len(e_present))
        for k in range(len(e_present)):
            # First do detected
            old_state = copy(states[e_present[k], :])
            inc_rate[k] = alpha_1 * old_state[no_compartments*i+1]
            new_state = copy(old_state)
            new_state[no_compartments*i + 1] -= 1
            new_state[no_compartments*i + 2] += 1
            inc_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]

        Q_int += sparse(
            (inc_rate, (e_present, inc_to)),
            shape=(total_size, total_size))
        # # disp('Incubaion events done')
        # # Now do prodromal to infectious
        dev_to = zeros(len(p_present), dtype=my_int)
        dev_rate = zeros(len(p_present))
        for k in range(len(p_present)):
            # First do detected
            old_state = copy(states[p_present[k], :])
            dev_rate[k] = alpha_2 * old_state[no_compartments*i+2]
            new_state = copy(old_state)
            new_state[no_compartments*i + 2] -= 1
            new_state[no_compartments*i + 3] += 1
            dev_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]

        Q_int += sparse(
            (dev_rate, (p_present, dev_to)),
            shape=(total_size, total_size))

        # Now do recovery of detected cases
        rec_to = zeros(len(i_present), dtype=my_int)
        rec_rate = zeros(len(i_present))
        for k in range(len(i_present)):
            old_state = copy(states[i_present[k], :])
            rec_rate[k] = gamma * old_state[no_compartments*i+3]
            new_state = copy(old_state)
            new_state[no_compartments*i+3] -= 1
            new_state[no_compartments*i+4] += 1
            rec_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]
        Q_int += sparse(
            (rec_rate, (i_present, rec_to)),
            shape=(total_size, total_size))
        # disp('Recovery events from detecteds done')


    S = Q_int.sum(axis=1).getA().squeeze()
    Q_int += sparse((
        -S, (arange(total_size), arange(total_size))))
    return \
        Q_int, states, \
        array(inf_event_row, dtype=my_int, ndmin=1), \
        array(inf_event_col, dtype=my_int, ndmin=1), \
        array(inf_event_class, dtype=my_int, ndmin=1), \
        reverse_prod, \
        index_vector

class RateEquations:
    '''This class represents a functor for evaluating the rate equations for
    the model with no imports of infection from outside the population. The
    state of the class contains all essential variables'''
    # pylint: disable=invalid-name
    def __init__(self,
                 model_input,
                 household_population,
                 epsilon=1.0,        # TODO: this needs a better name
                 no_compartments=5
                 ):

        self.household_population = household_population
        self.epsilon = epsilon
        self.Q_int = household_population.Q_int
        # To define external mixing we need to set up the transmission
        # matrices.
        # Scale rows of contact matrix by
        self.inf_trans_matrix = diag(model_input.sus).dot(model_input.k_ext)
        # age-specific susceptibilities
        # Scale columns by asymptomatic reduction in transmission
        self.pro_trans_matrix = diag(model_input.sus).dot(
            model_input.k_ext.dot(diag(model_input.tau)))
        # This stores number in each age class by household
        self.composition_by_state = household_population.composition_by_state
        # ::5 gives columns corresponding to susceptible cases in each age
        # class in each state
        self.states_sus_only = household_population.states[:, ::no_compartments]

        self.s_present = where(self.states_sus_only.sum(axis=1) > 0)[0]
        # 2::5 gives columns corresponding to detected cases in each age class
        # in each state
        self.states_pro_only = household_population.states[:, 2::no_compartments]
        # 4:5:end gives columns corresponding to undetected cases in each age
        # class in each state
        self.states_inf_only = household_population.states[:, 3::no_compartments]
        self.epsilon = epsilon
        self.import_model = model_input.import_model

    def __call__(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''
        Q_ext_pro, Q_ext_inf = self.external_matrices(t, H)
        if (H<0).any():
            # pdb.set_trace()
            H[where(H<0)[0]]=0
        if isnan(H).any():
            pdb.set_trace()
        dH = (H.T * (self.Q_int + Q_ext_inf + Q_ext_pro)).T
        return dH

    def external_matrices(self, t, H):
        FOI_pro, FOI_inf = self.get_FOI_by_class(t, H)
        return build_external_import_matrix_SEPIRQ(
            self.household_population,
            FOI_pro,
            FOI_inf)

    def get_FOI_by_class(self, t, H):
        '''This calculates the age-stratified force-of-infection (FOI) on each
        household composition'''
        # Average number of each class by household
        denom = H.T.dot(self.composition_by_state)
        # Average detected infected by household in each class
        pro_by_class = zeros(shape(denom))
        # Only want to do states with positive denominator
        pro_by_class[denom > 0] = (
            H.T.dot(self.states_pro_only)[denom > 0]
            / denom[denom > 0]).squeeze()
        # Average undetected infected by household in each class
        inf_by_class = zeros(shape(denom))
        inf_by_class[denom > 0] = (
            H.T.dot(self.states_inf_only)[denom > 0]
            / denom[denom > 0]).squeeze()
        # This stores the rates of generating an infected of each class in
        # each state
        FOI_pro = self.states_sus_only.dot(
            diag(self.pro_trans_matrix.dot(
                self.epsilon * pro_by_class.T
                +
                self.import_model.detected(t))))
        FOI_inf = self.states_sus_only.dot(
            diag(self.inf_trans_matrix.dot(
                self.epsilon * inf_by_class.T
                +
                self.import_model.undetected(t))))

        return FOI_pro, FOI_inf
