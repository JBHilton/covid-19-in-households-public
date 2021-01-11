from copy import deepcopy
from numpy import (
    append, arange, around, array, atleast_2d, concatenate, copy, cumprod, cumsum, delete, diag, hstack, isnan, ix_,
    ones, prod, shape, sum, hstack, unique, where, zeros)
from numpy import int64 as my_int
from numpy import exp, log
from numpy.linalg import eig
from pandas import read_csv
import pdb
from scipy.sparse import csc_matrix as sparse
from scipy.special import factorial
from scipy.stats import multinomial
from model.preprocessing import aggregate_contact_matrix, ModelInput
from model.common import build_state_matrix, build_external_import_matrix_SEPIRQ
from model.imports import NoImportModel



SEPIR_SPEC = {
    # Interpretable parameters:
    'AR': 0.45,                      # Reproduction number
    'recovery_rate': 1/4,           # Recovery rate
    'incubation_rate': 1/5,         # E->P incubation rate
    'symp_onset_rate': 1/3,         # P->I prodromal to symptomatic rate
    'prodromal_trans_scaling':
     array([0.5,0.5]),          # Prodromal transmission intensity relative to full inf transmission
    'sus': array([1,1]),          # Relative susceptibility by age/vulnerability class
    'density_expo' : 0.5, # "Cauchemez parameter"
    'vuln_prop': 2.2/56,            # Total proportion of adults who are shielding
    'k_home': {
        'file_name': 'inputs/MUestimates_home_2.xlsx',
        'sheet_name':'United Kingdom of Great Britain'
    },
    'k_all': {
        'file_name': 'inputs/MUestimates_all_locations_2.xlsx',
        'sheet_name': 'United Kingdom of Great Britain'
    },
    'pop_pyramid_file_name': 'inputs/United Kingdom-2019.csv'
}

class DataObject():
    def __init__(self,thing):
        self.thing = thing

class SEPIRInput(ModelInput):
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
        self.tau = spec['prodromal_trans_scaling']
        self.sus = spec['sus']

        home_eig = max(eig(

            self.sus * ((1/spec['recovery_rate']) *
             (self.k_home) + \
            (1/spec['symp_onset_rate']) *
            (self.k_home ) * self.tau)

            )[0])
        ext_eig = max(eig(

            self.sus * ((1/spec['recovery_rate']) *
             (self.k_ext) + \
            (1/spec['symp_onset_rate']) *
            (self.k_ext ) * self.tau)

            )[0])

        R_int = - log(1 - spec['AR'])

        self.k_home = R_int * self.k_home / home_eig
        print('Internal eigenvalue is',max(eig(

            self.sus * ((1/spec['recovery_rate']) *
             (self.k_home) + \
            (1/spec['symp_onset_rate']) *
            (self.k_home ) * self.tau)

            )[0]))
        self.k_ext = R_int * self.k_ext / home_eig
        print('External eigenvalue is',max(eig(

            self.sus * ((1/spec['recovery_rate']) *
             (self.k_ext) + \
            (1/spec['symp_onset_rate']) *
            (self.k_ext ) * self.tau)

            )[0]))
        self.density_expo = spec['density_expo']
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

def make_initial_condition_with_recovereds(
        household_population,
        rhs,
        prev=1.0e-2,
        seroprev=5.6e-2,
        AR=1.0):
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
    density_expo = model_input.density_expo

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
                    (old_state[i_pos] / composition[classes_present] ** density_expo)
                    + (old_state[p_pos] / composition[classes_present] ** density_expo) * tau)) # tau is prodromal reduction
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
        self.states_rec_only = household_population.states[:, 4::no_compartments]
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
