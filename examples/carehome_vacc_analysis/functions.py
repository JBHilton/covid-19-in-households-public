from abc import ABC
from copy import copy, deepcopy
from numpy import (
        append, arange, around, array, cumsum, log, ones, ones_like, where,
        zeros, concatenate, vstack, identity, tile, hstack, prod, ix_, shape,
        atleast_2d, diag)
from numpy.linalg import eig
from scipy.sparse import block_diag
from scipy.special import binom as binom_coeff
from scipy.stats import binom
from pandas import read_excel, read_csv
from tqdm import tqdm
from model.common import (sparse, my_int, build_state_matrix, RateEquations)
from model.imports import import_model_from_spec, NoImportModel
from model.subsystems import (inf_events,
    progression_events, stratified_progression_events, subsystem_key)

class CHModelInput(ABC):
    def __init__(self,
                spec,
                composition_list,
                composition_distribution,
                header=None):
        self.spec = deepcopy(spec)

        self.compartmental_structure = spec['compartmental_structure']
        self.inf_compartment_list = subsystem_key[self.compartmental_structure][2]
        self.no_inf_compartments = len(self.inf_compartment_list)

        self.k_home = spec['within_ch_contact']
        self.k_ext = spec['between_ch_contact']

        self.baseline_exit_rate = spec['baseline_exit_rate']

        self.density_expo = spec['density_expo']
        self.composition_list = composition_list
        self.composition_distribution = composition_distribution
        self.ave_hh_size = \
            composition_distribution.T.dot(
            composition_list.sum(axis=1))                                # Average household size
        self.dens_adj_ave_hh_size = \
            composition_distribution.T.dot((
            composition_list.sum(axis=1))**self.density_expo)                                # Average household size adjusted for density, needed to get internal transmission rate from secondary attack prob
        self.ave_hh_by_class = composition_distribution.T.dot(composition_list)



class SEMCRDInput(CHModelInput):
    def __init__(self, spec, composition_list, composition_distribution):
        super().__init__(spec, composition_list, composition_distribution)

        self.sus = spec['sus']
        self.inf_scales = [spec['mild_trans_scaling'],
                ones(shape(spec['mild_trans_scaling']))]

        self.covid_mortality_prob = spec['covid_mortality_prob']

        home_eig = max(eig(
            self.sus * ((1/spec['recovery_rate']) *
             (self.k_home) * spec['critical_inf_prob'] + \
            (1/spec['recovery_rate']) *
            (self.k_home ) * (1-spec['critical_inf_prob']) *
            spec['mild_trans_scaling'])
            )[0])
        ext_eig = max(eig(
            self.sus * ((1/spec['recovery_rate']) *
             (self.k_ext) + \
            (1/spec['recovery_rate']) *
            (self.k_ext ) * (1-spec['critical_inf_prob']) *
            spec['mild_trans_scaling'])
            )[0])

        R_int = - log(1 - spec['AR']) * self.dens_adj_ave_hh_size

        self.k_home = R_int * self.k_home / home_eig
        external_scale = spec['R*']/(self.ave_hh_size*spec['AR'])
        self.k_ext = external_scale * self.k_ext / ext_eig

    @property
    def alpha(self):
        return self.spec['incubation_rate']

    @property
    def crit_prob(self):
        return self.spec['critical_inf_prob']

    @property
    def gamma(self):
        return self.spec['recovery_rate']

def _semcrd_ch_subsystem(self, household_spec):
    '''This function processes a composition to create subsystems i.e.
    matrices and vectors describing all possible epdiemiological states
    for a given household composition
    Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    s_comp, e_comp, m_comp, c_comp, r_comp, d_comp = range(6)

    composition = household_spec.composition
    matrix_shape = household_spec.matrix_shape
    sus = self.model_input.sus
    K_home = self.model_input.k_home
    inf_scales = copy(self.model_input.inf_scales)
    alpha = self.model_input.alpha
    crit_prob = self.model_input.crit_prob
    gamma = self.model_input.gamma
    covid_mortality_prob = self.model_input.covid_mortality_prob
    baseline_exit_rate = self.model_input.baseline_exit_rate
    density_expo = self.model_input.density_expo

    # Set of individuals actually present here
    class_idx = household_spec.class_indexes

    K_home = K_home[ix_(class_idx, class_idx)]
    sus = sus[class_idx]
    r_home = atleast_2d(diag(sus).dot(K_home))
    crit_prob = crit_prob[class_idx]
    for i in range(len(inf_scales)):
        inf_scales[i] = inf_scales[i][class_idx]
    covid_mortality_prob = covid_mortality_prob[class_idx]
    baseline_exit_rate = baseline_exit_rate[class_idx]

    states, \
        reverse_prod, \
        index_vector, \
        rows = build_state_matrix(household_spec)

    Q_int = sparse(household_spec.matrix_shape,)
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)

    Q_int, inf_event_row, inf_event_col, inf_event_class = inf_events(s_comp,
                e_comp,
                [m_comp, c_comp],
                inf_scales,
                r_home,
                density_expo,
                6,
                composition,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int,
                inf_event_row,
                inf_event_col,
                inf_event_class)
    Q_int = stratified_progression_events(e_comp,
                    m_comp,
                    alpha*(1-crit_prob),
                    6,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = stratified_progression_events(e_comp,
                    c_comp,
                    alpha*crit_prob,
                    6,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = progression_events(m_comp,
                    r_comp,
                    gamma,
                    6,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = stratified_progression_events(c_comp,
                    r_comp,
                    gamma*(1-covid_mortality_prob),
                    6,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = stratified_progression_events(c_comp,
                    d_comp,
                    gamma*covid_mortality_prob,
                    6,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)

    '''Now do the non-disease related exit (to D) events - we can just cycle
    over the numerical index of the compartments since all compartments progress
    identically. This cycle only goes to range 5 since we assume no D->D
    events.'''
    for i in range(5):
        Q_int = stratified_progression_events(i,
                        d_comp,
                        baseline_exit_rate,
                        6,
                        states,
                        index_vector,
                        reverse_prod,
                        class_idx,
                        matrix_shape,
                        Q_int)

    Q_int = stratified_progression_events(d_comp,
                    s_comp,
                    baseline_exit_rate,
                    6,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)

    S = Q_int.sum(axis=1).getA().squeeze()
    Q_int += sparse((
        -S, (
            arange(household_spec.total_size),
            arange(household_spec.total_size)
        )))
    return tuple((
        Q_int,
        states,
        array(inf_event_row, dtype=my_int, ndmin=1),
        array(inf_event_col, dtype=my_int, ndmin=1),
        array(inf_event_class, dtype=my_int, ndmin=1)))

def combine_household_populations(hh_pop_list, weightings):
    combined_pop = hh_pop_list[0]
    combined_pop.composition_distribution = weightings[0] * \
                                        combined_pop.composition_distribution
    for i in range(1, len(hh_pop_list)):
        hh_pop = hh_pop_list[i]
        combined_pop.Q_int = block_diag((combined_pop.Q_int,hh_pop.Q_int))
        combined_pop.which_composition =  append(combined_pop.which_composition,
                                                hh_pop.which_composition)
        combined_pop.inf_event_row = append(combined_pop.inf_event_row,
                                            hh_pop.inf_event_row)
        combined_pop.inf_event_col = append(combined_pop.inf_event_col,
                                            hh_pop.inf_event_col)
        combined_pop.inf_event_class = append(combined_pop.inf_event_class,
                                            hh_pop.inf_event_class)
        combined_pop.cum_sizes = append(combined_pop.cum_sizes,
                                hh_pop.cum_sizes + combined_pop.cum_sizes[-1])
        combined_pop.system_sizes = append(combined_pop.system_sizes,
                                        hh_pop.system_sizes)
        combined_pop.offsets = append(combined_pop.offsets,
                                    hh_pop.offsets + combined_pop.offsets[-1])
        combined_pop.states = vstack((combined_pop.states, hh_pop.states))
        combined_pop.composition_list = vstack((combined_pop.composition_list,
                                                hh_pop.composition_list))
        combined_pop.composition_distribution = append(
                        combined_pop.composition_distribution,
                        weightings[i] * hh_pop.composition_distribution)

    return combined_pop


class SEMCRDRateEquations(RateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]
    @property
    def states_mild_only(self):
        return self.household_population.states[:, 2::self.no_compartments]
    @property
    def states_crit_only(self):
        return self.household_population.states[:, 3::self.no_compartments]
    @property
    def states_rec_only(self):
        return self.household_population.states[:, 4::self.no_compartments]
    @property
    def states_emp_only(self):
        return self.household_population.states[:, 5::self.no_compartments]

'''Note that the following function isn't currently working!'''
def initialise_carehome(household_population):
    total_size = len(household_population.which_composition)
    Q_int_aug = sparsevstack([household_population.Q_int, ones(total_size).T])
    print(Q_int_aug.shape)

    b = zeros(total_size+1)
    b[total_size] = 1
    x0 = zeros(total_size)
    x0[0]=1

    x, istop, itn, r1norm = lsqr(Q_int_aug,b,x0=x0)[:4] # Finds leading eigenvector of Q_int
    H0 = x/sum(x)
    print(max(H0),min(H0))
    print(sum(H0))

    print('Initial infection is',H0.T.dot(household_population.states[:,2::6]))

subsystem_key['carehome_SEMCRD'] = [_semcrd_ch_subsystem,6,[2,3]]

THREE_CLASS_CH_EPI_SPEC = {
    'compartmental_structure': 'carehome_SEMCRD', # This is which subsystem key to use
    'AR': 0.80,                     # Secondary attack probability
    'R*': 1.1,                      # Household-level reproduction number
    'recovery_rate': 1/4,           # Recovery rate
    'incubation_rate': 1/5,         # E->M or C incubation rate
    'critical_inf_prob': array([1.0,1.0,1.0]),      # Probability of going E->C
    'mild_trans_scaling':
     array([1.0,1.0,1.0]),          # Prodromal transmission intensity relative to full inf transmission
    'sus': array([1,1,1]),          # Relative susceptibility by age/vulnerability class
    'density_expo' : 1.0, # "Cauchemez parameter"
    'covid_mortality_prob': array([0.2,0.0,0.0]) # Mortality rate CONDITIONED ON BEING CRITICAL ALREADY!!!
}

THREE_CLASS_CH_SPEC = {
    'k_home': {                                                 # File location for UK within-household contact matrix
        'file_name': 'inputs/MUestimates_home_2.xlsx',
        'sheet_name':'United Kingdom of Great Britain'
    },
    'k_all': {                                                  # File location for UK pop-level contact matrix
        'file_name': 'inputs/MUestimates_all_locations_2.xlsx',
        'sheet_name': 'United Kingdom of Great Britain'
    },
    'pop_pyramid_file_name': 'inputs/United Kingdom-2019.csv',   # File location for UK age pyramid
    'fine_bds' : arange(0,81,5),                                # Boundaries used in pyramid/contact data
    'coarse_bds' : array([0,20]),                               # Desired boundaries for model population
    'within_ch_contact': array([[1,1,1], [1,1,1], [1,1,1]]),
    'between_ch_contact': array([[0,0,1], [0,0,0], [1,0,1]]),
    'baseline_exit_rate': array([0, 0, 0])
}

''' The following function is very hacky - it only works for the specific
example we're doing in parallel_sweep.py. '''
def simple_initialisation(
        household_population,
        rhs,
        state_list = [(2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0),
                  (1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0)],
        weightings = [0.5,0.5]):

    '''TODO: docstring'''
    H0 = zeros(len(household_population.which_composition))
    for i in range(len(state_list)):
        loc = where((household_population.states == state_list[i]).all(axis=1))[0]
        H0[loc] = weightings[i]
        print('I found the requested starting state at loc and states[loc,:]=',
                household_population.states[loc,:])
    return H0
