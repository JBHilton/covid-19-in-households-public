# Special definitions we need for the hostels vaccination analysis

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
from model.common import (sparse, my_int, RateEquations)
from model.imports import import_model_from_spec, NoImportModel
from model.preprocessing import ModelInput
from model.subsystems import (inf_events,
    progression_events, stratified_progression_events, subsystem_key)

LATENT_PERIOD = 0.2 * 5.8   # Time in days from infection to infectiousness
PRODROME_PERIOD = 0.8 * 5.8 # Time in days from infectiousness to symptom onset
SYMPTOM_PERIOD = 5          # Time in days from symptom onset to recovery

VACC_EFF = 0.5
VACC_INF_RED = 0.5

HOSTEL_VACC_SEIR_SPEC = {
    'compartmental_structure': 'SEIR', # This is which subsystem key to use
    'beta_int': 0.1,                     # Secondary inf probability
    'beta_ext': 0.1,                      # Household-level reproduction number
    'recovery_rate': 1 / (PRODROME_PERIOD +
                          SYMPTOM_PERIOD),           # Recovery rate
    'incubation_rate': 1 / LATENT_PERIOD,         # E->I incubation rate
    'sus': array([VACC_EFF,1]),          # Relative susceptibility by vacc. status
    'inf_scales': array([VACC_INF_RED,1]),    # Relative infectivity by vacc. status
    'density_expo': 1,             # Mixing density level
    'k_home': array([[1, 1], [1, 1]]),
    'k_ext': array([[1, 1], [1, 1]]),
}

class HostelSEIRInput(ModelInput):
    def __init__(self, spec, composition_list, composition_distribution):

        self.spec = deepcopy(spec)

        self.compartmental_structure = spec['compartmental_structure']
        self.no_compartments = subsystem_key[self.compartmental_structure][1]
        self.inf_compartment_list = \
            subsystem_key[self.compartmental_structure][2]
        self.no_inf_compartments = \
            len(self.inf_compartment_list)

        self.new_case_compartment = \
            subsystem_key[self.compartmental_structure][3]

        self.expandables = ['sus',
                            'inf_scales']

        self.R_compartment = 3

        self.beta_int = spec['beta_int']
        self.beta_ext = spec['beta_ext']

        self.sus = spec['sus']
        self.inf_scales = [spec['inf_scales']] # Needs to be in array since some of simulation code cycles over inf. compartments
        self.gamma = self.spec['recovery_rate']

        self.ave_trans = 1 / self.gamma

        self.prog_rates = array([self.gamma])

        self.k_home = spec['k_home']
        self.k_home = self.beta_int * self.k_home

        self.k_ext = spec['k_ext']

        self.density_expo = spec['density_expo']

        self.composition_list = composition_list
        self.composition_distribution = composition_distribution

    @property
    def hh_size_list(self):
        return self.composition_list.sum(axis=1)
    @property
    def ave_hh_size(self):
        # Average household size
        return self.composition_distribution.T.dot(self.hh_size_list)
    @property
    def max_hh_size(self):
        # Average household size
        return self.hh_size_list.max()
    @property
    def dens_adj_ave_hh_size(self):
          # Average household size adjusted for density,
          # needed to get internal transmission rate from secondary inf prob
        return self.composition_distribution.T.dot(
                                        (self.hh_size_list)**self.density_expo)
    @property
    def ave_hh_by_class(self):
        return self.composition_distribution.T.dot(self.composition_list)

    @property
    def ave_contact_dur(self):
        k_home_scaled = diag(self.ave_hh_by_class).dot(self.k_home)
        return eig(k_home_scaled)[0].max()

    @property
    def alpha(self):
        return self.spec['incubation_rate']
