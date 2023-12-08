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
from model.common import (sparse, my_int, build_state_matrix, RateEquations)
from model.imports import import_model_from_spec, NoImportModel
from model.preprocessing import ModelInput
from model.subsystems import (inf_events,
    progression_events, stratified_progression_events, subsystem_key)

LATENT_PERIOD = 0.2 * 5.8   # Time in days from infection to infectiousness
PRODROME_PERIOD = 0.8 * 5.8 # Time in days from infectiousness to symptom onset
SYMPTOM_PERIOD = 5          # Time in days from symptom onset to recovery

HOSTEL_VACC_SEIR_SPEC = {
    'compartmental_structure': 'SEIR', # This is which subsystem key to use
    'beta_int': 0.1,                     # Secondary inf probability
    'beta_ext': 0.1,                      # Household-level reproduction number
    'recovery_rate': 1 / (PRODROME_PERIOD +
                          SYMPTOM_PERIOD),           # Recovery rate
    'incubation_rate': 1 / LATENT_PERIOD,         # E->I incubation rate
    'sus': array([1,1]),          # Relative susceptibility by vacc. status
    'inf_scales': array([1,1]),    # Relative infectivity by vacc. status
    'density_expo': 1,             # Mixing density level
    'k_home': array([[1, 1], [1, 1]]),
    'k_all': array([[1, 1], [1, 1]])
}

class HostelSEIRInput(ModelInput):
    def __init__(self, spec, composition_list, composition_distribution):
        super().__init__(spec, composition_list, composition_distribution)

        self.expandables = ['sus',
                            'inf_scales']

        self.R_compartment = 3

        self.beta_int = spec['beta_int']
        self.beta_ext = spec['beta_ext']

        self.sus = spec['sus']
        self.inf_scales = spec['inf_scales']
        self.gamma = self.spec['recovery_rate']

        self.ave_trans = 1 / self.gamma

        self.prog_rates = array([self.gamma])

        self.k_home = self.beta_int * self.k_home

        ext_eig = max(eig(
            diag(self.sus).dot((1/spec['recovery_rate']) *
             (self.k_ext).dot(diag(self.inf_scales[0])))
            )[0])
        self.k_ext = self.beta_ext * self.k_ext / ext_eig

    @property
    def alpha(self):
        return self.spec['incubation_rate']
