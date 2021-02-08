'''In this module we should place simple tests for the models.'''
from numpy import arange, array, ones, where, zeros
from numpy.linalg import norm
from numpy.testing import assert_almost_equal
from pandas import read_csv
from model.imports import NoImportModel
from model.preprocessing import aggregate_vector_quantities, det_from_spec, make_aggregator, HouseholdPopulation, ModelInput
from model.common import SEDURRateEquations, sparse

TEST_SPEC = {
    # Interpretable parameters:
    'compartmental_structure': 'SEDUR',
    'R0': 2.4,                      # Reproduction number
    'recovery_rate': 0.5,                   # Mean infectious period
    'incubation_rate': 0.2,                   # Incubation period
    'asymp_trans_scaling': 0.0,                     # Asymptomatic transmission intensity relative to symptomatic rate
    'det_model': {
        'type': 'scaled',           # 'constant' and 'scaled' are the two options
        'max_det_fraction': 0.9     # Cap for detected cases (90%)
    },
    # These represent input files for the model. We can make it more flexible
    # in the future, but certain structure of input files must be assumed.
    # Check ModelInput class in model/preprocessing.py to see what assumptions
    # are used now.
    'k_home': {
        'file_name': 'inputs/MUestimates_home_2.xlsx',
        'sheet_name':'United Kingdom of Great Britain'
    },
    'k_all': {
        'file_name': 'inputs/MUestimates_all_locations_2.xlsx',
        'sheet_name': 'United Kingdom of Great Britain'
    },
    'pop_pyramid_file_name': 'inputs/United Kingdom-2019.csv',   # File location for UK age pyramid
    'fine_bds' : arange(0,81,5),                                # Boundaries used in pyramid/contact data
    'coarse_bds' : array([0,20]),                               # Desired boundaries for model population
    'rho_file_name': 'inputs/rho_estimate_cdc.csv',
    'density_expo': 1
}

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

class TestModelInput(ModelInput):
    '''TODO: add docstring'''
    def __init__(self, spec, composition_list, composition_distribution):
        super().__init__(spec, composition_list, composition_distribution)

        # This is in ten year blocks
        rho = read_csv(
            spec['rho_file_name'], header=None).to_numpy().flatten()

        # This is in ten year blocks
        # rho = read_csv(
        #     'inputs/rho_estimate_cdc.csv', header=None).to_numpy().flatten()

        cdc_bds = arange(0, 81, 10)
        aggregator = make_aggregator(cdc_bds, self.fine_bds)

        # This is in five year blocks
        rho = sparse((
            rho[aggregator],
            (arange(len(aggregator)), [0]*len(aggregator))))

        rho = spec['recovery_rate'] * spec['R0'] * aggregate_vector_quantities(
            rho, self.fine_bds, self.coarse_bds, self.pop_pyramid).toarray().squeeze()

        det_model = det_from_spec(self.spec)
        # self.det = (0.9/max(rho)) * rho
        self.det = det_model(rho)
        self.tau = spec['asymp_trans_scaling'] * ones(rho.shape)
        self.sus = rho / self.det
        self.import_model = NoImportModel(5,2)

        self.inf_scales = [ones(rho.shape),self.tau]

    @property
    def alpha(self):
        return self.spec['incubation_rate']

    @property
    def gamma(self):
        return self.spec['recovery_rate']

def test_simple():
    '''Test a simple two age model'''
    composition_list = array(
        [[0, 1], [0, 2], [1, 1], [1, 2], [2, 1], [2, 2]])
    # Proportion of households which are in each composition
    composition_distribution = array(
        [0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
    model_input = TestModelInput(TEST_SPEC, composition_list, composition_distribution)
    household_population = HouseholdPopulation(
        composition_list, composition_distribution, model_input)

    rhs = SEDURRateEquations(
        model_input,
        household_population,
        NoImportModel(2,5))

    H0 = make_initial_condition(household_population, rhs)
    dH = rhs(0.0, H0)
    assert_almost_equal(7.776182090170313e-05, norm(dH))
    assert_almost_equal(2.7801365751414517e-05, max(dH))
    assert_almost_equal(-3.270492048199998e-05, min(dH))
