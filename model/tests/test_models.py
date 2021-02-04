'''In this module we should place simple tests for the models.'''
from numpy import array, where, zeros
from numpy.linalg import norm
from numpy.testing import assert_almost_equal
from model.preprocessing import TwoAgeModelInput, HouseholdPopulation
from model.preprocessing import make_initial_condition
from model.common import SEDURRateEquations

DEFAULT_SPEC = {
    # Interpretable parameters:
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
    'pop_pyramid_file_name': 'inputs/United Kingdom-2019.csv',
    'rho_file_name': 'inputs/rho_estimate_cdc.csv'
}


def test_simple():
    '''Test a simple two age model'''
    model_input = TwoAgeModelInput(DEFAULT_SPEC)
    composition_list = array(
        [[0, 1], [0, 2], [1, 1], [1, 2], [2, 1], [2, 2]])
    # Proportion of households which are in each composition
    composition_distribution = array(
        [0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
    household_population = HouseholdPopulation(
        composition_list, composition_distribution, 'SEDUR', model_input)

    rhs = SEDURRateEquations(
        'SEDUR',
        model_input,
        household_population)

    H0 = make_initial_condition(household_population, rhs)
    dH = rhs(0.0, H0)
    assert_almost_equal(7.776182090170313e-05, norm(dH))
    assert_almost_equal(2.7801365751414517e-05, max(dH))
    assert_almost_equal(-3.270492048199998e-05, min(dH))
