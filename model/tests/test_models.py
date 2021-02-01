'''In this module we should place simple tests for the models.'''
from numpy import array, where, zeros
from numpy.linalg import norm
from numpy.testing import assert_almost_equal
from model.preprocessing import TwoAgeModelInput, HouseholdPopulation
from model.preprocessing import make_initial_condition
from model.specs import DEFAULT_SPEC
from model.common import SEDURRateEquations


def test_simple():
    '''Test a simple two age model'''
    model_input = TwoAgeModelInput(DEFAULT_SPEC)
    composition_list = array(
        [[0, 1], [0, 2], [1, 1], [1, 2], [2, 1], [2, 2]])
    # Proportion of households which are in each composition
    composition_distribution = array(
        [0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
    household_population = HouseholdPopulation(
        composition_list, composition_distribution, 'SEDUR' model_input)

    rhs = SEDURRateEquations(
        'SEDUR',
        model_input,
        household_population)

    H0 = make_initial_condition(household_population, rhs)
    dH = rhs(0.0, H0)
    assert_almost_equal(7.776182090170313e-05, norm(dH))
    assert_almost_equal(2.7801365751414517e-05, max(dH))
    assert_almost_equal(-3.270492048199998e-05, min(dH))
