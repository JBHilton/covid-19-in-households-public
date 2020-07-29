'''In this module we should place simple tests for the models.'''
from numpy import array, where, zeros
from numpy.linalg import norm
from numpy.testing import assert_almost_equal
from model.preprocessing import TwoAgeModelInput, build_household_population
from model.preprocessing import make_initial_condition
from model.specs import DEFAULT_SPEC
from model.common import NoImportRateEquations


def test_simple():
    '''Test a simple two age model'''
    model_input = TwoAgeModelInput(DEFAULT_SPEC)
    composition_list = array(
        [[0, 1], [0,2], [1, 1], [1, 2], [2, 1], [2, 2]])
    # Proportion of households which are in each composition
    composition_distribution = array(
        [0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
    Q_int, states, which_composition, \
        system_sizes, cum_sizes, \
        inf_event_row, inf_event_col, inf_event_class \
        = build_household_population(composition_list, model_input)

    rhs = NoImportRateEquations(
        model_input,
        Q_int,
        composition_list,
        which_composition,
        states,
        inf_event_row,
        inf_event_col,
        inf_event_class)

    H0 = make_initial_condition(
        composition_distribution, states, rhs, which_composition)
    dH = rhs(0.0, H0)
    assert_almost_equal(7.776182090170313e-05, norm(dH))
    assert_almost_equal(2.7801365751414517e-05, max(dH))
    assert_almost_equal(-3.270492048199998e-05, min(dH))
