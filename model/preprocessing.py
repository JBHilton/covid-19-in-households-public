'''Various functions and classes that help build the model'''
from numpy import (arange, array, cumsum, int64, ones, ones_like, power,
    where, zeros, concatenate)
from numpy import sum as nsum
from scipy.sparse import block_diag
from scipy.special import binom
from model.common import within_household_spread, sparse

def make_aggregator(coarse_bounds, fine_bounds):
    '''Construct a matrix that stores where each class in finer structure is
    in coarser structure'''
    return array([
        where(coarse_bounds >= fine_bounds[i + 1])[0][0] - 1
        for i in range(len(fine_bounds) - 1)])

def aggregate_contact_matrix(k_fine, fine_bds, coarse_bds, pyramid):
    '''Aggregates an age-structured contact matrice to return the corresponding
    transmission matrix under a finer age structure.'''

    aggregator = make_aggregator(coarse_bds, fine_bds)

    # Prem et al. estimates cut off at 80, so we bundle all >75 year olds into
    # one class for consistency with these estimates:
    pyramid[len(fine_bds) - 1] = nsum(pyramid[len(fine_bds) - 1:])
    pyramid = pyramid[:len(fine_bds) - 1]
    # Normalise to give proportions
    pyramid = pyramid / nsum(pyramid)


    # sparse matrix defined here just splits pyramid into rows corresponding to
    # coarse boundaries, then summing each row gives aggregated pyramid
    row_cols = (aggregator, arange(len(aggregator)))
    agg_pop_pyramid=nsum(
        sparse((pyramid, row_cols)),
        axis=1).getA().squeeze()

    rel_weights = pyramid / agg_pop_pyramid[aggregator]

    # Now define contact matrix with age classes from Li et al data
    pop_weight_matrix = sparse((rel_weights, row_cols))
    pop_no_weight=sparse((ones_like(aggregator), row_cols))

    return pop_weight_matrix * k_fine * pop_no_weight.T

def aggregate_vector_quantities(v_fine, fine_bds, coarse_bds, pyramid ):
    '''Aggregates an age-structured contact matrice to return the corresponding
    transmission matrix under a finer age structure.'''

    aggregator=make_aggregator(coarse_bds, fine_bds)

    # The Prem et al. estimates cut off at 80, so we all >75 year olds into one
    # class for consistency with these estimates:
    pyramid[len(fine_bds) - 1] = sum(pyramid[len(fine_bds)-1:])
    pyramid = pyramid[:len(fine_bds) - 1]
    # Normalise to give proportions
    pyramid = pyramid/nsum(pyramid)

    # sparse matrix defines here just splits pyramid into rows corresponding to
    # coarse boundaries, then summing each row gives aggregated pyramid
    row_cols = (aggregator, arange(len(aggregator)))
    agg_pop_pyramid=nsum(
        sparse((pyramid, row_cols)),
        axis=1).getA().squeeze()

    rel_weights = pyramid / agg_pop_pyramid[aggregator]

    # Now define contact matrix with age classes from Li et al data
    pop_weight_matrix = sparse((rel_weights, row_cols))

    return pop_weight_matrix * v_fine

def build_household_population(
        composition_list, sus, det, tau, K_home, alpha, gamma):
    '''This builds internal mixing matrix for entire system of age-structured
    households.'''

    # If the compositions include household size at the beginning, we throw it
    # away here. While we would expect to see some households with equal
    # numbers in age class 1 and all others combined, we should not see it
    # everywhere and so this is a safe way to check.
    condition = max(abs(
        composition_list[:, 0] - composition_list[:, 1:].sum(axis=1)))
    if condition == 0:
        size_list = composition_list[:,0]
        composition_list = composition_list[:,1:]
    else:
        size_list = composition_list.sum(axis=1)

    no_types, no_classes = composition_list.shape

    # This is an array of logicals telling you which classes are present in
    # each composition
    classes_present = composition_list > 0 

    system_sizes = ones(no_types, dtype=int64)
    for i, _ in enumerate(system_sizes):
        for j in where(classes_present[i, :])[0]:
            system_sizes[i] *= binom(
                composition_list[i, j] + 5 - 1,
                5 - 1)

    # This is useful for placing blocks of system states
    cum_sizes = cumsum(system_sizes) 
    # Total size is sum of the sizes of each composition-system, because we are
    # considering a single household which can be in any one composition
    total_size = cum_sizes[-1]
    # Stores list of (S,E,D,U,R)_a states for each composition
    states = zeros((total_size, 5 * no_classes), dtype=int64)
    which_composition = zeros(total_size, dtype=int64)

    # Initialise matrix of internal process by doing the first block
    which_composition[:system_sizes[0]] = zeros(system_sizes[0], dtype=int64)
    Q_temp, states_temp, inf_event_row, inf_event_col = within_household_spread(
        composition_list[0,:], sus, det, tau, K_home, alpha, gamma)
    print(inf_event_row)
    Q_int = sparse(Q_temp)
    class_list = where(classes_present[0, :])[0]
    for j in range(len(class_list)):
        this_class = class_list[j]
        states[:system_sizes[1], 5*this_class:5*(this_class+1)] = \
            states_temp[:, 5*j:5*(j+1)]

    # NOTE: The way I do this loop is very wasteful, I'm making lots of arrays
    # which I'm overwriting with different sizes
    # start_time=cputime
    # Just store this so we can estimate remaining time
    matrix_sizes = power(system_sizes, 2)
    # for i in range(1, no_types):
    for i in range(1, no_types):
        print('Processing {}/{}'.format(i, no_types))
        which_composition[cum_sizes[i-1]:cum_sizes[i]] = i * ones(
            system_sizes[i], dtype=int64)
        Q_temp, states_temp, inf_temp_row, inf_temp_col = within_household_spread(
            composition_list[i,:], sus, det, tau, K_home, alpha, gamma)
        Q_int = block_diag((Q_int, Q_temp), format='csc')
        Q_int.eliminate_zeros()
        class_list = where(classes_present[i,:])[0]
        print('Q_int shape={0} nnz={1} ({2:0.04f}%)'.format(
            Q_int.shape,
            Q_int.getnnz(),
            Q_int.getnnz()
            / (Q_int.shape[0] * Q_int.shape[1])*100))
        print('Q_temp shape={0} nnz={1} ({2:0.04f}%)'.format(
            Q_temp.shape,
            Q_temp.getnnz(),
            Q_temp.getnnz()
            / (Q_temp.shape[0]*Q_temp.shape[1])*100))
        for j in range(len(class_list)):
            this_class = class_list[j]
            states[
                cum_sizes[i-1]:cum_sizes[i],
                5*this_class:5*(this_class+1)] = states_temp[:, 5*j:5*(j+1)]

        inf_event_row = concatenate((inf_event_row, cum_sizes[i-1] + inf_temp_row))
        inf_event_col = concatenate((inf_event_col, cum_sizes[i-1] + inf_temp_col))
        #print(i, type(Q_int), Q_int.shape, type(states), states.shape, len(inf_event))
        # disp([num2str(i) ' blocks calculated. ' num2str(cputime-start_time) ' seconds elapsed, approximately ' num2str(((cputime-start_time)/sum(system_sizes(1:i)))*sum(system_sizes(i+1:end))) ' seconds remaining.'])
    return \
        Q_int, \
        states, \
        which_composition, \
        system_sizes, \
        cum_sizes, \
        inf_event_row, \
        inf_event_col 
