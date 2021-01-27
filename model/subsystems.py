'''This script defines all the subsystems for specific compartmental structures.
First, some functions for doing standard transition events are defined, then
these are used to create some functions for implementing common comprtmental
 structures. These are then stored in a dictionary, which always needs to go at
 the end of this script.'''

from copy import copy, deepcopy
from numpy import (
        append, arange, around, array, cumsum, ones, ones_like, where,
        zeros, concatenate, vstack, identity, tile, hstack, prod, ix_,
        atleast_2d, diag)
from model.common import build_state_matrix, my_int, sparse, zeros

def inf_events(from_compartment,
                to_compartment,
                inf_compartment_list,
                inf_scales,
                r_home,
                no_compartments,
                composition,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int,
                inf_event_row,
                inf_event_col,
                inf_event_class):

    # This function adds infection events to a within-household transition matrix, allowing for multiple infectious classes

    no_inf_compartments = len(inf_compartment_list) # Total number of compartments contributing to within-household infection

    for i in range(len(class_idx)):
        from_present = where(states[:, no_compartments*i+from_compartment] > 0)[0]

        inf_to = zeros(len(from_present), dtype=my_int)
        inf_rate = zeros(len(from_present))
        for k in range(len(from_present)):
            old_state = copy(states[from_present[k], :])
            old_infs = 0
            for ic in range(no_inf_compartments):
                old_infs += inf_scales[ic] * (old_state[inf_compartment_list[ic] + no_compartments * arange(len(class_idx))] /
                                                composition[class_idx])
            inf_rate[k] = old_state[no_compartments*i] * (
                r_home[i, :].dot( old_infs ))
            new_state = old_state.copy()
            new_state[no_compartments*i + from_compartment] -= 1
            new_state[no_compartments*i + to_compartment] += 1
            inf_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]

        Q_int += sparse(
            (inf_rate, (from_present, inf_to)),
            shape=matrix_shape,)
        inf_event_row = concatenate((inf_event_row, from_present))
        inf_event_col = concatenate((inf_event_col, inf_to))
        inf_event_class = concatenate(
            (inf_event_class, class_idx[i]*ones((len(from_present)))))

    return Q_int, inf_event_row, inf_event_col, inf_event_class

def progression_events(from_compartment,
                to_compartment,
                pc_rate,
                no_compartments,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int):

    # This function adds a single set of progression events to a within-household transition matrix

    for i in range(len(class_idx)):
        from_present = where(states[:, no_compartments*i+from_compartment] > 0)[0]

        prog_to = zeros(len(from_present), dtype=my_int)
        prog_rate = zeros(len(from_present))
        for k in range(len(from_present)):
            old_state = copy(states[from_present[k], :])
            prog_rate[k] = pc_rate * old_state[no_compartments*i+from_compartment]
            new_state = copy(old_state)
            new_state[no_compartments*i+from_compartment] -= 1
            new_state[no_compartments*i+to_compartment] += 1
            prog_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]

        Q_int += sparse(
            (prog_rate, (from_present, prog_to)),
            shape=matrix_shape,)

    return Q_int

def stratified_progression_events(from_compartment,
                to_compartment,
                pc_rate_by_class,
                no_compartments,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int):

    ''' This function adds a single set of progression events to a
    within-household transition matrix, with progression rates stratified by
    class.'''

    for i in range(len(class_idx)):
        from_present = where(states[:, no_compartments*i+from_compartment] > 0)[0]

        prog_to = zeros(len(from_present), dtype=my_int)
        prog_rate = zeros(len(from_present))
        for k in range(len(from_present)):
            old_state = copy(states[from_present[k], :])
            prog_rate[k] = pc_rate_by_class[i] * old_state[no_compartments*i+from_compartment]
            new_state = copy(old_state)
            new_state[no_compartments*i+from_compartment] -= 1
            new_state[no_compartments*i+to_compartment] += 1
            prog_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]
        Q_int += sparse(
            (prog_rate, (from_present, prog_to)),
            shape=matrix_shape)

    return Q_int

def _sir_subsystem(self, household_spec):
    '''This function processes a composition to create subsystems i.e.
    matrices and vectors describing all possible epdiemiological states
    for a given household composition
    Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    s_comp, i_comp, r_comp = range(3)

    composition = household_spec.composition
    matrix_shape = household_spec.matrix_shape
    sus = self.model_input.sus
    K_home = self.model_input.k_home
    gamma = self.model_input.gamma

    # Set of individuals actually present here
    class_idx = household_spec.class_indexes

    K_home = K_home[ix_(class_idx, class_idx)]
    sus = sus[class_idx]
    r_home = atleast_2d(diag(sus).dot(K_home))

    states, \
        reverse_prod, \
        index_vector, \
        rows = build_state_matrix(household_spec)

    Q_int = sparse(household_spec.matrix_shape,)
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)

    Q_int, inf_event_row, inf_event_col, inf_event_class = inf_events(s_comp,
                i_comp,
                [i_comp],
                [1],
                r_home,
                3,
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
    Q_int = progression_events(i_comp,
                    r_comp,
                    gamma,
                    3,
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

def _seir_subsystem(self, household_spec):
    '''This function processes a composition to create subsystems i.e.
    matrices and vectors describing all possible epdiemiological states
    for a given household composition
    Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    s_comp, e_comp, i_comp, r_comp = range(4)

    composition = household_spec.composition
    matrix_shape = household_spec.matrix_shape
    sus = self.model_input.sus
    K_home = self.model_input.k_home
    alpha = self.model_input.alpha
    gamma = self.model_input.gamma

    # Set of individuals actually present here
    class_idx = household_spec.class_indexes

    K_home = K_home[ix_(class_idx, class_idx)]
    sus = sus[class_idx]
    r_home = atleast_2d(diag(sus).dot(K_home))

    states, \
        reverse_prod, \
        index_vector, \
        rows = build_state_matrix(household_spec)

    Q_int = sparse(household_spec.matrix_shape,)
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)

    Q_int, inf_event_row, inf_event_col, inf_event_class = inf_events(s_comp,
                i_comp,
                [i_comp],
                [1],
                r_home,
                4,
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
    Q_int = progression_events(e_comp,
                    i_comp,
                    gamma,
                    4,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = progression_events(i_comp,
                    r_comp,
                    gamma,
                    4,
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

def _sepir_subsystem(self, household_spec):
    '''This function processes a composition to create subsystems i.e.
    matrices and vectors describing all possible epdiemiological states
    for a given household composition
    Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    s_comp, e_comp, p_comp, i_comp, r_comp = range(5)

    composition = household_spec.composition
    matrix_shape = household_spec.matrix_shape
    sus = self.model_input.sus
    K_home = self.model_input.k_home
    tau = self.model_input.tau
    alpha_1 = self.model_input.alpha_1
    alpha_2 = self.model_input.alpha_2
    gamma = self.model_input.gamma

    # Set of individuals actually present here
    class_idx = household_spec.class_indexes

    K_home = K_home[ix_(class_idx, class_idx)]
    sus = sus[class_idx]
    r_home = atleast_2d(diag(sus).dot(K_home))

    states, \
        reverse_prod, \
        index_vector, \
        rows = build_state_matrix(household_spec)

    Q_int = sparse(household_spec.matrix_shape,)
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)

    Q_int, inf_event_row, inf_event_col, inf_event_class = inf_events(s_comp,
                i_comp,
                [p_comp, i_comp],
                [tau, 1],
                r_home,
                5,
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
    Q_int = progression_events(e_comp,
                    p_comp,
                    alpha_1,
                    5,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = progression_events(p_comp,
                    i_comp,
                    alpha_2,
                    5,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = progression_events(i_comp,
                    r_comp,
                    gamma,
                    5,
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


def _sedur_subsystem(self, household_spec):
    '''This function processes a composition to create subsystems i.e.
    matrices and vectors describing all possible epdiemiological states
    for a given household composition
    Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    s_comp, e_comp, d_comp, u_comp, r_comp = range(5)

    composition = household_spec.composition
    matrix_shape = household_spec.matrix_shape
    sus = self.model_input.sus
    det = self.model_input.det
    tau = self.model_input.tau
    K_home = self.model_input.k_home
    alpha = self.model_input.alpha
    gamma = self.model_input.gamma

    # Set of individuals actually present here
    class_idx = household_spec.class_indexes

    K_home = K_home[ix_(class_idx, class_idx)]
    sus = sus[class_idx]
    det = det[class_idx]
    tau = tau[class_idx]
    r_home = atleast_2d(diag(sus).dot(K_home))

    states, \
        reverse_prod, \
        index_vector, \
        rows = build_state_matrix(household_spec)

    Q_int = sparse(household_spec.matrix_shape,)
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)


    Q_int = progression_events(d_comp,
        r_comp,
        gamma,
        5,
        states,
        index_vector,
        reverse_prod,
        class_idx,
        matrix_shape,
        Q_int)
    Q_int, inf_event_row, inf_event_col, inf_event_class = inf_events(s_comp,
                e_comp,
                [d_comp,u_comp],
                [1,tau],
                r_home,
                5,
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
    Q_int = progression_events(u_comp,
        r_comp,
        gamma,
        5,
        states,
        index_vector,
        reverse_prod,
        class_idx,
        matrix_shape,
        Q_int)
    Q_int = stratified_progression_events(e_comp,
                    d_comp,
                    alpha*det,
                    5,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = stratified_progression_events(e_comp,
                    u_comp,
                    alpha*(1-det),
                    5,
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



subsystem_key = {
'SIR' : [_sir_subsystem,3],
'SEIR' : [_seir_subsystem,4],
'SEPIR' : [_sepir_subsystem,5],
'SEDUR' : [_sedur_subsystem,5],
}
