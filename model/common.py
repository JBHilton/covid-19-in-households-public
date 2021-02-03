'''Module for additional computations required by the model'''
from numpy import (
    arange, array, atleast_2d, concatenate, copy, cumprod, diag, isnan, ix_,
    ones, shape, sum, where, zeros)
from numpy import int64 as my_int
import pdb
from scipy.sparse import csc_matrix as sparse
from model.imports import NoImportModel
from model.subsystems import subsystem_key

def state_recursor(
        states,
        no_compartments,
        age_class,
        b_size,
        n_blocks,
        con_reps,
        c,
        x,
        depth,
        k):
    if depth < no_compartments-1:
        for x_i in arange(c + 1 - x.sum()):
            x[0, depth] = x_i
            x[0, depth+1:] = zeros(
                (1, no_compartments-depth-1),
                dtype=my_int)
            states, k = state_recursor(
                states,
                no_compartments,
                age_class,
                b_size,
                n_blocks,
                con_reps,
                c,
                x,
                depth+1,
                k)
    else:
        x[0, -1] = c - sum(x[0, :depth])
        for block in arange(n_blocks):
            repeat_range = arange(
                block * b_size
                + k * con_reps,
                block * b_size +
                (k + 1) * con_reps)
            states[repeat_range, no_compartments*age_class:no_compartments*(age_class+1)] = \
                ones(
                    (con_reps, 1),
                    dtype=my_int) \
                * array(
                    x,
                    ndmin=2, dtype=my_int)
        k += 1
        return states, k
    return states, k


def build_states_recursively(
        total_size,
        no_compartments,
        classes_present,
        block_size,
        num_blocks,
        consecutive_repeats,
        composition):
    states = zeros(
        (total_size, no_compartments*len(classes_present)),
        dtype=my_int)
    for age_class in range(len(classes_present)):
        k = 0
        states, k = state_recursor(
            states,
            no_compartments,
            age_class,
            block_size[age_class],
            num_blocks[age_class],
            consecutive_repeats[age_class],
            composition[classes_present[age_class]],
            zeros([1, no_compartments], dtype=my_int),
            0,
            k)
    return states, k


def build_state_matrix(household_spec):
    # Number of times you repeat states for each configuration
    consecutive_repeats = concatenate((
        ones(1, dtype=my_int), cumprod(household_spec.system_sizes[:-1])))
    block_size = consecutive_repeats * household_spec.system_sizes
    num_blocks = household_spec.total_size // block_size

    states, k = build_states_recursively(
        household_spec.total_size,
        household_spec.no_compartments,
        household_spec.class_indexes,
        block_size,
        num_blocks,
        consecutive_repeats,
        household_spec.composition)
    # Now construct a sparse vector which tells you which row a state appears
    # from in the state array

    # This loop tells us how many values each column of the state array can
    # take
    state_sizes = concatenate([
        (household_spec.composition[i] + 1)
        * ones(household_spec.no_compartments, dtype=my_int)
        for i in household_spec.class_indexes]).ravel()

    # This vector stores the number of combinations you can get of all
    # subsequent elements in the state array, i.e. reverse_prod(i) tells you
    # how many arrangements you can get in states(:,i+1:end)
    reverse_prod = array([0, *cumprod(state_sizes[:0:-1])])[::-1]

    # We can then define index_vector look up the location of a state by
    # weighting its elements using reverse_prod - this gives a unique mapping
    # from the set of states to the integers. Because lots of combinations
    # don't actually appear in the states array, we use a sparse array which
    # will be much bigger than we actually require
    rows = [
        states[k, :].dot(reverse_prod) + states[k, -1]
        for k in range(household_spec.total_size)]

    if min(rows) < 0:
        print(
            'Negative row indices found, proportional total',
            sum(array(rows) < 0),
            '/',
            len(rows),
            '=',
            sum(array(rows) < 0) / len(rows))
    index_vector = sparse((
        arange(household_spec.total_size),
        (rows, [0]*household_spec.total_size)))

    return states, reverse_prod, index_vector, rows


def within_household_spread(
        composition, model_input):
    '''Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    sus = model_input.sus
    det = model_input.det
    tau = model_input.tau
    K_home = model_input.k_home
    alpha = model_input.alpha
    gamma = model_input.gamma

    # Set of individuals actually present here
    classes_present = where(composition.ravel() > 0)[0]

    K_home = K_home[ix_(classes_present, classes_present)]
    sus = sus[classes_present]
    det = det[classes_present]
    tau = tau[classes_present]
    r_home = atleast_2d(diag(sus).dot(K_home))

    states, total_size, reverse_prod, index_vector, rows = build_state_matrix(composition, classes_present, 5)

    d_pos = 2 + 5 * arange(len(classes_present))
    u_pos = 3 + 5 * arange(len(classes_present))

    Q_int = sparse((total_size, total_size))
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)

    # Add events for each age class
    for i in range(len(classes_present)):
        s_present = where(states[:, 5*i] > 0)[0]
        e_present = where(states[:, 5*i+1] > 0)[0]
        d_present = where(states[:, 5*i+2] > 0)[0]
        u_present = where(states[:, 5*i+3] > 0)[0]

        # First do infection events
        inf_to = zeros(len(s_present), dtype=my_int)
        inf_rate = zeros(len(s_present))
        for k in range(len(s_present)):
            old_state = copy(states[s_present[k], :])
            inf_rate[k] = old_state[5*i] * (
                r_home[i, :].dot(
                    (old_state[d_pos] / composition[classes_present])
                    + (old_state[u_pos] / composition[classes_present]) * tau))
            new_state = old_state.copy()
            new_state[5*i] -= 1
            new_state[5*i + 1] += 1
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
        # # Now do exposure to detected or undetected
        det_to = zeros(len(e_present), dtype=my_int)
        det_rate = zeros(len(e_present))
        undet_to = zeros(len(e_present), dtype=my_int)
        undet_rate = zeros(len(e_present))
        for k in range(len(e_present)):
            # First do detected
            old_state = copy(states[e_present[k], :])
            det_rate[k] = det[i] * alpha * old_state[5*i+1]
            new_state = copy(old_state)
            new_state[5*i + 1] -= 1
            new_state[5*i + 2] += 1
            det_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]
            # First do undetectednt(k),:)
            undet_rate[k] = (1.0 - det[i]) * alpha * old_state[5*i+1]
            new_state = copy(old_state)
            new_state[5*i + 1] -= 1
            new_state[5*i + 3] += 1
            undet_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]

        Q_int += sparse(
            (det_rate, (e_present, det_to)),
            shape=(total_size, total_size))
        Q_int += sparse(
            (undet_rate, (e_present, undet_to)),
            shape=(total_size, total_size))
        # # disp('Incubaion events done')

        # Now do recovery of detected cases
        rec_to = zeros(len(d_present), dtype=my_int)
        rec_rate = zeros(len(d_present))
        for k in range(len(d_present)):
            old_state = copy(states[d_present[k], :])
            rec_rate[k] = gamma * old_state[5*i+2]
            new_state = copy(old_state)
            new_state[5*i+2] -= 1
            new_state[5*i+4] += 1
            rec_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]
        Q_int += sparse(
            (rec_rate, (d_present, rec_to)),
            shape=(total_size, total_size))
        # disp('Recovery events from detecteds done')
        # Now do recovery of undetected cases
        rec_to = zeros(len(u_present), dtype=my_int)
        rec_rate = zeros(len(u_present))
        for k in range(len(u_present)):
            old_state = copy(states[u_present[k], :])
            rec_rate[k] = gamma*old_state[5*i+3]
            new_state = copy(old_state)
            new_state[5*i+3] -= 1
            new_state[5*i+4] += 1
            rec_to[k] = index_vector[
                new_state.dot(reverse_prod) +new_state[-1], 0]
        Q_int = Q_int + sparse(
            (rec_rate, (u_present, rec_to)),
            shape=(total_size, total_size))
        # disp('Recovery events from undetecteds done')

    S = Q_int.sum(axis=1).getA().squeeze()
    Q_int += sparse((
        -S, (arange(total_size), arange(total_size))))
    return \
        Q_int, states, \
        array(inf_event_row, dtype=my_int, ndmin=1), \
        array(inf_event_col, dtype=my_int, ndmin=1), \
        array(inf_event_class, dtype=my_int, ndmin=1)

def within_household_SEDURQ(
        composition, model_input):
    '''Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    sus = model_input.sigma
    det = model_input.det
    tau = model_input.tau
    K_home = model_input.k_home
    alpha = model_input.alpha
    gamma = model_input.gamma
    D_iso_rate = model_input.D_iso_rate
    U_iso_rate = model_input.U_iso_rate
    discharge_rate = model_input.discharge_rate
    adult_bd = model_input.adult_bd
    class_is_isolating = model_input.class_is_isolating

    # Set of individuals actually present here
    classes_present = where(composition.ravel() > 0)[0]

    # Check number of adults and whether children_present
    no_adults = sum(composition[adult_bd:])
    children_present = sum(composition[:adult_bd])>0

    K_home = K_home[ix_(classes_present, classes_present)]
    sus = sus[classes_present]
    det = det[classes_present]
    tau = tau[classes_present]
    r_home = atleast_2d(diag(sus).dot(K_home))

    states, total_size, reverse_prod, index_vector, rows = build_state_matrix(composition, classes_present, 6)

    d_pos = 2 + 6 * arange(len(classes_present))
    u_pos = 3 + 6 * arange(len(classes_present))
    iso_pos = 5 + 6 * arange(len(classes_present))

    iso_adjusted_comp = composition[classes_present] - states[:,iso_pos] # This is number of people of each age class present in the household given some may isolate
    iso_adjusted_comp[iso_adjusted_comp==0] = 1 # Replace zeros with ones - we only ever use this as a denominator whose numerator will be zero anyway if it should be zero
    if (iso_adjusted_comp<1).any():
        pdb.set_trace()
    adults_isolating = states[:,6*adult_bd+5::6].sum(axis=1) # Number of adults isolating by state


    Q_int = sparse((total_size, total_size))
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)

    # Add events for each age class
    for i in range(len(classes_present)):
        s_present = where(states[:, 6*i] > 0)[0]
        e_present = where(states[:, 6*i+1] > 0)[0]
        d_present = where(states[:, 6*i+2] > 0)[0]
        u_present = where(states[:, 6*i+3] > 0)[0]

        # First do infection events
        inf_to = zeros(len(s_present), dtype=my_int)
        inf_rate = zeros(len(s_present))
        for k in range(len(s_present)):
            old_state = copy(states[s_present[k], :])
            inf_rate[k] = old_state[6*i] * (
                r_home[i, :].dot(
                    (old_state[d_pos] / iso_adjusted_comp[k])
                    + (old_state[u_pos] / iso_adjusted_comp[k]) * tau))
            new_state = old_state.copy()
            new_state[6*i] -= 1
            new_state[6*i + 1] += 1
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
        # # Now do exposure to detected or undetected
        det_to = zeros(len(e_present), dtype=my_int)
        det_rate = zeros(len(e_present))
        undet_to = zeros(len(e_present), dtype=my_int)
        undet_rate = zeros(len(e_present))
        for k in range(len(e_present)):
            # First do detected
            old_state = copy(states[e_present[k], :])
            det_rate[k] = det[i] * alpha * old_state[6*i+1]
            new_state = copy(old_state)
            new_state[6*i + 1] -= 1
            new_state[6*i + 2] += 1
            det_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]
            # First do undetectednt(k),:)
            undet_rate[k] = (1.0 - det[i]) * alpha * old_state[6*i+1]
            new_state = copy(old_state)
            new_state[6*i + 1] -= 1
            new_state[6*i + 3] += 1
            undet_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]

        Q_int += sparse(
            (det_rate, (e_present, det_to)),
            shape=(total_size, total_size))
        Q_int += sparse(
            (undet_rate, (e_present, undet_to)),
            shape=(total_size, total_size))
        # # disp('Incubaion events done')

        # Now do recovery of detected cases
        rec_to = zeros(len(d_present), dtype=my_int)
        rec_rate = zeros(len(d_present))
        for k in range(len(d_present)):
            old_state = copy(states[d_present[k], :])
            rec_rate[k] = gamma * old_state[6*i+2]
            new_state = copy(old_state)
            new_state[6*i+2] -= 1
            new_state[6*i+4] += 1
            rec_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]
        Q_int += sparse(
            (rec_rate, (d_present, rec_to)),
            shape=(total_size, total_size))
        # disp('Recovery events from detecteds done')
        # Now do recovery of undetected cases
        rec_to = zeros(len(u_present), dtype=my_int)
        rec_rate = zeros(len(u_present))
        for k in range(len(u_present)):
            old_state = copy(states[u_present[k], :])
            rec_rate[k] = gamma*old_state[6*i+3]
            new_state = copy(old_state)
            new_state[6*i+3] -= 1
            new_state[6*i+4] += 1
            rec_to[k] = index_vector[
                new_state.dot(reverse_prod) +new_state[-1], 0]
        Q_int = Q_int + sparse(
            (rec_rate, (u_present, rec_to)),
            shape=(total_size, total_size))
        # disp('Recovery events from undetecteds done')

        #Now do isolation
        if (class_is_isolating[i,classes_present]).any():
            if (i<adult_bd) or not children_present: # If i is a child class or there are no children around, anyone can isolate
                d_can_isolate = d_present
                u_can_isolate = u_present
            else: # If children are present adults_isolating must stay below no_adults-1 so the children still have a guardian
                d_can_isolate = where((states[:, 6*i+2] > 0)*(adults_isolating<no_adults-1))[0]
                u_can_isolate = where((states[:, 6*i+3] > 0)*(adults_isolating<no_adults-1))[0]
            iso_present = where(states[:, 6*i+5] > 0)[0]
            # Isolation of detected cases
            iso_to = zeros(len(d_can_isolate), dtype=my_int)
            iso_rate = zeros(len(d_can_isolate))
            for k in range(len(d_can_isolate)):
                old_state = copy(states[d_can_isolate[k], :])
                iso_rate[k] = D_iso_rate * old_state[6*i+2]
                new_state = copy(old_state)
                new_state[6*i+2] -= 1
                new_state[6*i+5] += 1
                iso_to[k] = index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (iso_rate, (d_can_isolate, iso_to)),
                shape=(total_size, total_size))
            # Isolation of undetected cases
            iso_to = zeros(len(u_can_isolate), dtype=my_int)
            iso_rate = zeros(len(u_can_isolate))
            for k in range(len(u_can_isolate)):
                old_state = copy(states[u_can_isolate[k], :])
                iso_rate[k] = U_iso_rate * old_state[6*i+3]
                new_state = copy(old_state)
                new_state[6*i+3] -= 1
                new_state[6*i+5] += 1
                iso_to[k] = index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (iso_rate, (u_can_isolate, iso_to)),
                shape=(total_size, total_size))
            # Return home of isolated cases
            return_to = zeros(len(iso_present), dtype=my_int)
            return_rate = zeros(len(iso_present))
            for k in range(len(iso_present)):
                old_state = copy(states[iso_present[k], :])
                return_rate[k] = discharge_rate * old_state[6*i+5]
                new_state = copy(old_state)
                new_state[6*i+5] -= 1
                new_state[6*i+4] += 1
                return_to[k] = index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (return_rate, (iso_present, return_to)),
                shape = (total_size,total_size))


    S = Q_int.sum(axis=1).getA().squeeze()
    Q_int += sparse((
        -S, (arange(total_size), arange(total_size))))
    return \
        Q_int, states, \
        array(inf_event_row, dtype=my_int, ndmin=1), \
        array(inf_event_col, dtype=my_int, ndmin=1), \
        array(inf_event_class, dtype=my_int, ndmin=1)

def within_household_SEPIRQ(
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
    E_iso_rate = model_input.E_iso_rate
    P_iso_rate = model_input.P_iso_rate
    I_iso_rate = model_input.I_iso_rate
    discharge_rate = model_input.discharge_rate
    adult_bd = model_input.adult_bd
    class_is_isolating = model_input.class_is_isolating
    iso_method = model_input.iso_method # Set to 0 if isolating externaly, 1 if isolating internally

    tau_Q = (tau/alpha_2 + 1/gamma)/(1/alpha_1+1/alpha_2+1/gamma) # Scaling for infection from quarantined cases

    # Set of individuals actually present here
    classes_present = where(composition.ravel() > 0)[0]

    # Check number of adults and whether children_present
    no_adults = sum(composition[adult_bd:])
    children_present = sum(composition[:adult_bd])>0

    K_home = K_home[ix_(classes_present, classes_present)]
    sus = sus[classes_present]
    tau = tau[classes_present]
    tau_Q = tau_Q[classes_present]
    r_home = atleast_2d(diag(sus).dot(K_home))

    states, total_size, reverse_prod, index_vector, rows = build_state_matrix(composition, classes_present, 6)

    p_pos = 2 + 6 * arange(len(classes_present))
    i_pos = 3 + 6 * arange(len(classes_present))
    iso_pos = 5 + 6 * arange(len(classes_present))

    iso_adjusted_comp = composition[classes_present] - (1-iso_method)*states[:,iso_pos] # This is number of people of each age class present in the household given some may isolate
    iso_adjusted_comp[iso_adjusted_comp==0] = 1 # Replace zeros with ones - we only ever use this as a denominator whose numerator will be zero anyway if it should be zero
    if (iso_adjusted_comp<1).any():
        pdb.set_trace()
    adults_isolating = states[:,6*adult_bd+5::6].sum(axis=1) # Number of adults isolating by state

    Q_int = sparse((total_size, total_size))
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)

    # Add events for each age class
    for i in range(len(classes_present)):
        s_present = where(states[:, 6*i] > 0)[0]
        e_present = where(states[:, 6*i+1] > 0)[0]
        p_present = where(states[:, 6*i+2] > 0)[0]
        i_present = where(states[:, 6*i+3] > 0)[0]

        # First do infection events
        inf_to = zeros(len(s_present), dtype=my_int)
        inf_rate = zeros(len(s_present))
        for k in range(len(s_present)):
            old_state = copy(states[s_present[k], :])
            inf_rate[k] = old_state[6*i] * (
                r_home[i, :].dot(
                    (old_state[i_pos] / iso_adjusted_comp[k])
                    + (old_state[p_pos] / iso_adjusted_comp[k]) * tau # tau is prodromal reduction
                    + iso_method*(old_state[iso_pos] / iso_adjusted_comp[k]) * tau_Q)) # if we are doing internal isolation we scale down by tau_Q
            new_state = old_state.copy()
            new_state[6*i] -= 1
            new_state[6*i + 1] += 1
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
            inc_rate[k] = alpha_1 * old_state[6*i+1]
            new_state = copy(old_state)
            new_state[6*i + 1] -= 1
            new_state[6*i + 2] += 1
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
            dev_rate[k] = alpha_2 * old_state[6*i+2]
            new_state = copy(old_state)
            new_state[6*i + 2] -= 1
            new_state[6*i + 3] += 1
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
            rec_rate[k] = gamma * old_state[6*i+3]
            new_state = copy(old_state)
            new_state[6*i+3] -= 1
            new_state[6*i+4] += 1
            rec_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]
        Q_int += sparse(
            (rec_rate, (i_present, rec_to)),
            shape=(total_size, total_size))
        # disp('Recovery events from detecteds done')

        #Now do isolation
        if (class_is_isolating[classes_present[i],classes_present]).any(): # This checks whether class i is meant to isolated and whether any of the vulnerable classes are present
            # if not classes_present[i]==1:
            #     pdb.set_trace()
            if iso_method==1 or (i<adult_bd) or not children_present: # If isolating internally, i is a child class, or there are no children around, anyone can isolate
                e_can_isolate = where((states[:, 6*i+1] > 0)*(states[:, 6*i+5] == 0))[0]
                p_can_isolate = where((states[:, 6*i+2] > 0)*(states[:, 6*i+5] == 0))[0]
                i_can_isolate = where((states[:, 6*i+3] > 0)*(states[:, 6*i+5] == 0))[0]
            else: # If children are present adults_isolating must stay below no_adults-1 so the children still have a guardian
                e_can_isolate = where((states[:, 6*i+1] > 0)*(adults_isolating<no_adults-1))[0]
                p_can_isolate = where((states[:, 6*i+2] > 0)*(adults_isolating<no_adults-1))[0]
                i_can_isolate = where((states[:, 6*i+3] > 0)*(adults_isolating<no_adults-1))[0]
            iso_present = where(states[:, 6*i+5] > 0)[0]
            # Isolation of incubating cases
            iso_to = zeros(len(e_can_isolate), dtype=my_int)
            iso_rate = zeros(len(e_can_isolate))
            for k in range(len(e_can_isolate)):
                old_state = copy(states[e_can_isolate[k], :])
                iso_rate[k] = E_iso_rate * old_state[6*i+1]
                new_state = copy(old_state)
                new_state[6*i+1] -= 1
                new_state[6*i+5] += 1
                iso_to[k] = index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (iso_rate, (e_can_isolate, iso_to)),
                shape=(total_size, total_size))
            # Isolation of prodromal cases
            iso_to = zeros(len(p_can_isolate), dtype=my_int)
            iso_rate = zeros(len(p_can_isolate))
            for k in range(len(p_can_isolate)):
                old_state = copy(states[p_can_isolate[k], :])
                iso_rate[k] = P_iso_rate * old_state[6*i+2]
                new_state = copy(old_state)
                new_state[6*i+2] -= 1
                new_state[6*i+5] += 1
                iso_to[k] = index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (iso_rate, (p_can_isolate, iso_to)),
                shape=(total_size, total_size))
            # Isolation of fully infectious cases
            iso_to = zeros(len(i_can_isolate), dtype=my_int)
            iso_rate = zeros(len(i_can_isolate))
            for k in range(len(i_can_isolate)):
                old_state = copy(states[i_can_isolate[k], :])
                iso_rate[k] = I_iso_rate * old_state[6*i+3]
                new_state = copy(old_state)
                new_state[6*i+3] -= 1
                new_state[6*i+5] += 1
                iso_to[k] = index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (iso_rate, (i_can_isolate, iso_to)),
                shape=(total_size, total_size))
            # Return home of isolated cases
            return_to = zeros(len(iso_present), dtype=my_int)
            return_rate = zeros(len(iso_present))
            for k in range(len(iso_present)):
                old_state = copy(states[iso_present[k], :])
                return_rate[k] = discharge_rate * old_state[6*i+5]
                new_state = copy(old_state)
                new_state[6*i+5] -= 1
                new_state[6*i+4] += 1
                return_to[k] = index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (return_rate, (iso_present, return_to)),
                shape = (total_size,total_size))


    S = Q_int.sum(axis=1).getA().squeeze()
    Q_int += sparse((
        -S, (arange(total_size), arange(total_size))))
    return \
        Q_int, states, \
        array(inf_event_row, dtype=my_int, ndmin=1), \
        array(inf_event_col, dtype=my_int, ndmin=1), \
        array(inf_event_class, dtype=my_int, ndmin=1)

def build_external_import_matrix(
        household_population, FOI_list):
    '''Gets sparse matrices containing rates of external infection in a
    household of a given type'''

    row = household_population.inf_event_row
    col = household_population.inf_event_col
    inf_class = household_population.inf_event_class
    total_size = len(household_population.which_composition)

    matrix_shape = (total_size, total_size)

    Q_ext = sparse(matrix_shape,)

    for FOI in FOI_list:
        vals = FOI[row, inf_class]
        Q_ext += sparse((vals, (row, col)), shape = matrix_shape)

    diagonal_idexes = (arange(total_size), arange(total_size))
    S = Q_ext.sum(axis=1).getA().squeeze()
    Q_ext += sparse((-S, diagonal_idexes))

    return Q_ext


def build_external_import_matrix_SEPIRQ(
        household_population, FOI_pro, FOI_inf):
    '''Gets sparse matrices containing rates of external infection in a
    household of a given type'''

    row = household_population.inf_event_row
    col = household_population.inf_event_col
    inf_class = household_population.inf_event_class
    total_size = len(household_population.which_composition)

    # Figure out which class gets infected in this transition
    p_vals = FOI_pro[row, inf_class]
    i_vals = FOI_inf[row, inf_class]

    matrix_shape = (total_size, total_size)
    Q_ext_p = sparse(
        (p_vals, (row, col)),
        shape=matrix_shape)
    Q_ext_i = sparse(
        (i_vals, (row, col)),
        shape=matrix_shape)

    diagonal_idexes = (arange(total_size), arange(total_size))
    S = Q_ext_p.sum(axis=1).getA().squeeze()
    Q_ext_p += sparse((-S, diagonal_idexes))
    S = Q_ext_i.sum(axis=1).getA().squeeze()
    Q_ext_i += sparse((-S, diagonal_idexes))

    return Q_ext_p, Q_ext_i


class RateEquations:
    '''This class represents a functor for evaluating the rate equations for
    the model with no imports of infection from outside the population. The
    state of the class contains all essential variables'''
    # pylint: disable=invalid-name
    def __init__(self,
                 compartmental_structure,
                 model_input,
                 household_population,
                 epsilon=1.0):

        self.no_compartments = subsystem_key[compartmental_structure][1]
        self.household_population = household_population
        self.epsilon = epsilon
        self.Q_int = household_population.Q_int
        self.composition_by_state = household_population.composition_by_state
        self.states_sus_only = household_population.states[:, ::self.no_compartments]
        self.s_present = where(self.states_sus_only.sum(axis=1) > 0)[0]
        self.epsilon = epsilon
        self.import_model = model_input.import_model
        self.inf_compartment_list = subsystem_key[compartmental_structure][2]
        self.no_inf_compartments = len(self.inf_compartment_list)
        self.ext_matrix_list = []
        self.inf_by_state_list = []
        for ic in range(self.no_inf_compartments):
            self.ext_matrix_list.append(diag(model_input.sus).dot(model_input.k_ext).dot(diag(model_input.inf_scales[ic])))
            self.inf_by_state_list.append(household_population.states[:, self.inf_compartment_list[ic]::self.no_compartments])

    def __call__(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''
        Q_ext = self.external_matrices(t, H)
        if (H < 0).any():
            # pdb.set_trace()
            H[where(H < 0)[0]] = 0
        if isnan(H).any():
            pdb.set_trace()
        dH = (H.T * (self.Q_int + Q_ext)).T
        return dH

    def external_matrices(self, t, H):
        FOI_list = self.get_FOI_by_class(t, H)
        return build_external_import_matrix(
            self.household_population,
            FOI_list)

    def get_FOI_by_class(self, t, H):
        '''This calculates the age-stratified force-of-infection (FOI) on each
        household composition'''
        # Average number of each class by household
        denom = H.T.dot(self.composition_by_state)

        FOI_list = []

        for ic in range(self.no_inf_compartments):
            states_inf_only =  self.inf_by_state_list[ic]
            inf_by_class = zeros(shape(denom))
            inf_by_class[denom > 0] = (
                H.T.dot(states_inf_only)[denom > 0]
                / denom[denom > 0]).squeeze()
            FOI_list.append(self.states_sus_only.dot(
                    diag(self.ext_matrix_list[ic].dot(
                    self.epsilon * inf_by_class.T))))

        return FOI_list

class SIRRateEquations(RateEquations):
    @property
    def states_inf_only(self):
        return self.household_population.states[:, 1::self.no_compartments]
    @property
    def states_rec_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

class SEIRRateEquations(RateEquations):
    @property
    def states_inf_only(self):
        return self.household_population.states[:, 2::self.no_compartments]
    @property
    def states_rec_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

class SEPIRRateEquations(RateEquations):
    @property
    def states_inf_only(self):
        return self.household_population.states[:, 3::self.no_compartments]
    @property
    def states_rec_only(self):
        return self.household_population.states[:, 4::self.no_compartments]

class SEDURRateEquations(RateEquations):
    @property
    def states_det_only(self):
        return self.household_population.states[:, 2::self.no_compartments]
    @property
    def states_undet_only(self):
        return self.household_population.states[:, 3::self.no_compartments]
    @property
    def states_rec_only(self):
        return self.household_population.states[:, 4::self.no_compartments]


class SEPIRQRateEquations:
    '''This class represents a functor for evaluating the rate equations for
    the model with no imports of infection from outside the population. The
    state of the class contains all essential variables'''
    # pylint: disable=invalid-name
    def __init__(self,
                 model_input,
                 household_population,
                 import_model=NoImportModel()
                 ):

        no_compartments=6

        self.household_population = household_population
        self.states = household_population.states # We don't actually use this anywhere but it's very useful to have for debugging purposes
        self.epsilon = model_input.epsilon
        self.Q_int = household_population.Q_int
        # To define external mixing we need to set up the transmission
        # matrices.
        # Scale rows of contact matrix by
        self.pro_trans_matrix = model_input.sus*model_input.k_ext.dot(diag(model_input.tau))
        # age-specific susceptibilities
        # Scale columns by asymptomatic reduction in transmission
        self.inf_trans_matrix = model_input.sus*model_input.k_ext
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
        self.states_rec_only = household_population.states[:, 4::no_compartments]
        self.states_inf_only = household_population.states[:, 3::no_compartments]
        self.epsilon = model_input.epsilon
        self.import_model = import_model

        self.states_iso_only = household_population.states[:,5::no_compartments]

        self.iso_method = model_input.iso_method

        if self.iso_method==1:
            total_iso_by_state = self.states_iso_only.sum(axis=1)
            self.no_isos = total_iso_by_state==0
            self.isos_present = total_iso_by_state>0
            self.iso_prob = model_input.iso_prob

    def __call__(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''
        if (H<-0).any():
            # if (H<-1e-6).any():
            #     pdb.set_trace()
            H[where(H<0)[0]]=0
            H = H/sum(H)
        Q_ext_pro, Q_ext_inf = self.external_matrices(t, H)
        if isnan(H).any():
            pdb.set_trace()
        dH = (H.T * (self.Q_int + Q_ext_pro + Q_ext_inf)).T
        # if (H+dH<-1e-6).any():
        #     pdb.set_trace()
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
        denom = H.T.dot(self.composition_by_state) # Average number of each class by household
        if self.iso_method == 0:
            # Average prodromal infected by household in each class
            pro_by_class = zeros(shape(denom))
            pro_by_class[denom>0] = (
                H.T.dot(self.states_pro_only[ix_(arange(len(H)),denom>0)]) # Only want to do states with positive denominator
                / denom[denom>0]).squeeze()
            # Average full infectious infected by household in each class
            inf_by_class = zeros(shape(denom))
            inf_by_class[denom>0] = (
                H.T.dot(self.states_inf_only[ix_(range(len(H)),denom>0)])
                / denom[denom>0]).squeeze()
        else:
            # Average prodromal infected by household in each class
            pro_by_class = zeros(shape(denom))
            pro_by_class[denom > 0] = (
                (H[where(self.no_isos)[0]].T.dot(self.states_pro_only[ix_(where(self.no_isos)[0],denom>0)]) +
                (1 - self.iso_prob)*H[where(self.isos_present)[0]].T.dot(self.states_pro_only[ix_(where(self.isos_present)[0],denom>0)]))
                / denom[denom > 0]).squeeze()
            # Average full infectious infected by household in each class
            inf_by_class = zeros(shape(denom))
            inf_by_class[denom > 0] = ((
                H[where(self.no_isos)[0]].T.dot(
                    self.states_inf_only[ix_(where(self.no_isos)[0],denom > 0)])
                + (1.0 - self.iso_prob)
                * H[where(self.isos_present)[0]].T.dot(
                    self.states_inf_only[ix_(where(self.isos_present)[0],denom > 0)]))
                / denom[denom > 0]).squeeze()
        # This stores the rates of generating an infected of each class in each
        # state
        FOI_pro = self.states_sus_only.dot(
            diag(self.pro_trans_matrix.dot(
                self.epsilon * pro_by_class.T
                + self.import_model.detected(t))))
        # This stores the rates of generating an infected of each class in each
        # state
        FOI_inf = self.states_sus_only.dot(
            diag(self.inf_trans_matrix.dot(
                self.epsilon * inf_by_class.T
                + self.import_model.undetected(t))))

        return FOI_pro, FOI_inf


class CareHomeRateEquations:
    '''This class represents a functor for evaluating the rate equations for
    the model with no imports of infection from outside the population. The
    state of the class contains all essential variables'''
    # pylint: disable=invalid-name
    def __init__(self,
                 model_input,
                 household_population,
                 import_model
                 ):

        no_compartments=6

        self.household_population = household_population
        self.states = household_population.states # We don't actually use this anywhere but it's very useful to have for debugging purposes
        self.Q_int = household_population.Q_int
        # To define external mixing we need to set up the transmission
        # matrices.
        # Scale rows of contact matrix by
        # age-specific susceptibilities
        self.pro_trans_matrix = model_input.sus*model_input.k_ext.dot(diag(model_input.tau))
        # Scale columns by asymptomatic reduction in transmission
        self.inf_trans_matrix = model_input.sus*model_input.k_ext
        # This stores number in each age class by household
        self.composition_by_state = household_population.composition_by_state

        self.states_sus_only = household_population.states[:, ::no_compartments]
        self.states_pro_only = household_population.states[:, 2::no_compartments]
        self.states_inf_only = household_population.states[:, 3::no_compartments]
        self.states_emp_only = household_population.states[:, 5::no_compartments]

        self.import_rate = model_input.import_rate
        self.import_model = import_model

        self.epsilon = model_input.epsilon

    def __call__(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''
        if (H<-0).any():
            # if (H<-1e-6).any():
            #     pdb.set_trace()
            H[where(H<0)[0]]=0
            H = H/sum(H)
        Q_ext_pro, Q_ext_inf = self.external_matrices(t, H)
        if isnan(H).any():
            pdb.set_trace()
        dH = (H.T * (self.Q_int + Q_ext_pro + Q_ext_inf)).T
        # if (H+dH<-1e-6).any():
        #     pdb.set_trace()
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
        denom = H.T.dot(self.composition_by_state) # Average number of each class by household
        # Average prodromal infected by household in each class
        pro_by_class = zeros(shape(denom))
        pro_by_class[denom>0] = (
            H.T.dot(self.states_pro_only[ix_(arange(len(H)),denom>0)]) # Only want to do states with positive denominator
            / denom[denom>0]).squeeze()
        # Average full infectious infected by household in each class
        inf_by_class = zeros(shape(denom))
        inf_by_class[denom>0] = (
            H.T.dot(self.states_inf_only[ix_(range(len(H)),denom>0)])
            / denom[denom>0]).squeeze()

        FOI_pro = self.states_sus_only.dot(
            diag(self.epsilon *
                self.pro_trans_matrix.dot(pro_by_class.T)
                + self.import_rate.dot(
                self.import_model.prodromal(t))))
        FOI_inf = self.states_sus_only.dot(
            diag(self.epsilon *
                self.inf_trans_matrix.dot(inf_by_class.T)
                + self.import_rate.dot(
                self.import_model.infected(t))))

        return FOI_pro, FOI_inf
