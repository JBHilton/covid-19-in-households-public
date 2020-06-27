'''Module for additional computations required by the model'''
from numpy import (
    arange, array, atleast_2d, concatenate, copy, cumprod, diag,
    int64, ix_, ones, prod, where, zeros, ones)
from scipy.sparse import csc_matrix as sparse
from scipy.special import binom

def within_household_spread(
        composition, sus, det, tau, K_home, alpha, gamma):
    '''Assuming frequency-dependent homogeneous within-household mixing
    composition(i) isnumber of age class i individuals in the household'''

    # Set of individuals actually present here
    classes_present = where(composition.ravel() > 0)[0]

    K_home = K_home[ix_(classes_present, classes_present)]
    sus = sus[classes_present]
    det = det[classes_present]
    tau = tau[classes_present]
    r_home = atleast_2d(diag(sus).dot(K_home))

    system_sizes = array([
        binom(composition[classes_present[i]] + 5 - 1, 5 - 1)
        for i, _ in enumerate(classes_present)], dtype=int64)

    total_size = prod(system_sizes)

    states = zeros((total_size, 5*len(classes_present)), dtype=int64)
    # Number of times you repeat states for each configuration
    consecutive_repeats = concatenate((
        ones(1, dtype=int64), cumprod(system_sizes[:-1])))
    block_size = consecutive_repeats * system_sizes
    num_blocks = total_size // block_size

    for i in range(len(classes_present)):
        k = 0
        c = composition[classes_present[i]]
        for s in arange(c + 1):
            for e in arange(c - s + 1):
                for d in arange(c - s - e + 1):
                    for u in arange(c - s - e - d + 1):
                        for block in arange(num_blocks[i]):
                            repeat_range = arange(
                                block * block_size[i]
                                + k * consecutive_repeats[i],
                                block * block_size[i] +
                                (k + 1) * consecutive_repeats[i])
                            states[repeat_range, 5*i:5*(i+1)] = \
                                ones((consecutive_repeats[i], 1), dtype=int64) \
                                * array(
                                    [s, e, d, u, c - s - e - d - u],
                                    ndmin=2, dtype=int64)
                        k += 1
    # Q_int=sparse(total_size, total_size)

    d_pos = 2 + 5 * arange(len(classes_present))
    u_pos = 3 + 5 * arange(len(classes_present))

    # Now construct a sparse vector which tells you which row a state appears
    # from in the state array

    # This loop tells us how many values each column of the state array can
    # take
    state_sizes = concatenate([
        (composition[i] + 1) * ones(5, dtype=int64) for i in classes_present])

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
        for k in range(total_size)]
    index_vector = sparse((
        arange(total_size),
        (rows, [0]*total_size)))

    Q_int = sparse((total_size, total_size))
    inf_event_row = array([], dtype=int64)
    inf_event_col = array([], dtype=int64)
    inf_event_class = array([], dtype=int64)

    # Add events for each age class
    for i in range(len(classes_present)):
        s_present = where(states[:, 5*i] > 0)[0]
        e_present = where(states[:, 5*i+1] > 0)[0]
        d_present = where(states[:, 5*i+2] > 0)[0]
        u_present = where(states[:, 5*i+3] > 0)[0]

        # First do infection events
        inf_to = zeros(len(s_present), dtype=int64)
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
        inf_event_class = concatenate((inf_event_class,classes_present[i]*ones((len(s_present)))))
        # input('Press enter to continue')
        # # disp('Infection events done')
        # # Now do exposure to detected or undetected
        det_to = zeros(len(e_present), dtype=int64)
        det_rate = zeros(len(e_present))
        undet_to = zeros(len(e_present), dtype=int64)
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
        rec_to = zeros(len(d_present), dtype=int64)
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
        rec_to = zeros(len(u_present), dtype=int64)
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
        array(inf_event_row, dtype=int64, ndmin=1), \
        array(inf_event_col, dtype=int64, ndmin=1), \
        array(inf_event_class, dtype=int64, ndmin=1)


def build_external_import_matrix(states, row, col, inf_class, FOI_det, FOI_undet, total_size):
    '''Gets sparse matrices containing rates of external infection in a household
    of a given type'''
    d_vals = zeros(len(row))
    u_vals = zeros(len(row))

    for i in range(len(row)):
        old_state = states[row[i], :]
        new_state = states[col[i], :]
        # Figure out which class gets infected in this transition
        d_vals[i] = FOI_det[row[i], inf_class[i]]
        u_vals[i] = FOI_undet[row[i], inf_class[i]]

    matrix_shape = (total_size, total_size)
    Q_ext_d = sparse(
        (d_vals, (row, col)),
        shape=matrix_shape)
    Q_ext_u = sparse(
        (u_vals, (row, col)),
        shape=matrix_shape)


    diagonal_idexes = (arange(total_size), arange(total_size))
    S = Q_ext_d.sum(axis=1).getA().squeeze()
    Q_ext_d += sparse((-S, diagonal_idexes))
    S = Q_ext_u.sum(axis=1).getA().squeeze()
    Q_ext_u += sparse((-S, diagonal_idexes))

    return Q_ext_d, Q_ext_u


class RateEquations:
    '''This class represents a functor for evaluating the rate equations. The state
    of the class contains all essential variables'''
    # pylint: disable=invalid-name
    def __init__(self,
                 model_input,
                 Q_int,
                 composition_list,
                 which_composition,
                 states,
                 inf_event_row,
                 inf_event_col,
                 inf_event_class):

        self.Q_int = Q_int
        self.states = states
        self.inf_event_row = inf_event_row
        self.inf_event_col = inf_event_col
        self.inf_event_class = inf_event_class
        self.total_size = len(which_composition)
        # To define external mixing we need to set up the transmission matrices:
        # Scale rows of contact matrix by
        self.det_trans_matrix = diag(model_input.sigma).dot(model_input.k_ext)
        # age-specific susceptibilities
        # Scale columns by asymptomatic reduction in transmission
        self.undet_trans_matrix = diag(model_input.sigma).dot(
            model_input.k_ext.dot(diag(model_input.tau)))
        # This stores number in each age class by household
        self.composition_by_state = composition_list[which_composition, :]
        self.states_sus_only = states[:, ::5] # ::5 gives columns corresponding to
                                         # susceptible cases in each age class in
                                         # each state
        self.s_present = where(self.states_sus_only.sum(axis=1) > 0)[0]

        # Our starting state H is the composition distribution with a small amount of
        # infection present:

        # 2::5 gives columns corresponding to detected cases in each age class
        # in each state
        self.states_det_only = states[:, 2::5]
        # 4:5:end gives columns corresponding to undetected cases in each age
        # class in each state
        self.states_undet_only = states[:, 3::5]

    def get_FOI_by_class(self, H):
        '''TODO: improve docstring
        What is FOI?
        H is distribution of states by household'''
        # Average detected infected by household in each class
        det_by_class = (
            H.T.dot(self.states_det_only)
            / H.T.dot(self.composition_by_state)).squeeze()
        # Average undetected infected by household in each class
        undet_by_class = (
            H.T.dot(self.states_undet_only)
            / H.T.dot(self.composition_by_state)).squeeze()
        # This stores the rates of generating an infected of each class in each state
        FOI_det = self.states_sus_only.dot(
            diag(self.det_trans_matrix.dot(det_by_class.T)))
        # This stores the rates of generating an infected of each class in each state
        FOI_undet = self.states_sus_only.dot(
            diag(self.undet_trans_matrix.dot(undet_by_class.T)))

        return FOI_det, FOI_undet

    def external_matrices(self, H):
        FOI_det, FOI_undet = self.get_FOI_by_class(H)
        return build_external_import_matrix(
            self.states,
            self.inf_event_row,
            self.inf_event_col,
            self.inf_event_class,
            FOI_det,
            FOI_undet,
            self.total_size)


    def __call__(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''
        Q_ext_det, Q_ext_undet = self.external_matrices(H)
        dH = (H.T * (self.Q_int + Q_ext_det + Q_ext_undet)).T
        return dH

class RateEquationsWithImports:
    '''This class represents a functor for evaluating the rate equations. The state
    of the class contains all essential variables'''
    # pylint: disable=invalid-name
    def __init__(self,
                 model_input,
                 Q_int,
                 composition_list,
                 which_composition,
                 states,
                 inf_event_row,
                 inf_event_col,
                 inf_event_class,
                 det_imports,
                 undet_imports):

        self.Q_int = Q_int
        self.states = states
        self.inf_event_row = inf_event_row
        self.inf_event_col = inf_event_col
        self.inf_event_class = inf_event_class
        self.total_size = len(which_composition)
        # To define external mixing we need to set up the transmission matrices:
        # Scale rows of contact matrix by
        self.det_trans_matrix = diag(model_input.sigma).dot(model_input.k_ext)
        # age-specific susceptibilities
        # Scale columns by asymptomatic reduction in transmission
        self.undet_trans_matrix = diag(model_input.sigma).dot(
            model_input.k_ext.dot(diag(model_input.tau)))
        # This stores number in each age class by household
        self.composition_by_state = composition_list[which_composition, :]
        self.states_sus_only = states[:, ::5] # ::5 gives columns corresponding to
                                         # susceptible cases in each age class in
                                         # each state
        self.s_present = where(self.states_sus_only.sum(axis=1) > 0)[0]

        # Our starting state H is the composition distribution with a small amount of
        # infection present:

        # 2::5 gives columns corresponding to detected cases in each age class
        # in each state
        self.states_det_only = states[:, 2::5]
        # 4:5:end gives columns corresponding to undetected cases in each age
        # class in each state
        self.states_undet_only = states[:, 3::5]

        self.det_imports = det_imports
        self.undet_imports = undet_imports

    def get_FOI_by_class(self, H):
        '''TODO: improve docstring
        What is FOI?
        H is distribution of states by household'''
        # Average detected infected by household in each class
        det_by_class = (
            H.T.dot(self.states_det_only)
            / H.T.dot(self.composition_by_state)).squeeze()
        # Average undetected infected by household in each class
        undet_by_class = (
            H.T.dot(self.states_undet_only)
            / H.T.dot(self.composition_by_state)).squeeze()
        # This stores the rates of generating an infected of each class in each state
        FOI_det = self.states_sus_only.dot(
            diag(self.det_trans_matrix.dot(det_by_class.T + self.det_imports)))
        # This stores the rates of generating an infected of each class in each state
        FOI_undet = self.states_sus_only.dot(
            diag(self.undet_trans_matrix.dot(undet_by_class.T + self.undet_imports)))

        return FOI_det, FOI_undet

    def external_matrices(self, H):
        FOI_det, FOI_undet = self.get_FOI_by_class(H)
        return build_external_import_matrix(
            self.states,
            self.inf_event_row,
            self.inf_event_col,
            self.inf_event_class,
            FOI_det,
            FOI_undet,
            self.total_size)


    def __call__(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''
        Q_ext_det, Q_ext_undet = self.external_matrices(H)
        dH = (H.T * (self.Q_int + Q_ext_det + Q_ext_undet)).T
        return dH

class RateEquationsWithTimeImports:
    '''This class represents a functor for evaluating the rate equations. The state
    of the class contains all essential variables'''
    # pylint: disable=invalid-name
    def __init__(self,
                 t,
                 model_input,
                 Q_int,
                 composition_list,
                 which_composition,
                 states,
                 inf_event_row,
                 inf_event_col,
                 inf_event_class,
                 det_imports,
                 undet_imports,
                 import_times):

        self.t = t
        self.Q_int = Q_int
        self.states = states
        self.inf_event_row = inf_event_row
        self.inf_event_col = inf_event_col
        self.inf_event_class = inf_event_class
        self.total_size = len(which_composition)
        # To define external mixing we need to set up the transmission matrices:
        # Scale rows of contact matrix by
        self.det_trans_matrix = diag(model_input.sigma).dot(model_input.k_ext)
        # age-specific susceptibilities
        # Scale columns by asymptomatic reduction in transmission
        self.undet_trans_matrix = diag(model_input.sigma).dot(
            model_input.k_ext.dot(diag(model_input.tau)))
        # This stores number in each age class by household
        self.composition_by_state = composition_list[which_composition, :]
        self.states_sus_only = states[:, ::5] # ::5 gives columns corresponding to
                                         # susceptible cases in each age class in
                                         # each state
        self.s_present = where(self.states_sus_only.sum(axis=1) > 0)[0]

        # Our starting state H is the composition distribution with a small amount of
        # infection present:

        # 2::5 gives columns corresponding to detected cases in each age class
        # in each state
        self.states_det_only = states[:, 2::5]
        # 4:5:end gives columns corresponding to undetected cases in each age
        # class in each state
        self.states_undet_only = states[:, 3::5]

        self.det_imports = det_imports
        self.undet_imports = undet_imports
        self.import_times = import_times

    def get_FOI_by_class(self, H):
        '''TODO: improve docstring
        What is FOI?
        H is distribution of states by household'''
        # Average detected infected by household in each class
        time_step = -1
        time_located = False
        while not time_located:
            time_step+=1
            if (self.t>=self.import_times[time_step]) and (self.t<self.import_times[time_step+1]):
                time_located = True
            elif (self.t>=self.import_times[-1]):
                time_step = len(self.import_times)-1
                time_located = True
                
        det_by_class = (
            H.T.dot(self.states_det_only)
            / H.T.dot(self.composition_by_state)).squeeze()
        # Average undetected infected by household in each class
        undet_by_class = (
            H.T.dot(self.states_undet_only)
            / H.T.dot(self.composition_by_state)).squeeze()
        # This stores the rates of generating an infected of each class in each state
        FOI_det = self.states_sus_only.dot(
            diag(self.det_trans_matrix.dot(det_by_class.T + self.det_imports[:,time_step])))
        # This stores the rates of generating an infected of each class in each state
        FOI_undet = self.states_sus_only.dot(
            diag(self.undet_trans_matrix.dot(undet_by_class.T + self.undet_imports[:,time_step])))

        return FOI_det, FOI_undet

    def external_matrices(self, H):
        FOI_det, FOI_undet = self.get_FOI_by_class(H)
        return build_external_import_matrix(
            self.states,
            self.inf_event_row,
            self.inf_event_col,
            self.inf_event_class,
            FOI_det,
            FOI_undet,
            self.total_size)


    def __call__(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''
        Q_ext_det, Q_ext_undet = self.external_matrices(H)
        dH = (H.T * (self.Q_int + Q_ext_det + Q_ext_undet)).T
        return dH
