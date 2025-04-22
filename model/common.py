'''Module for additional computations required by the model'''
from numpy import (
    arange, array, asarray, diag, exp, isnan, ix_, ones, outer,
    shape, sum, where, zeros)
from numpy import int64 as my_int
import pdb

from pandas.io.pytables import Fixed
from scipy.sparse import csc_matrix as sparse
from scipy.sparse import diags

from model.imports import ExponentialImportModel, FixedImportModel, NoImportModel
from model.subsystems import subsystem_key

from os import mkdir
from os.path import isdir
if isdir('outputs') is False:
    mkdir('outputs')


def build_external_import_matrix(household_population, FOI):
    '''Gets sparse matrices containing rates of external infection in a
    household of a given type'''

    row = household_population.inf_event_row
    col = household_population.inf_event_col
    inf_class = household_population.inf_event_class
    total_size = len(household_population.which_composition)

    matrix_shape = (total_size, total_size)

    Q_ext = sparse(matrix_shape,)

    vals = FOI[row, inf_class]
    Q_ext += sparse((vals, (row, col)), shape=matrix_shape)

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

class MatrixImportRateEquations:
    '''This class represents a functor for evaluating the rate equations for
    the model with no imports of infection from outside the population. The
    state of the class contains all essential variables. This uses a matrix
    form for the external infection terms, which while mathematically convenient
    does not appear to be computationally efficient, and so use is not
    recommended.'''
    # pylint: disable=invalid-name
    def __init__(self,
                 model_input,
                 household_population,
                 import_model,
                 epsilon=1.0):

        self.compartmental_structure = \
            household_population.compartmental_structure
        self.no_compartments = subsystem_key[self.compartmental_structure][1]
        self.model_input = model_input
        self.household_population = household_population
        self.epsilon = epsilon
        self.Q_int = household_population.Q_int
        self.composition_by_state = household_population.composition_by_state
        self.states_sus_only = \
            household_population.states[:, ::self.no_compartments]
        self.s_present = where(self.states_sus_only.sum(axis=1) > 0)[0]
        self.states_new_cases_only = \
            household_population.states[
                :, model_input.new_case_compartment::self.no_compartments]
        self.inf_compartment_list = \
            subsystem_key[self.compartmental_structure][2]
        self.no_inf_compartments = len(self.inf_compartment_list)
        self.import_model = import_model
        self.ext_matrix_list = []
        self.inf_by_state_list = []
        for ic in range(self.no_inf_compartments):
            self.ext_matrix_list.append(
                diag(model_input.sus).dot(
                    model_input.k_ext).dot(
                        diag(model_input.inf_scales[ic])))
            self.inf_by_state_list.append(household_population.states[
                :, self.inf_compartment_list[ic]::self.no_compartments])
        
        self.total_size = len(household_population.which_composition) 
        self.matrix_shape = (self.total_size, self.total_size)
        self.inf_event_row = household_population.inf_event_row
        self.inf_event_col = household_population.inf_event_col
        self.inf_event_class = household_population.inf_event_class
        
        self.import_rate_mat = sparse(self.matrix_shape,)

        denom = model_input.ave_hh_by_class
        self.outflow_mat = sum(
            [(self.ext_matrix_list[ic].dot((self.inf_by_state_list[ic]/model_input.ave_hh_by_class).T)).T
            for ic in range(self.no_inf_compartments)],
            axis=0)
        
        self.ext_rate_mat_list = []
        self.adjuster_list = []
        for rc in range(self.household_population.no_risk_groups):
            this_sus = self.states_sus_only[:, rc]
            this_class_rows = self.inf_event_row[where(self.inf_event_class==rc)[0]]
            this_class_cols = self.inf_event_col[where(self.inf_event_class==rc)[0]]
            outer_prod = zeros(self.matrix_shape)
            outer_prod[this_class_rows, :] = outer(this_sus[this_class_rows], self.outflow_mat[:, rc])
            self.ext_rate_mat_list.append(sparse(outer_prod))
            adjuster_mat = sparse((ones(len(this_class_rows),), (this_class_rows, this_class_cols)), shape=self.matrix_shape) -\
                sparse((ones(len(this_class_rows),), (this_class_rows, this_class_rows)), shape=self.matrix_shape)
            self.adjuster_list.append(adjuster_mat)
        self.Q_ext = 0 * self.Q_int
        self.Q_import = 0 * self.Q_int
        self.diagonal_idexes = (arange(self.total_size), arange(self.total_size))
        self.import_rates = self.states_sus_only.dot(diag(self.import_model.cases(0)))

    def __call__(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''
        # print("t=",t)
        self.Q_ext *= 0
        self.Q_import *= 0
        self.import_rates = self.states_sus_only.dot(diag(self.import_model.cases(t)))
        for rc in range(self.household_population.no_risk_groups):
            self.Q_ext += diags(self.ext_rate_mat_list[rc].dot(H))*(self.adjuster_list[rc])
            self.Q_import += sparse((self.import_rates[self.inf_event_row[where(self.inf_event_class==rc)[0]], rc],
                                     (self.inf_event_row[where(self.inf_event_class==rc)[0]],
                                     self.inf_event_col[where(self.inf_event_class==rc)[0]])), shape=self.matrix_shape) -\
                                        sparse((self.import_rates[self.inf_event_row[where(self.inf_event_class==rc)[0]], rc],
                                     (self.inf_event_row[where(self.inf_event_class==rc)[0]],
                                     self.inf_event_row[where(self.inf_event_class==rc)[0]])), shape=self.matrix_shape)
        
        # self.Q_ext += sparse((-self.Q_ext.sum(axis=1).getA().squeeze(), self.diagonal_idexes))
        
        # import_rate = self.states_sus_only.dot(self.model_input.k_ext.dot(self.import_model.cases(t)))

        # self.import_rate_mat = 0 * self.import_mat
        # vals = FOI[self.inf_event_row, self.inf_event_class]
        # self.import_rate_mat += sparse((vals, (self.inf_event_row, self.inf_event_col)), shape=self.matrix_shape)

        # diagonal_idexes = (arange(self.total_size), arange(self.total_size))
        # S = self.import_rate_mat.sum(axis=1).getA().squeeze()
        # self.import_rate_mat += sparse((-S, diagonal_idexes))
        if (H < 0).any():
            # pdb.set_trace()
            H[where(H < 0)[0]] = 0
        if isnan(H).any():
            # pdb.set_trace()
            raise ValueError('State vector contains NaNs at t={0}'.format(t))
        # print(self.Q_ext.shape)
        dH = ((H.T * (self.Q_int + self.Q_ext + self.Q_import)).T).squeeze()
        return dH

    def external_matrices(self, t, H):
        FOI = self.get_FOI_by_class(t, H)
        return build_external_import_matrix(
            self.household_population,
            FOI)

    def get_FOI_by_class(self, t, H):
        '''This calculates the age-stratified force-of-infection (FOI) on each
        household composition'''
        # Average number of each class by household
        denom = H.T.dot(self.composition_by_state)

        FOI = self.states_sus_only.dot(diag(self.import_model.cases(t)))

        for ic in range(self.no_inf_compartments):
            states_inf_only = self.inf_by_state_list[ic]
            inf_by_class = zeros(shape(denom))
            inf_by_class[denom > 0] = (
                H.T.dot(states_inf_only)[denom > 0]
                / denom[denom > 0]).squeeze()
            FOI += self.states_sus_only.dot(
                    diag(self.ext_matrix_list[ic].dot(
                        self.epsilon * inf_by_class.T)))

        return FOI
    
def update_ext_matrices(rhs, t, H):
        if (rhs.sources=="ALL")|(rhs.sources=="BETWEEN"):
            rhs.between_hh_rate = rhs.inf_event_sus.dot(diag(rhs.outflow_mat.T.dot(H)))

            rhs.Q_ext -= rhs.Q_ext
            
            rhs.Q_ext += sparse((rhs.between_hh_rate[arange(len(rhs.inf_event_row)), rhs.inf_event_class],
                                        (rhs.inf_event_row,
                                        rhs.inf_event_col)), shape=rhs.matrix_shape) -\
                                        sparse((rhs.between_hh_rate[arange(len(rhs.inf_event_row)), rhs.inf_event_class],
                                        (rhs.inf_event_row,
                                        rhs.inf_event_row)), shape=rhs.matrix_shape)
        if (rhs.sources=="ALL")|(rhs.sources=="IMPORT"):
            # if type(rhs.import_model==NoImportModel):
            #     rhs.Q_import *= 0
            if (type(rhs.import_model)==FixedImportModel)|(type(rhs.import_model)==ExponentialImportModel):
                rhs.Q_import = rhs.import_model.matrix(t)
            else:
                rhs.import_rates =  rhs.inf_event_sus.dot(
                        rhs.model_input.k_ext).dot(diag(rhs.import_model.cases(t)))

                rhs.Q_import -= rhs.Q_import

                rhs.Q_import += sparse((rhs.import_rates[arange(len(rhs.inf_event_row)), rhs.inf_event_class],
                                            (rhs.inf_event_row,
                                            rhs.inf_event_col)), shape=rhs.matrix_shape) -\
                                            sparse((rhs.import_rates[arange(len(rhs.inf_event_row)), rhs.inf_event_class],
                                            (rhs.inf_event_row,
                                            rhs.inf_event_row)), shape=rhs.matrix_shape)

class UnloopedRateEquations:
    '''This class represents a functor for evaluating the rate equations for
    the model with no imports of infection from outside the population. The
    state of the class contains all essential variables. This method uses
    more preallocation in calculating the external infection terms, although
    does not currently appear to be more computationally efficient than the
    RateEquations approach.'''
    # pylint: disable=invalid-name
    def __init__(self,
                 model_input,
                 household_population,
                 import_model,
                 sources="ALL",
                 epsilon=1.0):

        self.compartmental_structure = \
            household_population.compartmental_structure
        self.no_compartments = subsystem_key[self.compartmental_structure][1]
        self.model_input = model_input
        self.household_population = household_population
        self.sources = sources
        self.epsilon = epsilon
        self.Q_int = household_population.Q_int
        self.composition_by_state = household_population.composition_by_state
        self.states_sus_only = \
            household_population.states[:, ::self.no_compartments]
        self.s_present = where(self.states_sus_only.sum(axis=1) > 0)[0]
        self.states_new_cases_only = \
            household_population.states[
                :, model_input.new_case_compartment::self.no_compartments]
        self.inf_compartment_list = \
            subsystem_key[self.compartmental_structure][2]
        self.no_inf_compartments = len(self.inf_compartment_list)
        self.import_model = import_model
        self.ext_matrix_list = []
        self.inf_by_state_list = []
        for ic in range(self.no_inf_compartments):
            self.ext_matrix_list.append(
                diag(model_input.sus).dot(
                    model_input.k_ext).dot(
                        diag(model_input.inf_scales[ic])))
            self.inf_by_state_list.append(household_population.states[
                :, self.inf_compartment_list[ic]::self.no_compartments])
        
        self.total_size = len(household_population.which_composition) 
        self.matrix_shape = (self.total_size, self.total_size)
        self.inf_event_row = household_population.inf_event_row
        self.inf_event_col = household_population.inf_event_col
        self.inf_event_class = household_population.inf_event_class
        
        self.import_rate_mat = sparse(self.matrix_shape,)

        denom = model_input.ave_hh_by_class
        self.outflow_mat = sum(
            [(self.ext_matrix_list[ic].dot((self.inf_by_state_list[ic]/model_input.ave_hh_by_class).T)).T
            for ic in range(self.no_inf_compartments)],
            axis=0)
        
        self.Q_ext = 0 * self.Q_int
        self.Q_import = 0 * self.Q_int
        self.between_hh_rate  = self.states_sus_only.dot(diag(self.import_model.cases(0)))
        self.import_rates = self.states_sus_only.dot(diag(self.import_model.cases(0)))
        self.inf_event_sus = self.states_sus_only[self.inf_event_row, :]

        # These attributes divide the internal dynamics matrix into an infection events component and everything else for
        # convenience when implementing MCMC. Infection events happen at a rate which is linear in the per-contact infection
        # rate and so we can update parameters by multiplying the infection event matrix by a proposed parameter.
        self.Q_int_inf = sparse((array(self.Q_int[self.inf_event_row, self.inf_event_col]).flatten(),
                                        (self.inf_event_row,
                                        self.inf_event_col)), shape=self.matrix_shape) - \
                         sparse((array(self.Q_int[self.inf_event_row, self.inf_event_col]).flatten(),
                                 (self.inf_event_row,
                                  self.inf_event_row)), shape=self.matrix_shape)
        self.Q_int_fixed = self.Q_int - self.Q_int_inf # "Fixed" as in does not change during MCMC routine

        # Set infection rate scalings equal to 1. Changing these scalings during the MCMC routine allows for parameter
        # updates without having to redefine the entire system from the model_input term onwards.
        self.int_rate = 1
        self.ext_rate = 1

    def __call__(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''

        jac = self.jacobian(t, H)
        
        dH = (jac * H).squeeze()
        return dH

    def jacobian(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''

        update_ext_matrices(self, t, H)

        if (H < 0).any():
            # pdb.set_trace()
            H[where(H < 0)[0]] = 0
        if isnan(H).any():
            # pdb.set_trace()
            raise ValueError('State vector contains NaNs at t={0}'.format(t))

        jac = (self.Q_int_fixed +
               self.int_rate * self.Q_int_inf +
               self.ext_rate * self.Q_ext +
               self.ext_rate * self.Q_import).T
        return jac

    def external_matrices(self, t, H):
        if (self.sources=="ALL")|(self.sources=="BETWEEN"):
            self.between_hh_rate = self.inf_event_sus.dot(diag(self.outflow_mat.T.dot(H)))

            self.Q_ext *= 0
            
            self.Q_ext += sparse((self.between_hh_rate[arange(len(self.inf_event_row)), self.inf_event_class],
                                        (self.inf_event_row,
                                        self.inf_event_col)), shape=self.matrix_shape) -\
                                        sparse((self.between_hh_rate[arange(len(self.inf_event_row)), self.inf_event_class],
                                        (self.inf_event_row,
                                        self.inf_event_row)), shape=self.matrix_shape)
        if (self.sources=="ALL")|(self.sources=="IMPORT"):
            self.import_rates =  self.inf_event_sus.dot(diag(self.import_model.cases(t)))

            self.Q_import *= 0

            self.Q_import += sparse((self.import_rates[arange(len(self.inf_event_row)), self.inf_event_class],
                                        (self.inf_event_row,
                                        self.inf_event_col)), shape=self.matrix_shape) -\
                                        sparse((self.import_rates[arange(len(self.inf_event_row)), self.inf_event_class],
                                        (self.inf_event_row,
                                        self.inf_event_row)), shape=self.matrix_shape)


class LogRateEquations:
    '''This class represents a functor for evaluating the rate equations for
    the model with no imports of infection from outside the population. The
    state of the class contains all essential variables. This method uses
    more preallocation in calculating the external infection terms, although
    does not currently appear to be more computationally efficient than the
    RateEquations approach.'''

    # pylint: disable=invalid-name
    def __init__(self,
                 model_input,
                 household_population,
                 import_model,
                 sources="ALL",
                 epsilon=1.0):

        self.compartmental_structure = \
            household_population.compartmental_structure
        self.no_compartments = subsystem_key[self.compartmental_structure][1]
        self.model_input = model_input
        self.household_population = household_population
        self.sources = sources
        self.epsilon = epsilon
        self.Q_int = household_population.Q_int
        self.composition_by_state = household_population.composition_by_state
        self.states_sus_only = \
            household_population.states[:, ::self.no_compartments]
        self.s_present = where(self.states_sus_only.sum(axis=1) > 0)[0]
        self.states_new_cases_only = \
            household_population.states[
            :, model_input.new_case_compartment::self.no_compartments]
        self.inf_compartment_list = \
            subsystem_key[self.compartmental_structure][2]
        self.no_inf_compartments = len(self.inf_compartment_list)
        self.import_model = import_model
        self.ext_matrix_list = []
        self.inf_by_state_list = []
        for ic in range(self.no_inf_compartments):
            self.ext_matrix_list.append(
                diag(model_input.sus).dot(
                    model_input.k_ext).dot(
                    diag(model_input.inf_scales[ic])))
            self.inf_by_state_list.append(household_population.states[
                                          :, self.inf_compartment_list[ic]::self.no_compartments])

        self.total_size = len(household_population.which_composition)
        self.matrix_shape = (self.total_size, self.total_size)
        self.inf_event_row = household_population.inf_event_row
        self.inf_event_col = household_population.inf_event_col
        self.inf_event_class = household_population.inf_event_class

        self.import_rate_mat = sparse(self.matrix_shape, )

        denom = model_input.ave_hh_by_class
        self.outflow_mat = sum(
            [(self.ext_matrix_list[ic].dot((self.inf_by_state_list[ic] / model_input.ave_hh_by_class).T)).T
             for ic in range(self.no_inf_compartments)],
            axis=0)

        self.Q_ext = 0 * self.Q_int
        self.Q_import = 0 * self.Q_int
        self.between_hh_rate = self.states_sus_only.dot(diag(self.import_model.cases(0)))
        self.import_rates = self.states_sus_only.dot(diag(self.import_model.cases(0)))
        self.inf_event_sus = self.states_sus_only[self.inf_event_row, :]

    def __call__(self, t, Z):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''

        print(t)

        H = exp(Z)

        jac = self.jacobian(t, H)

        diffmat = outer(exp(-Z.T), H)

        dZ = (diffmat *  jac.T).sum(0).squeeze()

        # dZ = (exp(-Z).T * (H.T * jac.T)).squeeze()
        return dZ

    def jacobian(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''

        update_ext_matrices(self, t, H)

        # if (H < 0).any():
        #     # pdb.set_trace()
        #     H[where(H < 0)[0]] = 0
        # if isnan(H).any():
        #     # pdb.set_trace()
        #     raise ValueError('State vector contains NaNs at t={0}'.format(t))

        jac = (self.Q_int + self.Q_ext + self.Q_import).T
        return jac

    def external_matrices(self, t, H):
        if (self.sources == "ALL") | (self.sources == "BETWEEN"):
            self.between_hh_rate = self.inf_event_sus.dot(diag(self.outflow_mat.T.dot(H)))

            self.Q_ext *= 0

            self.Q_ext += sparse((self.between_hh_rate[arange(len(self.inf_event_row)), self.inf_event_class],
                                  (self.inf_event_row,
                                   self.inf_event_col)), shape=self.matrix_shape) - \
                          sparse((self.between_hh_rate[arange(len(self.inf_event_row)), self.inf_event_class],
                                  (self.inf_event_row,
                                   self.inf_event_row)), shape=self.matrix_shape)
        if (self.sources == "ALL") | (self.sources == "IMPORT"):
            self.import_rates = self.inf_event_sus.dot(diag(self.import_model.cases(t)))

            self.Q_import *= 0

            self.Q_import += sparse((self.import_rates[arange(len(self.inf_event_row)), self.inf_event_class],
                                     (self.inf_event_row,
                                      self.inf_event_col)), shape=self.matrix_shape) - \
                             sparse((self.import_rates[arange(len(self.inf_event_row)), self.inf_event_class],
                                     (self.inf_event_row,
                                      self.inf_event_row)), shape=self.matrix_shape)


class RateEquations:
    '''This class represents a functor for evaluating the rate equations for
    the model with no imports of infection from outside the population. The
    state of the class contains all essential variables'''
    # pylint: disable=invalid-name
    def __init__(self,
                 model_input,
                 household_population,
                 import_model,
                 epsilon=1.0):

        self.compartmental_structure = \
            household_population.compartmental_structure
        self.no_compartments = subsystem_key[self.compartmental_structure][1]
        self.household_population = household_population
        self.epsilon = epsilon
        self.Q_int = household_population.Q_int
        self.composition_by_state = household_population.composition_by_state
        self.states_sus_only = \
            household_population.states[:, ::self.no_compartments]
        self.s_present = where(self.states_sus_only.sum(axis=1) > 0)[0]
        self.states_new_cases_only = \
            household_population.states[
                :, model_input.new_case_compartment::self.no_compartments]
        self.inf_compartment_list = \
            subsystem_key[self.compartmental_structure][2]
        self.no_inf_compartments = len(self.inf_compartment_list)
        self.import_model = import_model
        self.ext_matrix_list = []
        self.inf_by_state_list = []
        for ic in range(self.no_inf_compartments):
            self.ext_matrix_list.append(
                diag(model_input.sus).dot(
                    model_input.k_ext).dot(
                        diag(model_input.inf_scales[ic])))
            self.inf_by_state_list.append(household_population.states[
                :, self.inf_compartment_list[ic]::self.no_compartments])

    def __call__(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''
        Q_ext = self.external_matrices(t, H)
        if (H < 0).any():
            # pdb.set_trace()
            H[where(H < 0)[0]] = 0
        if isnan(H).any():
            # pdb.set_trace()
            raise ValueError('State vector contains NaNs at t={0}'.format(t))
        dH = (H.T * (self.Q_int + Q_ext)).T
        return dH

    def external_matrices(self, t, H):
        FOI = self.get_FOI_by_class(t, H)
        return build_external_import_matrix(
            self.household_population,
            FOI)

    def get_FOI_by_class(self, t, H):
        '''This calculates the age-stratified force-of-infection (FOI) on each
        household composition'''
        # Average number of each class by household
        denom = H.T.dot(self.composition_by_state)

        FOI = self.states_sus_only.dot((self.household_population.model_input.k_ext).dot(diag(self.import_model.cases(t))))
        # print(asarray(FOI).squeeze().max())


        for ic in range(self.no_inf_compartments):
            states_inf_only = self.inf_by_state_list[ic]
            inf_by_class = zeros(shape(denom))
            inf_by_class[denom > 0] = (
                H.T.dot(states_inf_only)[denom > 0]
                / denom[denom > 0]).squeeze()
            FOI += self.states_sus_only.dot(
                    diag(self.ext_matrix_list[ic].dot(
                        self.epsilon * inf_by_class.T)))

        return FOI


class SIRRateEquations(RateEquations):
    @property
    def states_inf_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 2::self.no_compartments]


class SEIRRateEquations(RateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_inf_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 3::self.no_compartments]
    
class MatrixImportSEIRRateEquations(MatrixImportRateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_inf_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 3::self.no_compartments]
    
class UnloopedSEIRRateEquations(UnloopedRateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_inf_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

class UnloopedSEpIpRpEsIsRsRateEquations(UnloopedRateEquations):
    @property
    def states_exp_p_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_inf_p_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_rec_p_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

    @property
    def states_exp_s_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_inf_s_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_rec_s_only(self):
        return self.household_population.states[:, 3::self.no_compartments]
    
class SEIRCRateEquations(RateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_inf_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

    @property
    def states_cum_only(self):
        return self.household_population.states[:, 4::self.no_compartments]


class SEPIRRateEquations(RateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_pro_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_inf_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 4::self.no_compartments]


class MatrixImportSEPIRRateEquations(MatrixImportRateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_pro_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_inf_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 4::self.no_compartments]


class UnloopedSEPIRRateEquations(UnloopedRateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_pro_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_inf_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 4::self.no_compartments]

class LogSEPIRRateEquations(LogRateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_pro_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_inf_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 4::self.no_compartments]


class SEPIRQRateEquations(RateEquations):
    def __init__(self,
                 model_input,
                 household_population,
                 import_model,
                 epsilon=1.0):
        super().__init__(
            model_input,
            household_population,
            import_model,
            epsilon)

        self.iso_pos = 5 + self.no_compartments * \
            arange(model_input.no_age_classes)
        self.states_iso_only = \
            household_population.states[:, self.iso_pos]

        self.iso_method = model_input.iso_method

        if self.iso_method == "int":
            total_iso_by_state = self.states_iso_only.sum(axis=1)
            self.no_isos = (total_iso_by_state == 0)
            self.isos_present = (total_iso_by_state > 0)
            self.ad_prob = model_input.ad_prob
        else:
            # If we are isolating externally, remove quarantined from inf
            # compartment list
            self.inf_compartment_list = [2, 3]
            self.no_inf_compartments = len(self.inf_compartment_list)
            self.ext_matrix_list = []
            self.inf_by_state_list = []
            for ic in range(self.no_inf_compartments):
                self.ext_matrix_list.append(
                    diag(model_input.sus).dot(
                        model_input.k_ext).dot(
                            diag(model_input.inf_scales[ic])))
                self.inf_by_state_list.append(household_population.states[
                    :, self.inf_compartment_list[ic]::self.no_compartments])

    def get_FOI_by_class(self, t, H):
        '''This calculates the age-stratified force-of-infection (FOI) on each
        household composition'''

        FOI = self.states_sus_only.dot(diag(self.import_model.cases(t)))

        if self.iso_method == 'ext':
            # Under ext. isolation, we need to take iso's away from total
            # household size
            denom = H.T.dot(
                self.composition_by_state - self.states_iso_only)
            for ic in range(self.no_inf_compartments):
                states_inf_only = self.inf_by_state_list[ic]
                inf_by_class = zeros(shape(denom))
                inf_by_class[denom > 0] = (
                    H.T.dot(states_inf_only)[denom > 0]
                    / denom[denom > 0]).squeeze()
                FOI += self.states_sus_only.dot(
                        diag(self.ext_matrix_list[ic].dot(
                            self.epsilon * inf_by_class.T)))
        else:
            # Under internal isoltion, we scale down contribution to infections
            # of any houshold containing Q individuals
            denom = H.T.dot(self.composition_by_state)

            for ic in range(self.no_inf_compartments):
                states_inf_only = self.inf_by_state_list[ic]
                inf_by_class = zeros(shape(denom))
                index = (denom > 0)
                inf_by_class[index] = (
                    (
                        H[where(self.no_isos)[0]].T.dot(
                            states_inf_only[where(self.no_isos)[0], :])[index]
                        + (1 - self.ad_prob)
                        * H[where(self.isos_present)[0]].T.dot(
                            states_inf_only[where(self.isos_present)[0], :])
                    )[index]
                    / denom[index]).squeeze()
                FOI += self.states_sus_only.dot(
                    diag(self.ext_matrix_list[ic].dot(
                        self.epsilon * inf_by_class.T)))

        return FOI

    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_pro_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_inf_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 4::self.no_compartments]


class SEDURRateEquations(RateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_det_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_undet_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 4::self.no_compartments]


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

        no_compartments = 6

        self.household_population = household_population
        # We don't actually use this anywhere but it's very useful to have for
        # debugging purposes
        self.states = household_population.states
        self.Q_int = household_population.Q_int
        # To define external mixing we need to set up the transmission
        # matrices.
        # Scale rows of contact matrix by
        # age-specific susceptibilities
        self.pro_trans_matrix = \
            model_input.sus \
            * model_input.k_ext.dot(diag(model_input.tau))
        # Scale columns by asymptomatic reduction in transmission
        self.inf_trans_matrix = model_input.sus*model_input.k_ext
        # This stores number in each age class by household
        self.composition_by_state = household_population.composition_by_state

        self.states_sus_only = household_population.states[
            :, ::no_compartments]
        self.states_pro_only = household_population.states[
            :, 2::no_compartments]
        self.states_inf_only = household_population.states[
            :, 3::no_compartments]
        self.states_emp_only = household_population.states[
            :, 5::no_compartments]

        self.import_rate = model_input.import_rate
        self.import_model = import_model

        self.epsilon = model_input.epsilon

    def __call__(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''
        if (H < -0).any():
            # if (H<-1e-6).any():
            #     pdb.set_trace()
            H[where(H < 0)[0]] = 0
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
        # Average number of each class by household
        denom = H.T.dot(self.composition_by_state)
        # Average prodromal infected by household in each class
        pro_by_class = zeros(shape(denom))
        # Only want to do states with positive denominator
        pro_by_class[denom > 0] = (
            H.T.dot(self.states_pro_only[ix_(arange(len(H)), denom > 0)])
            / denom[denom > 0]).squeeze()
        # Average full infectious infected by household in each class
        inf_by_class = zeros(shape(denom))
        inf_by_class[denom > 0] = (
            H.T.dot(self.states_inf_only[ix_(range(len(H)), denom > 0)])
            / denom[denom > 0]).squeeze()

        FOI_pro = self.states_sus_only.dot(
            diag(
                self.epsilon
                * self.pro_trans_matrix.dot(pro_by_class.T)
                + self.import_rate.dot(
                    self.import_model.prodromal(t))))
        FOI_inf = self.states_sus_only.dot(
            diag(
                self.epsilon
                * self.inf_trans_matrix.dot(inf_by_class.T)
                + self.import_rate.dot(
                    self.import_model.infected(t))))

        return FOI_pro, FOI_inf
