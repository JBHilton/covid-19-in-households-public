'''Module containing model specifications'''

from copy import deepcopy
from numpy import arange, array, multiply, ones
from numpy.random import rand

def draw_random_two_age_SEPIR_specs(spec,
                      AR_pars = [0.25, 0.75],
                      rec_pars = [1, 7],
                      inc_pars = [1, 10],
                      pro_pars = [1, 5],
                      pro_trans_pars = [1, 1],
                      sus_pars = [1, 1],
                      dens_pars = [0, 1],
                      R_pars = [1, 2]):

    rand_spec = deepcopy(spec)
    rand_spec['AR'] = \
        AR_pars[0] +(AR_pars[1] - AR_pars[0]) * rand(1,) # SAR unif, default [0.25, 0.75]
    rand_spec['recovery_rate'] = 1 / (
                                        rec_pars[0] +
                                        (rec_pars[1] - rec_pars[0]) *
                                        rand(1,) ) # Rec rate unif, default [1, 7]
    rand_spec['incubation_rate'] = 1 / (
                                        inc_pars[0] +
                                        (inc_pars[1] - inc_pars[0]) *
                                        rand(1,) ) # Inc rate unif, default [1, 7]
    rand_spec['symp_onset_rate'] = 1 / (
                                        pro_pars[0] +
                                        (pro_pars[1] - pro_pars[0]) *
                                        rand(1,) ) # Onset rate rate unif, default [1, 5]
    rand_spec['prodromal_trans_scaling'] = multiply( array(pro_trans_pars),
                                            rand(len(pro_trans_pars),) ) # Unif prodrome scalings, default <1
    unscaled_sus = multiply( array(sus_pars), rand(len(sus_pars),) ) # Unif sus scalings
    rand_spec['sus'] = unscaled_sus/unscaled_sus.max() # Set sus of most sus class to 1, all others <1
    rand_spec['density_expo'] = \
        dens_pars[0] +(dens_pars[1] - dens_pars[0]) * rand(1,) # Unif density expo, default [0, 1]
    if spec['fit_method'] == 'R*':
        rand_spec['R*'] = \
            R_pars[0] +(R_pars[1] - R_pars[0]) * rand(1,) # Unif R*, default [1, 2]

    return rand_spec

TWO_AGE_SIR_SPEC = {
    'compartmental_structure': 'SIR', # This is which subsystem key to use
    'AR': 0.45,                     # Secondary attack probability
    'R*': 1.1,                      # Household-level reproduction number
    'recovery_rate': 1/12,           # Recovery rate
    'sus': array([1,1]),          # Relative susceptibility by age/vulnerability class
    'density_expo' : 0.5, # "Cauchemez parameter"
    'fit_method' : 'R*'
}

SINGLE_AGE_SEIR_SPEC = {
    'compartmental_structure': 'SEIR', # This is which subsystem key to use
    'AR': 0.45,                     # Secondary attack probability
    'R*': 1.1,                      # Household-level reproduction number
    'recovery_rate': 1/7,           # Recovery rate
    'incubation_rate': 1/5,         # E->I incubation rate
    'sus': array([1]),          # Relative susceptibility by age/vulnerability class
    'density_expo' : 0.5, # "Cauchemez parameter"
    'fit_method' : 'R*'
}

SINGLE_AGE_SEIR_SPEC_FOR_FITTING = {
    'compartmental_structure': 'SEIR', # This is which subsystem key to use
    'AR': 0.45,                     # Secondary attack probability
    'recovery_rate': 1/7,           # Recovery rate
    'incubation_rate': 1/5,         # E->I incubation rate
    'sus': array([1]),          # Relative susceptibility by age/vulnerability class
    'density_expo' : 0.5, # "Cauchemez parameter"
    'fit_method' : 'EL'
}

TWO_AGE_SEIR_SPEC = {
    'compartmental_structure': 'SEIR', # This is which subsystem key to use
    'AR': 0.45,                     # Secondary attack probability
    'R*': 1.1,                      # Household-level reproduction number
    'recovery_rate': 1/7,           # Recovery rate
    'incubation_rate': 1/5,         # E->I incubation rate
    'sus': array([1,1]),          # Relative susceptibility by age/vulnerability class
    'density_expo' : 0.5, # "Cauchemez parameter"
    'fit_method' : 'R*'
}

TWO_AGE_SEPIR_SPEC = {
    'compartmental_structure': 'SEPIR', # This is which subsystem key to use
    'AR': 0.45,                     # Secondary attack probability
    'R*': 1.1,                      # Household-level reproduction number
    'recovery_rate': 1/4,           # Recovery rate
    'incubation_rate': 1/5,         # E->P incubation rate
    'symp_onset_rate': 1/3,         # P->I prodromal to symptomatic rate
    'prodromal_trans_scaling':
     array([0.5,0.5]),          # Prodromal transmission intensity relative to full inf transmission
    'sus': array([1,1]),          # Relative susceptibility by age/vulnerability class
    'density_expo' : 0.5, # "Cauchemez parameter"
    'fit_method' : 'R*'
}

TWO_AGE_SEPIR_SPEC_FOR_FITTING = {
    'compartmental_structure': 'SEPIR', # This is which subsystem key to use
    'AR': 0.45,                     # Secondary attack probability
    'recovery_rate': 1/4,           # Recovery rate
    'incubation_rate': 1/5,         # E->P incubation rate
    'symp_onset_rate': 1/3,         # P->I prodromal to symptomatic rate
    'prodromal_trans_scaling':
     array([0.5,0.5]),          # Prodromal transmission intensity relative to full inf transmission
    'sus': array([1,1]),          # Relative susceptibility by age/vulnerability class
    'density_expo' : 0.5, # "Cauchemez parameter"
    'fit_method' : 'EL'
}

TWO_AGE_INT_SEPIRQ_SPEC = {
    'compartmental_structure': 'SEPIRQ', # This is which subsystem key to use
    'AR': 0.45,                     # Secondary attack probability
    'R*': 1.1,                      # Household-level reproduction number
    'recovery_rate': 1/4,           # Recovery rate
    'incubation_rate': 1/5,         # E->P incubation rate
    'symp_onset_rate': 1/3,         # P->I prodromal to symptomatic rate
    'exp_iso_rate': 1/1 * ones(2,),  # Ave. time in days to detection by class
    'pro_iso_rate': 1/1 * ones(2,),
    'inf_iso_rate': 1/0.5 * ones(2,),
    'discharge_rate': 1/14,         # 1 / ave time in isolation
    'iso_method': "int",            # This is either "int" or "ext"
    'ad_prob': 1,                   # Probability under internal isolation that household members actually isolate
    'class_is_isolating':
    array([[True, True, True],
           [True, True, True],
           [True, True, True]]), # Element (i,j) is "If someone of class j is present, class i will isolate externally"
    'prodromal_trans_scaling':
     array([0.5,0.5]),          # Prodromal transmission intensity relative to full inf transmission
    'iso_trans_scaling':
     array([1,1]),          # Prodromal transmission intensity relative to full inf transmission
    'sus': array([1,1]),          # Relative susceptibility by age/vulnerability class
    'density_expo' : 0.5, # "Cauchemez parameter"
    'fit_method' : 'R*'
}

TWO_AGE_EXT_SEPIRQ_SPEC = {
    'compartmental_structure': 'SEPIRQ', # This is which subsystem key to use
    'AR': 0.45,                     # Secondary attack probability
    'R*': 1.1,                      # Household-level reproduction number
    'recovery_rate': 1/4,           # Recovery rate
    'incubation_rate': 1/5,         # E->P incubation rate
    'symp_onset_rate': 1/3,         # P->I prodromal to symptomatic rate
    'exp_iso_rate': 1/1 * ones(2,),  # Ave. time in days to detection by class
    'pro_iso_rate': 1/1 * ones(2,),
    'inf_iso_rate': 1/0.5 * ones(2,),
    'discharge_rate': 1/14,         # 1 / ave time in isolation
    'iso_method': "ext",            # This is either "int" or "ext"
    'ad_prob': 0.2,                   # Probability under internal isolation that household members actually isolate
    'class_is_isolating':
    array([[False, False, False],
           [False, False, True],
           [False, False, False]]), # Element (i,j) is "If someone of class j is present, class i will isolate externally"
    'prodromal_trans_scaling':
     array([0.5,0.5]),          # Prodromal transmission intensity relative to full inf transmission
    'iso_trans_scaling':
     array([1,1]),          # Prodromal transmission intensity relative to full inf transmission
    'sus': array([1,1]),          # Relative susceptibility by age/vulnerability class
    'density_expo' : 0.5, # "Cauchemez parameter"
    'fit_method' : 'R*'
}

SINGLE_AGE_UK_SPEC = {
    'k_home': {                                                 # File location for UK within-household contact matrix
        'file_name': 'inputs/MUestimates_home_2.xlsx',
        'sheet_name':'United Kingdom of Great Britain'
    },
    'k_all': {                                                  # File location for UK pop-level contact matrix
        'file_name': 'inputs/MUestimates_all_locations_2.xlsx',
        'sheet_name': 'United Kingdom of Great Britain'
    },
    'pop_pyramid_file_name': 'inputs/United Kingdom-2019.csv',   # File location for UK age pyramid
    'fine_bds' : arange(0,81,5),                                # Boundaries used in pyramid/contact data
    'coarse_bds' : array([0]),                                # Desired boundaries for model population
    'adult_bd' : 1
}

TWO_AGE_UK_SPEC = {
    'k_home': {                                                 # File location for UK within-household contact matrix
        'file_name': 'inputs/MUestimates_home_2.xlsx',
        'sheet_name':'United Kingdom of Great Britain'
    },
    'k_all': {                                                  # File location for UK pop-level contact matrix
        'file_name': 'inputs/MUestimates_all_locations_2.xlsx',
        'sheet_name': 'United Kingdom of Great Britain'
    },
    'pop_pyramid_file_name': 'inputs/United Kingdom-2019.csv',   # File location for UK age pyramid
    'fine_bds' : arange(0,81,5),                                # Boundaries used in pyramid/contact data
    'coarse_bds' : array([0,20]),                                # Desired boundaries for model population
    'adult_bd' : 1
}

VO_SPEC = {
    # Interpretable parameters:
    'R0': 2.4,
    'incubation_rate': 1/1,
    'recovery_rate': 1/9,
    # Age bands used to stratify the vector parameters
    'age_quant_bounds': array([20,60]),
    'asymp_trans_scaling': array([1.0,
                                  1.0,
                                  1.0]),
    'symptom_prob' : array([0.2,
                            0.2,
                            0.2,]),
    # Relative susceptibility
    'sus' : array([1.0,
                   1.0,
                   1.0],),
    'k_home': {
        'file_name': 'inputs/MUestimates_home_1.xlsx',
        'sheet_name': 'Italy'
    },
    'k_all': {
        'file_name': 'inputs/MUestimates_all_locations_1.xlsx',
        'sheet_name': 'Italy'
    },
    'pop_pyramid_file_name': 'inputs/Italy-2019.csv',
    # TODO: Parameter below (rho) may be redundant
    'rho_file_name': 'inputs/rho_estimate_cdc.csv',
    'external_importation': {
        'type': 'exponential',
        'exponent': 1.0e-2,
        # TODO: Parameter below (alpha) needs a better name
        'alpha': 1.0e-5,
    }
}


SEPIRQ_SPEC = {
    # Interpretable parameters:
    'R0': 1.01,                      # Reproduction number
    'recovery_rate': 1/4,           # Recovery rate
    'incubation_rate': 1/5,         # E->P incubation rate
    'symp_onset_rate': 1/3,         # P->I prodromal to symptomatic rate
    'prodromal_trans_scaling':
     array([0.5,0.5,0.5]),          # Prodromal transmission intensity relative to full inf transmission
    'sus': array([1,1,1]),          # Relative susceptibility by age/vulnerability class
    'external_trans_scaling': 0.5,  # Relative intensity of external compared to internal contacts
    'vuln_prop': 2.2/56,            # Total proportion of adults who are shielding
    'k_home': {
        'file_name': 'inputs/MUestimates_home_2.xlsx',
        'sheet_name':'United Kingdom of Great Britain'
    },
    'k_all': {
        'file_name': 'inputs/MUestimates_all_locations_2.xlsx',
        'sheet_name': 'United Kingdom of Great Britain'
    },
    'pop_pyramid_file_name': 'inputs/United Kingdom-2019.csv'
}

CAREHOME_SPEC = {
   # Interpretable parameters:
    'R_carehome': 1.1,                      # Within-carehome reproduction number
    'incubation_rate': 1/1,                   # E->P incubation rate
    'symp_onset_rate': 1/5,                   # P->I prodromal to symptomatic rate
    'recovery_rate': 1/4,                   # Recovery rate
    'prodromal_trans_scaling': array([0.7,0.7,0.7]),           # Prodromal transmission intensity relative to full inf transmission
    'sus': array([1,1,1]),                 # Relative susceptibility by age/vulnerability class
    'empty_rate': array([1/240,0,0]),                    # Rate of bed emptying
    'covid_mortality_rate': array([(0.01)*(1/4),0,0]),        # Coronavirus death rate - death prob times time to death
    'refill_rate': array([1/75, 0, 0]),                      # Rate of bed refilling - 1/(ave. days until refil)
    'inter_home_coupling': 0                # Coupling strength between different care homes, between 0 and 1
}
