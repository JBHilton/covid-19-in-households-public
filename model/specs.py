'''Module containing model specifications'''

from numpy import array

TWO_AGE_SIR_SPEC = {
    'compartmental_structure': 'SIR', # This is which subsystem key to use
    'AR': 0.45,                     # Secondary attack probability
    'R*': 1.1,                      # Household-level reproduction number
    'recovery_rate': 1/12,           # Recovery rate
    'sus': array([1,1]),          # Relative susceptibility by age/vulnerability class
    'density_expo' : 0.5, # "Cauchemez parameter"
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
    'coarse_bds' : array([0,20])                                # Desired boundaries for model population
}

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
