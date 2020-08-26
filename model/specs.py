'''Module containing model specifications'''

from numpy import array

DEFAULT_SPEC = {
    # Interpretable parameters:
    'R0': 2.4,                      # Reproduction number
    'gamma': 0.5,                   # Mean infectious period
    'alpha': 0.2,                   # Incubation period
    'tau': 0.0,                     # Asymptomatic transmission intensity relative to symptomatic rate
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
    'R0': 2.4,                      # Reproduction number
    'gamma': 0.5,                   # Mean infectious period
    'alpha': 0.2,                   # Incubation period
    'tau': 0.0,                     # Asymptomatic transmission intensity relative to symptomatic rate
    'det_model': {
        'type': 'scaled',           # 'constant' and 'scaled' are the two options
        'max_det_fraction': 0.9     # Cap for detected cases (90%)
    },
    # These represent input files for the model. We can make it more flexible
    # in the future, but certain structure of input files must be assumed.
    # Check ModelInput class in model/preprocessing.py to see what assumptions
    # are used now.
    'k_home': {
        'file_name': 'inputs/MUestimates_home_1.xlsx',
        'sheet_name':'Italy'
    },
    'k_all': {
        'file_name': 'inputs/MUestimates_all_locations_1.xlsx',
        'sheet_name': 'Italy'
    },
    'pop_pyramid_file_name': 'inputs/Italy-2019.csv',
    'rho_file_name': 'inputs/rho_estimate_cdc.csv'
}


SEPIRQ_SPEC = {
    # Interpretable parameters:
    'R0': 1.1,                      # Reproduction number
    'gamma': 1/4,                   # Recovery rate
    'alpha_1': 1/1,                   # E->P incubation rate
    'alpha_2': 1/5,                   # P->I prodromal to symptomatic rate
    'tau': array([0.5,0.5,0.5]),           # Prodromal transmission intensity relative to full inf transmission
    'sus': array([1,1,1]),                 # Relative susceptibility by age/vulnerability class
    'epsilon': 0.5,                  # Relative intensity of external compared to internal contacts
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
