'''Module containing model specifications'''

from pandas import read_excel, read_csv

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
    'k_home': read_excel(
        'inputs/MUestimates_home_2.xlsx',
        sheet_name='United Kingdom of Great Britain',
        header=None).to_numpy(),
    'k_all': read_excel(
        'inputs/MUestimates_all_locations_2.xlsx',
        sheet_name='United Kingdom of Great Britain',
        header=None).to_numpy(),
    'pop_pyramid': read_csv(
        'inputs/United Kingdom-2019.csv', index_col=0),
    'rho': read_csv(
        'inputs/rho_estimate_cdc.csv', header=None).to_numpy().flatten()
}
