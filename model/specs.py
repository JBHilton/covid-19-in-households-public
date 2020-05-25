'''Module containing model specifications'''

DEFAULT_SPEC = {
    # Interpretable parameters:
    'R0': 2.4,                      # Reproduction number
    'gamma': 0.5,                   # Mean infectious period
    'alpha': 0.2,                   # Incubation period
    'tau': 0.0,                     # Asymptomatic transmission intensity relative to symptomatic rate
    'det_model': {
        'type': 'scaled',           # 'constant' and 'scaled' are the two options
        'max_det_fraction': 0.9     # Cap for detected cases (90%)
    }
    # TODO: Consider adding input files here: xlsx, csv etc.
}
