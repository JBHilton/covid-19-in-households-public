'''Module containing model specifications'''

DEFAULT_SPEC = {
    # Interpretable parameters:
    'R0': 2.4,                      # Reproduction number
    'gamma': 0.5,                   # Some scaling parameter?
    'alpha': 0.2,                   # Some scaling parameter?
    'tau': 0.0,                     # Rate of asymptomatic transmission?
    'det_model': {
        'type': 'scaled',           # 'constant' and 'scaled' are the two options
        'max_det_fraction': 0.9     # Cap for detected cases (90%)
    }
    # TODO: Consider adding input files here: xlsx, csv etc.
}
