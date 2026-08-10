# Compartmental household epidemiological models with age-structure

This repository is an implementation of household [compartmental
models](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology).
The models incorporate demographic information through explicit household
composition. The demographic information is typically age-structure but this
can be generalised into any risk-structure e.g. patient, staff, shielded
or non-shielded household member. The purpose of this software is to enable
quantitative evaluation of health policies in wide range of scenarios.

The code contains a set of functions and classes as well several example
analyses with heterogeneous risk-structure. Functions and classes automate the
construction of objects defining household populations and rate equations.

For detailed information on how to use this code refer to the [wiki
pages](https://github.com/JBHilton/covid-19-in-households-public/wiki). For
methodological and background information please refer to the following

1. Hilton, J and Keeling, M.J., [Incorporating household structure and demography
into models of endemic
disease](https://royalsocietypublishing.org/doi/10.1098/rsif.2019.0317), J. R.
Soc. Interface 16, 2019
1. Hilton, J., [Stochastic approaches to infectious disease in heterogeneous
populations](http://wrap.warwick.ac.uk/147970/), PhD Thesis, University of
Warwick.
