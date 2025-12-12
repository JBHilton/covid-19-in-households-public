'''This runs the UK-like model with a single set of parameters for 100 days
'''
from copy import deepcopy
from matplotlib.pyplot import subplots
from numpy import arange, array, diag, log, ones, vstack, where, zeros
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import concat, read_csv
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm
from scipy.special import kl_div
from scipy.stats import binom
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate, estimate_hh_reproductive_ratio,
        SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import SINGLE_AGE_SEIR_SPEC, SEIR_VACC_SPEC, VACC_INF_RED, VACC_SUS_RED
from model.common import UnloopedSEIRRateEquations
from model.imports import FixedImportModel, NoImportModel, ExponentialImportModel

# pylint: disable=invalid-name

if isdir('outputs/vaccine-uptake') is False:
    mkdir('outputs/vaccine-uptake')

if isdir('plots/vaccine-uptake') is False:
    mkdir('plots/vaccine-uptake')

VACC_INF_RED = .9
VACC_SUS_RED = .9

SEIR_VACC_SPEC['sus'] = array([1, 1 - VACC_SUS_RED])
SEIR_VACC_SPEC['inf'] = [array([1, 1 - VACC_INF_RED])]

DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

# Composition list should be all possible combinations of unvaccinated and vaccinated:
uptake_df = concat([read_csv('inputs/hh_vax_1.csv').drop(["Unnamed: 0"], axis = 1),
                    read_csv('inputs/hh_vax_2.csv').drop(["Unnamed: 0", "week"], axis = 1),
                    read_csv('inputs/hh_vax_3.csv').drop(["Unnamed: 0", "week"], axis = 1),
                    read_csv('inputs/hh_vax_4.csv').drop(["Unnamed: 0", "week"], axis = 1),
                    read_csv('inputs/hh_vax_5.csv').drop(["Unnamed: 0", "week"], axis = 1),
                    read_csv('inputs/hh_vax_6.csv').drop(["Unnamed: 0", "week"], axis = 1)],
                          axis = 1)

hh_size_by_uptake_column = array(uptake_df.drop('week', axis=1).columns.str.extract('H=(\d+)')).astype(int)
vacc_by_uptake_column = array(uptake_df.drop('week', axis=1).columns.str.extract('V=(\d+)')).astype(int)
max_hh_size = hh_size_by_uptake_column.max()
n_wks = uptake_df.shape[0]

fig_raw, axs = subplots(3,
                           2,
                           figsize = (10, 8),
                          sharex=True)
for hh_size in range(1, max_hh_size + 1):
    where_size = where(hh_size_by_uptake_column == hh_size)[0]
    udf = uptake_df.iloc[:, 1 + where_size]
    udf.columns = vacc_by_uptake_column[where_size].flatten().tolist()
    udf.plot(kind = 'line',
             ax=axs[int((hh_size - 1) / 2), ((hh_size - 1) % 2)],
             legend = False,
             title = 'HH size =' + str(hh_size)
             )
axs[2, 1].legend(bbox_to_anchor=(1.1, 1.05))
fig_raw.tight_layout()
fig_raw.show()
fname = 'plots/vaccine-uptake/raw_uptake_data.png'
fig_raw.savefig(fname, bbox_inches='tight', dpi=300)


n_combinations = len(vacc_by_uptake_column)
composition_list = vstack([array([hh_size_by_uptake_column[i] - vacc_by_uptake_column[i],
                                  vacc_by_uptake_column[i]]).T for i in range(n_combinations)])



# Start with some exploratory analysis with the uptake by time

uptake_by_time = [0] * uptake_df.shape[0]
uptake_by_time_by_size = zeros((max_hh_size, uptake_df.shape[0]))
for t in range(uptake_df.shape[0]):
    composition_dist = uptake_df.drop('week', axis=1).iloc[t].to_numpy()
    composition_dist = composition_dist / sum(composition_dist)

    uptake_by_time[t] = (composition_dist.dot(composition_list[:, 1]) / composition_dist.dot(hh_size_by_uptake_column))[
        0]

    for hh_size in range(1, max_hh_size + 1):
        where_size = where(hh_size_by_uptake_column == hh_size)[0]
        p_hh_size = sum(composition_dist[where_size])
        uptake_by_time_by_size[hh_size-1, t] = \
        (composition_dist[where_size].dot(composition_list[where_size, 1]) / (p_hh_size * hh_size))

# Quick plot of uptake by household size
fig_uptake, ax = subplots(1,
                          1,
                          sharex=True,
                          sharey=True)
for i in range(max_hh_size):
    ax.plot(uptake_by_time_by_size[i, :], label = 'N ='+ str(i+1))
    ax.set_ylabel('Uptake')
    ax.set_xlabel('Week')
ax.plot(uptake_by_time, ls = ":", color = "k", label = 'Overall')
ax.legend()
fig_uptake.show()

fname = 'plots/vaccine-uptake/uptake.png'
fig_uptake.savefig(fname, bbox_inches='tight', dpi=300)


# Proportion of households with zero doses under observed, IRV, and household vaccination
def get_zero_dose_prop(t, hh_size):
    # Get position in composition_list of [hh_size, 0]
    zero_dose_loc = where(abs(composition_list - array([hh_size, 0])).max(1) < .1)[0]

    composition_dist = uptake_df.drop('week', axis=1).iloc[t].to_numpy()
    composition_dist = composition_dist / sum(composition_dist)

    # Condition on household size
    p_hh_size = sum(composition_dist[where(hh_size_by_uptake_column==hh_size)[0]])

    return composition_dist[zero_dose_loc][0] / p_hh_size

zero_dose_by_size = array([[get_zero_dose_prop(t, hh_size) for t in range(n_wks)]
                           for hh_size in range(1, max_hh_size + 1)])

zero_dose_irv = array([(1 - uptake_by_time_by_size[hh_size-1, :])**hh_size
                       for hh_size in range(1, max_hh_size + 1)])

# Proportion of households with zero unvaccinated under observed, IRV, and household vaccination
def get_full_vacc_prop(t, hh_size):
    # Get position in composition_list of [hh_size, 0]
    full_vacc_loc = where(abs(composition_list - array([0, hh_size])).max(1) < .1)[0]

    composition_dist = uptake_df.drop('week', axis=1).iloc[t].to_numpy()
    composition_dist = composition_dist / sum(composition_dist)

    # Condition on household size
    p_hh_size = sum(composition_dist[where(hh_size_by_uptake_column == hh_size)[0]])

    return composition_dist[full_vacc_loc][0] / p_hh_size

full_vacc_by_size = array([[get_full_vacc_prop(t, hh_size) for t in range(n_wks)]
                           for hh_size in range(1, max_hh_size + 1)])
full_vacc_irv = array([uptake_by_time_by_size[hh_size-1, :]**hh_size
                       for hh_size in range(1, max_hh_size + 1)])

# Plots of zero/all vacc

fig_zero_full, ax = subplots(max_hh_size-1,
                             2,
                             figsize = (6, 10),
                             sharex=True,
                             sharey=True)

# Don't bother plotting size 1 households:
for i in range(max_hh_size - 1):
    ax[i, 0].plot(zero_dose_by_size[i+1, :], label = 'Observed')
    ax[i, 0].plot(zero_dose_irv[i+1, :], label = 'IRV')
    ax[i, 0].plot(1 - uptake_by_time_by_size[i+1, :], label = "Household")
    ax[i, 0].set_ylabel('Proportion')
    
    ax[i, 1].plot(full_vacc_by_size[i+1, :], label = 'Observed')
    ax[i, 1].plot(full_vacc_irv[i+1, :], label = 'IRV')
    ax[i, 1].plot(uptake_by_time_by_size[i+1, :], label = "Household")
    ax[i, 1].text(180, 0.25, 'N =' + str(i+1+1))

ax[0, 0].set_title("Fully unvaccinated")
ax[0, 1].set_title("Fully vaccinated")
ax[max_hh_size-2, 0].set_xlabel("Week")
ax[0, 1].legend(bbox_to_anchor=(1.1, 1.1))

fig_zero_full.tight_layout()

fig_zero_full.show()

fname = 'plots/vaccine-uptake/zero_full_uptake.png'
fig_zero_full.savefig(fname, bbox_inches='tight', dpi=300)

# Use KL divergence to check difference between observed distribution and IRV/household
def get_irv_kl_div(t, hh_size):
    obs_dist = uptake_df.drop('week', axis=1).iloc[t].to_numpy()
    obs_dist = obs_dist / sum(obs_dist)

    # Condition on household size
    where_size = where(hh_size_by_uptake_column == hh_size)[0]
    obs_dist = obs_dist[where_size] / sum(obs_dist[where_size])

    irv_dist = binom.pmf(arange(hh_size+1), hh_size, uptake_by_time_by_size[hh_size-1, t])

    return kl_div(obs_dist, irv_dist).sum()


def get_hh_kl_div(t, hh_size):
    zero_dose_loc = where(abs(composition_list - array([hh_size, 0])).max(1) < .1)[0]
    full_vacc_loc = where(abs(composition_list - array([0, hh_size])).max(1) < .1)[0]

    obs_dist = uptake_df.drop('week', axis=1).iloc[t].to_numpy()
    obs_dist = obs_dist / sum(obs_dist)

    hh_dist = zeros(obs_dist.shape)
    hh_dist[zero_dose_loc] = 1 - uptake_by_time_by_size[hh_size-1, t]
    hh_dist[full_vacc_loc] = uptake_by_time_by_size[hh_size-1, t]

    # Condition on household size
    where_size = where(hh_size_by_uptake_column == hh_size)[0]
    obs_dist = obs_dist[where_size] / sum(obs_dist[where_size])

    hh_dist = hh_dist[where_size]
    hh_dist += .01 * binom.pmf(arange(hh_size+1), hh_size, uptake_by_time_by_size[hh_size-1, t]) # Need small amount of mass here for KL to be well-defined
    hh_dist = hh_dist / sum(hh_dist)

    return kl_div(obs_dist, hh_dist).sum()

vstart = where(array(uptake_by_time) > 0)[0][0]

irv_kl_div = array([[get_irv_kl_div(t, hh_size) for t in range(vstart, n_wks)]
                    for hh_size in range(1, max_hh_size + 1)])

hh_kl_div = array([[get_hh_kl_div(t, hh_size) for t in range(vstart, n_wks)]
                    for hh_size in range(1, max_hh_size + 1)])

fig_kl, ax = subplots(max_hh_size - 1,
                      1,
                      figsize=(5, 10),
                      sharex=True,
                      sharey=True)

# Only plotting divergence from IRV since KL doesn't work for household distribution
for i in range(max_hh_size - 1):
    ax[i].plot(irv_kl_div[i + 1, :], label = "IRV")
    ax[i].plot(hh_kl_div[i + 1, :], label = "Household")
    ax[i].set_ylabel('KL divergence')
    ax[i].text(125, 0.25, 'N =' + str(i + 1 + 1))

ax[max_hh_size-2].set_xlabel("Week")
ax[0].legend(bbox_to_anchor=(1., 1.))

fig_kl.tight_layout()

fig_kl.show()
fname = 'plots/vaccine-uptake/kl_divergence.png'
fig_kl.savefig(fname, bbox_inches='tight', dpi=300)

# R* analysis
gr_interval = [-SEIR_VACC_SPEC['recovery_rate'], 1] # Interval used in growth rate estimation
gr_tol = 1e-3 # Absolute tolerance for growth rate estimation
def get_rstar_at_time(t):
    print("t =", t)
    composition_dist = uptake_df.drop('week', axis=1).iloc[t].to_numpy()
    composition_dist = composition_dist / sum(composition_dist)

    uptake = (composition_dist.dot(composition_list[:, 1]) / composition_dist.dot(hh_size_by_uptake_column))[0]

    # If no vaccine is administered, just do the single-type version:
    if uptake==0:
        composition_list_t = composition_list[where(composition_list[:, 1]==0)[0], 0]
        composition_list_t = composition_list_t.reshape(len(composition_list_t), 1)
        composition_dist = composition_dist[where(composition_list[:, 1]==0)[0]]
        n_class = 1
        TEMP_SPEC = {
            'compartmental_structure': 'SEIR',  # This is which subsystem key to use
            'beta_int': .1,  # Internal infection rate
            'density_expo': 1.,
            'recovery_rate': SEIR_VACC_SPEC['recovery_rate'],  # Recovery rate
            'incubation_rate': SEIR_VACC_SPEC['incubation_rate'],  # E->I incubation rate
            'sus': array([1.]),  # Relative susceptibility by risk class
            'inf': [array([1.])],
            'fit_method': 'EL',
            'k_home': ones((1, 1), dtype=float),
            'k_ext': ones((1, 1), dtype=float),
            'skip_ext_scale': True
        }
    else:
        composition_list_t = composition_list
        n_class = 2
        TEMP_SPEC = deepcopy(SEIR_VACC_SPEC)
        TEMP_SPEC["k_ext"] = array([[1 - uptake, uptake],
                                    [1 - uptake, uptake]])


    model_input_t = SEIRInput(TEMP_SPEC,
                                   composition_list_t,
                                   composition_dist)
    household_population_t = HouseholdPopulation(
        composition_list_t, composition_dist, model_input_t)

    rhs_t = UnloopedSEIRRateEquations(model_input_t,
                                    household_population_t,
                                    NoImportModel(4,
                                                  n_class))


    return (composition_dist,
            rhs_t,
            estimate_hh_reproductive_ratio(household_population_t,
                                           rhs_t),
            estimate_growth_rate(household_population_t,
                                 rhs_t,
                                 gr_interval,
                                 gr_tol,
                                 x0=1e-3,
                                 r_min_discount=0.99),
            uptake
            )

results = [get_rstar_at_time(t) for t in range(uptake_df.shape[0])]

rstar_by_time = [r[2][0] for r in results]

# Compare against model with no vaccination
single_type_composition_list = array([[i] for i in range(1, hh_size_by_uptake_column.max()+1)])
single_type_composition_dist = uptake_df.drop('week', axis=1).iloc[0].to_numpy()
single_type_composition_dist = single_type_composition_dist[where(single_type_composition_dist>0)[0]]
single_type_composition_dist = single_type_composition_dist / sum(single_type_composition_dist)

# Can simulate effect of full vaccination by plugging in appropriate sus_scale and inf_scale, or no vaccination by
# setting both to zero
def get_single_type_rstar(sus_red,
                          inf_red):
    SINGLE_TYPE_SPEC = {
        'compartmental_structure': 'SEIR', # This is which subsystem key to use
        'beta_int': .1,                     # Internal infection rate
        'density_expo' : 1.,
        'recovery_rate': SEIR_VACC_SPEC['recovery_rate'],           # Recovery rate
        'incubation_rate': SEIR_VACC_SPEC['incubation_rate'],         # E->I incubation rate
        'sus': array([1. - sus_red]),          # Relative susceptibility by risk class
        'inf': [array([1. - inf_red])],
        'fit_method' : 'EL',
        'k_home': ones((1, 1), dtype=float),
        'k_ext': ones((1, 1), dtype=float),
        'skip_ext_scale' : True
    }

    single_type_input = SEIRInput(SINGLE_TYPE_SPEC,
                                  single_type_composition_list,
                                  single_type_composition_dist)
    single_type_household_population = HouseholdPopulation(
            single_type_composition_list, single_type_composition_dist, single_type_input)

    single_type_rhs = UnloopedSEIRRateEquations(single_type_input,
                                        single_type_household_population,
                                        NoImportModel(4,
                                                      1))
    return (estimate_hh_reproductive_ratio(single_type_household_population,
                                          single_type_rhs),
            single_type_rhs)

print("At zero uptake, single-type model gives R* =",
      get_single_type_rstar(0., 0.)[0][0])
print("At full uptake, single-type model gives R* =",
      get_single_type_rstar(VACC_SUS_RED, VACC_INF_RED)[0][0])

# Function to do homogeneous vaccination
def get_irv_rstar_at_time(t):

    uptake = uptake_by_time[t]

    # If no vaccine is administered, just do the single-type version:
    if uptake==0:
        composition_list_t = single_type_composition_list
        composition_dist = single_type_composition_dist
        n_class = 1
        TEMP_SPEC = {
            'compartmental_structure': 'SEIR',  # This is which subsystem key to use
            'beta_int': .1,  # Internal infection rate
            'density_expo': 1.,
            'recovery_rate': SEIR_VACC_SPEC['recovery_rate'],  # Recovery rate
            'incubation_rate': SEIR_VACC_SPEC['incubation_rate'],  # E->I incubation rate
            'sus': array([1.]),  # Relative susceptibility by risk class
            'inf': [array([1.])],
            'fit_method': 'EL',
            'k_home': ones((1, 1), dtype=float),
            'k_ext': ones((1, 1), dtype=float),
            'skip_ext_scale': True
        }
    else:
        # Work out household size distribution
        composition_list_t = composition_list
        hh_size_by_comp = array([comp.sum() for comp in composition_list_t])
        uptake_dist = uptake_df.drop('week', axis=1).iloc[t].to_numpy()
        hh_size_dist_t = array([sum(uptake_dist[where(hh_size_by_comp==hhs)[0]]) for
                          hhs in range(1, max(hh_size_by_comp) + 1)])
        hh_size_dist_t = hh_size_dist_t / sum(hh_size_dist_t)

        # Set composition distribution to zeros and fill in with binomial uptake probabilities
        composition_dist = zeros(len(composition_list_t))
        for idx, comp in enumerate(composition_list_t):
            p = binom.pmf(comp[1], comp.sum(), uptake) * hh_size_dist_t[comp.sum() - 1]
            composition_dist[idx] = p

        if (composition_dist.sum() - 1) > 1e-9:
            print("Possible error in composition distribution at time",
                  t,
                  "; sum is",
                  composition_dist.sum())


        n_class = 2
        TEMP_SPEC = deepcopy(SEIR_VACC_SPEC)
        TEMP_SPEC["k_ext"] = array([[1 - uptake, uptake],
                                    [1 - uptake, uptake]])


    model_input_t = SEIRInput(TEMP_SPEC,
                                   composition_list_t,
                                   composition_dist)
    household_population_t = HouseholdPopulation(
        composition_list_t, composition_dist, model_input_t)

    rhs_t = UnloopedSEIRRateEquations(model_input_t,
                                    household_population_t,
                                    NoImportModel(4,
                                                  n_class))


    return (composition_dist,
            rhs_t,
            estimate_hh_reproductive_ratio(household_population_t,
                                           rhs_t),
            estimate_growth_rate(household_population_t,
                                 rhs_t,
                                 gr_interval,
                                 gr_tol,
                                 x0=1e-3,
                                 r_min_discount=0.99),
            uptake
            )

results_irv = [get_irv_rstar_at_time(t) for t in range(uptake_df.shape[0])]
rstar_irv_by_time = [r[2][0] for r in results_irv]

# Now do vaccination by household

# Start by getting indices of compositions with all/none vaccinated:
all_u = [comp[1]==0 for comp in composition_list]
all_v = [comp[0]==0 for comp in composition_list]

def get_hh_vacc_rstar_at_time(t):

    uptake = uptake_by_time[t]

    # If no vaccine is administered, just do the single-type version:
    if uptake==0:
        composition_list_t = single_type_composition_list
        composition_dist = single_type_composition_dist
        n_class = 1
        TEMP_SPEC = {
            'compartmental_structure': 'SEIR',  # This is which subsystem key to use
            'beta_int': .1,  # Internal infection rate
            'density_expo': 1.,
            'recovery_rate': SEIR_VACC_SPEC['recovery_rate'],  # Recovery rate
            'incubation_rate': SEIR_VACC_SPEC['incubation_rate'],  # E->I incubation rate
            'sus': array([1.]),  # Relative susceptibility by risk class
            'inf': [array([1.])],
            'fit_method': 'EL',
            'k_home': ones((1, 1), dtype=float),
            'k_ext': ones((1, 1), dtype=float),
            'skip_ext_scale': True
        }
    else:
        # Work out household size distribution
        composition_list_t = composition_list
        hh_size_by_comp = array([comp.sum() for comp in composition_list_t])
        uptake_dist = uptake_df.drop('week', axis=1).iloc[t].to_numpy()
        hh_size_dist_t = array([sum(uptake_dist[where(hh_size_by_comp==hhs)[0]]) for
                                hhs in range(1, max(hh_size_by_comp) + 1)])
        hh_size_dist_t = hh_size_dist_t / sum(hh_size_dist_t)

        # Composition distribution is weighted combination of vacc and unvacc households:
        composition_dist = zeros(len(composition_list_t))
        composition_dist[all_u] = (1 - uptake) * hh_size_dist_t
        composition_dist[all_v] = uptake * hh_size_dist_t

        if (composition_dist.sum() - 1) > 1e-9:
            print("Possible error in composition distribution at time",
                  t,
                  "; sum is",
                  composition_dist.sum())


        n_class = 2
        TEMP_SPEC = deepcopy(SEIR_VACC_SPEC)
        TEMP_SPEC["k_ext"] = array([[1 - uptake, uptake],
                                    [1 - uptake, uptake]])

        # Update uptake with empirical value for checking purposes
        uptake = composition_dist.dot(array([comp[1] for comp in composition_list_t])) / composition_dist.dot(array([comp.sum() for comp in composition_list_t]))


    model_input_t = SEIRInput(TEMP_SPEC,
                                   composition_list_t,
                                   composition_dist)
    household_population_t = HouseholdPopulation(
        composition_list_t, composition_dist, model_input_t)

    rhs_t = UnloopedSEIRRateEquations(model_input_t,
                                    household_population_t,
                                    NoImportModel(4,
                                                  n_class))

    return (composition_dist,
            rhs_t,
            estimate_hh_reproductive_ratio(household_population_t,
                                           rhs_t),
            estimate_growth_rate(household_population_t,
                                 rhs_t,
                                 gr_interval,
                                 gr_tol,
                                 x0=1e-3,
                                 r_min_discount=0.99),
            uptake
            )

results_hh = [get_hh_vacc_rstar_at_time(t) for t in range(uptake_df.shape[0])]
rstar_hh_by_time = [r[2][0] for r in results_hh]
uptake_from_hh_results = [r[4] for r in results_hh]

uptake_err = abs(array(uptake_from_hh_results[33:]) - array(uptake_by_time[33:])) / array(uptake_by_time[33:])
print("Max difference in uptake =", uptake_err.max())

rstar_shortfall = array(rstar_by_time) - array(rstar_irv_by_time)
rstar_benefit = array(rstar_hh_by_time) - array(rstar_by_time)
rstar_irv_benefit = array(rstar_hh_by_time) - array(rstar_irv_by_time)

# Now estimate control threshold
threshold_by_time = 1 - 1 / array(rstar_by_time)
threshold_irv = 1 - 1 / array(rstar_irv_by_time)
threshold_hh = 1 - 1 / array(rstar_hh_by_time)

# Plots
fig, (ax_rstar, ax_threshold) = subplots(1, 2)

ax_rstar.plot(rstar_by_time, label = 'Observed')
ax_rstar.plot(rstar_irv_by_time, label = 'IRV')
ax_rstar.plot(rstar_hh_by_time, label = 'Household')
ax_rstar.set_ylim([0, 14])
ax_rstar.set_xlabel("Week")
ax_rstar.set_ylabel("$R_*$")
ax_rstar.set_box_aspect(1)

ax_threshold.plot(100 * threshold_by_time, label = 'Observed')
ax_threshold.plot(100 * threshold_irv, label = 'IRV')
ax_threshold.plot(100 * threshold_hh, label = 'Household')
ax_threshold.set_ylim([50, 100])
ax_threshold.set_xlabel("Week")
ax_threshold.set_ylabel("Critical control threshold (%)")
ax_threshold.set_box_aspect(1)

ax_threshold.legend()

fig.suptitle(str(100 * VACC_INF_RED) +
              "% inf. reduction, " +
              str(100 * VACC_SUS_RED) +
              "% sus. reduction.")
fig.tight_layout()
fig.show()

fname = 'plots/vaccine-uptake/rstar_in_time_' +\
        str(VACC_INF_RED) +\
        '_' +\
        str(VACC_SUS_RED) +\
        '.png'
fig.savefig(fname, bbox_inches='tight', dpi=300)

