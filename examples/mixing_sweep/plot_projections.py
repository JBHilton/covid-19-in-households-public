'''This plots the bubble results
'''
from pickle import load
from numpy import arange, array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import close, colorbar, imshow, set_cmap, subplots
from examples.temp_bubbles.common import DataObject
from seaborn import heatmap
from time import sleep

def plot_bars(max_hh_size,household_population,results):
    R_probs = []
    SAP = zeros((max_hh_size,))
    for hh_size in range(1,max_hh_size+1):
        R_probs.append(zeros((hh_size+1,)))
        for R in range(hh_size+1):
            this_hh_range = where(
            household_population.states.sum(axis=1)==hh_size)
            this_R_range = where(
            (household_population.states.sum(axis=1)==hh_size) &
            (household_population.states[:,4::5].sum(axis=1)==R))[0]
            R_probs[hh_size-1][R] = sum(results.H[this_R_range,-1]) / sum(results.H[this_hh_range,-1])
        SAP[hh_size-1] = sum(arange(1,hh_size+1,1)*R_probs[hh_size-1][1:]/sum(R_probs[hh_size-1][1:]))/hh_size
    return R_probs, SAP

labels = ['Int. red. 0%, ext. red 0%',
        'Int. red. 0%, ext. red 50%',
        'Int. red. 50%, ext. red 0%',
        'Int. red. 25%, ext. red 25%']

fig_pro, ax_pro = subplots()
fig_SAP, ax_SAP = subplots()

with open('mix_sweep_results_AR0.45_intred0.0_extred0.0.pkl', 'rb') as f:
    AR_now, household_population, results = load(f)
ax_pro.plot(results.t-10,100*results.I,label=labels[0])
R_probs, SAP = plot_bars(6, household_population, results)
ax_SAP.plot(range(1,7), SAP, label=labels[0])

with open('mix_sweep_results_AR0.45_intred0.0_extred0.5.pkl', 'rb') as f:
    AR_now, household_population, results = load(f)
ax_pro.plot(results.t-10,100*results.I,label=labels[1])
R_probs, SAP = plot_bars(6, household_population, results)
ax_SAP.plot(range(1,7), SAP, label=labels[1])

with open('mix_sweep_results_AR0.45_intred0.5_extred0.0.pkl', 'rb') as f:
    AR_now, household_population, results = load(f)
ax_pro.plot(results.t-10,100*results.I,label=labels[2])
R_probs, SAP = plot_bars(6, household_population, results)
ax_SAP.plot(range(1,7), SAP, label=labels[2])

with open('mix_sweep_results_AR0.45_intred0.25_extred0.25.pkl', 'rb') as f:
    AR_now, household_population, results = load(f)
ax_pro.plot(results.t-10,100*results.I,label=labels[3])
R_probs, SAP = plot_bars(6, household_population, results)
ax_SAP.plot(range(1,7), SAP, label=labels[3])

ax_pro.legend()
ax_pro.set_xlim([0,90])
ax_pro.set_xlabel('Time in days')
ax_pro.set_ylabel('Percentage prevalence')
ax_pro.set_title('Secondary attack probability=0.45')
fig_pro.savefig('projections_AR_045.png', bbox_inches='tight', dpi=300)

ax_SAP.legend()
ax_SAP.set_xlim([1,6])
ax_SAP.set_xlabel('Household size')
ax_SAP.set_ylabel('Estimated attack ratio')
ax_SAP.set_title('Baseline secondary attack probability=0.45')
fig_SAP.savefig('SAP_AR_045.png', bbox_inches='tight', dpi=300)

close()

fig_pro, ax_pro = subplots()
fig_SAP, ax_SAP = subplots()

fig_pro, ax_pro = subplots()
fig_SAP, ax_SAP = subplots()

with open('mix_sweep_results_AR0.3_intred0.0_extred0.0.pkl', 'rb') as f:
    AR_now, household_population, results = load(f)
ax_pro.plot(results.t-10,100*results.I,label=labels[0])
R_probs, SAP = plot_bars(6, household_population, results)
ax_SAP.plot(range(1,7), SAP, label=labels[0])

with open('mix_sweep_results_AR0.3_intred0.0_extred0.5.pkl', 'rb') as f:
    AR_now, household_population, results = load(f)
ax_pro.plot(results.t-10,100*results.I,label=labels[1])
R_probs, SAP = plot_bars(6, household_population, results)
ax_SAP.plot(range(1,7), SAP, label=labels[1])

with open('mix_sweep_results_AR0.3_intred0.5_extred0.0.pkl', 'rb') as f:
    AR_now, household_population, results = load(f)
ax_pro.plot(results.t-10,100*results.I,label=labels[2])
R_probs, SAP = plot_bars(6, household_population, results)
ax_SAP.plot(range(1,7), SAP, label=labels[2])

with open('mix_sweep_results_AR0.3_intred0.25_extred0.25.pkl', 'rb') as f:
    AR_now, household_population, results = load(f)
ax_pro.plot(results.t-10,100*results.I,label=labels[3])
R_probs, SAP = plot_bars(6, household_population, results)
ax_SAP.plot(range(1,7), SAP, label=labels[3])

ax_pro.legend()
ax_pro.set_xlim([0,90])
ax_pro.set_xlabel('Time in days')
ax_pro.set_ylabel('Percentage prevalence')
ax_pro.set_title('Secondary attack probability=0.3')
fig_pro.savefig('projections_AR_03.png', bbox_inches='tight', dpi=300)

ax_SAP.legend()
ax_SAP.set_xlim([1,6])
ax_SAP.set_xlabel('Household size')
ax_SAP.set_ylabel('Estimated attack ratio')
ax_SAP.set_title('Baseline secondary attack probability=0.3')
fig_SAP.savefig('SAP_AR_03.png', bbox_inches='tight', dpi=300)

close()

fig_pro, ax_pro = subplots()
fig_SAP, ax_SAP = subplots()

with open('mix_sweep_results_AR0.15_intred0.0_extred0.0.pkl', 'rb') as f:
    AR_now, household_population, results = load(f)
ax_pro.plot(results.t-10,100*results.I,label=labels[0])
R_probs, SAP = plot_bars(6, household_population, results)
ax_SAP.plot(range(1,7), SAP, label=labels[0])

with open('mix_sweep_results_AR0.15_intred0.0_extred0.5.pkl', 'rb') as f:
    AR_now, household_population, results = load(f)
ax_pro.plot(results.t-10,100*results.I,label=labels[1])
R_probs, SAP = plot_bars(6, household_population, results)
ax_SAP.plot(range(1,7), SAP, label=labels[1])

with open('mix_sweep_results_AR0.15_intred0.5_extred0.0.pkl', 'rb') as f:
    AR_now, household_population, results = load(f)
ax_pro.plot(results.t-10,100*results.I,label=labels[2])
R_probs, SAP = plot_bars(6, household_population, results)
ax_SAP.plot(range(1,7), SAP, label=labels[2])

with open('mix_sweep_results_AR0.15_intred0.25_extred0.25.pkl', 'rb') as f:
    AR_now, household_population, results = load(f)
ax_pro.plot(results.t-10,100*results.I,label=labels[3])
R_probs, SAP = plot_bars(6, household_population, results)
ax_SAP.plot(range(1,7), SAP, label=labels[3])

ax_pro.legend()
ax_pro.set_xlim([0,90])
ax_pro.set_xlabel('Time in days')
ax_pro.set_ylabel('Percentage prevalence')
ax_pro.set_title('Secondary attack probability=0.15')
fig_pro.savefig('projections_AR_015.png', bbox_inches='tight', dpi=300)

ax_SAP.legend()
ax_SAP.set_xlim([1,6])
ax_SAP.set_xlabel('Household size')
ax_SAP.set_ylabel('Estimated attack ratio')
ax_SAP.set_title('Baseline secondary attack probability=0.15')
fig_SAP.savefig('SAP_AR_015.png', bbox_inches='tight', dpi=300)

close()
