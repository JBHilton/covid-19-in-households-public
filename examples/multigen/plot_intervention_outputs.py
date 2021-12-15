'''This plots the UK-like model with a single set of parameters for 100 days
'''
from abc import ABC
from os import mkdir
from os.path import isdir
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from numpy import where

class Outputs(ABC):
    def __init__(self, name):
        self.name = name

if isdir('plots/ov_shielding') is False:
    mkdir('plots/ov_shielding')

with open('outputs/ov_shielding/results_25_25.pkl', 'rb') as f:
    baseline, whh_red, bhh_red, both_red, targ = load(f)
with open('outputs/ov_shielding/fitted_model_input.pkl', 'rb') as f:
    model_input = load(f)

fig, (ax_prev, ax_cum) = subplots(2,1,constrained_layout=True)

ax_prev.plot(baseline.t, 100 * baseline.P_allhh+100 * baseline.I_allhh,label='All hh')
ax_prev.plot(baseline.t, 100 * baseline.P_001+100 * baseline.I_001,label='Single gen')
ax_prev.plot(baseline.t, 100 * baseline.P_011+100 * baseline.I_011,label='65+ with 20-64')
ax_prev.plot(baseline.t, 100 * baseline.P_101+100 * baseline.I_101,label='65+ with 0-19')
ax_prev.plot(baseline.t, 100 * baseline.P_111+100 * baseline.I_111,label='Three gen')
ax_prev.set_ylim([0, 45])
ax_prev.set_xlabel('Time in days')
ax_prev.set_ylabel('Prevalence in 65+ class')

ax_cum.plot(baseline.t, 100 * baseline.R_allhh,label='All hh')
ax_cum.plot(baseline.t, 100 * baseline.R_001,label='Single gen')
ax_cum.plot(baseline.t, 100 * baseline.R_011,label='65+ with 20-64')
ax_cum.plot(baseline.t, 100 * baseline.R_101,label='65+ with 0-19')
ax_cum.plot(baseline.t, 100 * baseline.R_111,label='Three gen')
ax_cum.set_ylim([0, 100])
ax_cum.set_xlabel('Time in days')
ax_cum.set_ylabel('Cumulative prevalence\n in 65+ class')
ax_prev.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.suptitle('No interventions')

fig.savefig('plots/ov_shielding/no_measures_proj.png', bbox_inches='tight')

fig, (ax_prev, ax_cum) = subplots(2,1,constrained_layout=True)

ax_prev.plot(whh_red.t, 100 * whh_red.P_allhh+100 * whh_red.I_allhh,label='All hh')
ax_prev.plot(whh_red.t, 100 * whh_red.P_001+100 * whh_red.I_001,label='Single gen')
ax_prev.plot(whh_red.t, 100 * whh_red.P_011+100 * whh_red.I_011,label='65+ with 20-64')
ax_prev.plot(whh_red.t, 100 * whh_red.P_101+100 * whh_red.I_101,label='65+ with 0-19')
ax_prev.plot(whh_red.t, 100 * whh_red.P_111+100 * whh_red.I_111,label='Three gen')
ax_prev.set_ylim([0, 45])
ax_prev.set_xlabel('Time in days')
ax_prev.set_ylabel('Prevalence in 65+ class')

ax_cum.plot(whh_red.t, 100 * whh_red.R_allhh,label='All hh')
ax_cum.plot(whh_red.t, 100 * whh_red.R_001,label='Single gen')
ax_cum.plot(whh_red.t, 100 * whh_red.R_011,label='65+ with 20-64')
ax_cum.plot(whh_red.t, 100 * whh_red.R_101,label='65+ with 0-19')
ax_cum.plot(whh_red.t, 100 * whh_red.R_111,label='Three gen')
ax_cum.set_ylim([0, 100])
ax_cum.set_xlabel('Time in days')
ax_cum.set_ylabel('Cumulative prevalence\n in 65+ class')
ax_prev.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.suptitle('25% reduction in within-hh mixing')

fig.savefig('plots/ov_shielding/whh_measures_proj_25.png', bbox_inches='tight')

fig, (ax_prev, ax_cum) = subplots(2,1,constrained_layout=True)

ax_prev.plot(bhh_red.t, 100 * bhh_red.P_allhh+100 * bhh_red.I_allhh,label='All hh')
ax_prev.plot(bhh_red.t, 100 * bhh_red.P_001+100 * bhh_red.I_001,label='Single gen')
ax_prev.plot(bhh_red.t, 100 * bhh_red.P_011+100 * bhh_red.I_011,label='65+ with 20-64')
ax_prev.plot(bhh_red.t, 100 * bhh_red.P_101+100 * bhh_red.I_101,label='65+ with 0-19')
ax_prev.plot(bhh_red.t, 100 * bhh_red.P_111+100 * bhh_red.I_111,label='Three gen')
ax_prev.set_ylim([0, 45])
ax_prev.set_xlabel('Time in days')
ax_prev.set_ylabel('Prevalence in 65+ class')

ax_cum.plot(bhh_red.t, 100 * bhh_red.R_allhh,label='All hh')
ax_cum.plot(bhh_red.t, 100 * bhh_red.R_001,label='Single gen')
ax_cum.plot(bhh_red.t, 100 * bhh_red.R_011,label='65+ with 20-64')
ax_cum.plot(bhh_red.t, 100 * bhh_red.R_101,label='65+ with 0-19')
ax_cum.plot(bhh_red.t, 100 * bhh_red.R_111,label='Three gen')
ax_cum.set_ylim([0, 100])
ax_cum.set_xlabel('Time in days')
ax_cum.set_ylabel('Cumulative prevalence\n in 65+ class')
ax_prev.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.suptitle('25% reduction in between-hh mixing')

fig.savefig('plots/ov_shielding/bhh_measures_proj_25.png', bbox_inches='tight')

fig, (ax_prev, ax_cum) = subplots(2,1,constrained_layout=True)

ax_prev.plot(both_red.t, 100 * both_red.P_allhh+100 * both_red.I_allhh,label='All hh')
ax_prev.plot(both_red.t, 100 * both_red.P_001+100 * both_red.I_001,label='Single gen')
ax_prev.plot(both_red.t, 100 * both_red.P_011+100 * both_red.I_011,label='65+ with 20-64')
ax_prev.plot(both_red.t, 100 * both_red.P_101+100 * both_red.I_101,label='65+ with 0-19')
ax_prev.plot(both_red.t, 100 * both_red.P_111+100 * both_red.I_111,label='Three gen')
ax_prev.set_ylim([0, 45])
ax_prev.set_xlabel('Time in days')
ax_prev.set_ylabel('Prevalence in 65+ class')

ax_cum.plot(both_red.t, 100 * both_red.R_allhh,label='All hh')
ax_cum.plot(both_red.t, 100 * both_red.R_001,label='Single gen')
ax_cum.plot(both_red.t, 100 * both_red.R_011,label='65+ with 20-64')
ax_cum.plot(both_red.t, 100 * both_red.R_101,label='65+ with 0-19')
ax_cum.plot(both_red.t, 100 * both_red.R_111,label='Three gen')
ax_cum.set_ylim([0, 100])
ax_cum.set_xlabel('Time in days')
ax_cum.set_ylabel('Cumulative prevalence\n in 65+ class')
ax_prev.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.suptitle('25% reduction on both levels of mixing')

fig.savefig('plots/ov_shielding/both_measures_proj_25_25.png', bbox_inches='tight')

fig, (ax_prev, ax_cum) = subplots(2,1,constrained_layout=True)

ax_prev.plot(targ.t, 100 * targ.P_allhh+100 * targ.I_allhh,label='All hh')
ax_prev.plot(targ.t, 100 * targ.P_001+100 * targ.I_001,label='Single gen')
ax_prev.plot(targ.t, 100 * targ.P_011+100 * targ.I_011,label='65+ with 20-64')
ax_prev.plot(targ.t, 100 * targ.P_101+100 * targ.I_101,label='65+ with 0-19')
ax_prev.plot(targ.t, 100 * targ.P_111+100 * targ.I_111,label='Three gen')
ax_prev.set_ylim([0, 45])
ax_prev.set_xlabel('Time in days')
ax_prev.set_ylabel('Prevalence in 65+ class')

ax_cum.plot(targ.t, 100 * targ.R_allhh,label='All hh')
ax_cum.plot(targ.t, 100 * targ.R_001,label='Single gen')
ax_cum.plot(targ.t, 100 * targ.R_011,label='65+ with 20-64')
ax_cum.plot(targ.t, 100 * targ.R_101,label='65+ with 0-19')
ax_cum.plot(targ.t, 100 * targ.R_111,label='Three gen')
ax_cum.set_ylim([0, 100])
ax_cum.set_xlabel('Time in days')
ax_cum.set_ylabel('Cumulative prevalence\n in 65+ class')
ax_prev.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.suptitle('Targetted 25% reduction in between-hh mixing')

fig.savefig('plots/ov_shielding/targ_measures_proj_25.png', bbox_inches='tight')
