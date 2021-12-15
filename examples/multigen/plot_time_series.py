'''This plots the UK-like model with a single set of parameters for 100 days
'''
from os import mkdir
from os.path import isdir
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap
from numpy import where

if isdir('plots/ov_shielding') is False:
    mkdir('plots/ov_shielding')

with open('outputs/ov_shielding/results.pkl', 'rb') as f:
    (H,
    time_series,
    P_001,
    I_001,
    R_001,
    P_101,
    I_101,
    R_101,
    P_011,
    I_011,
    R_011,
    P_111,
    I_111,
    R_111) = load(f)
with open('outputs/ov_shielding/fitted_model_input.pkl', 'rb') as f:
    model_input = load(f)

t = time_series['time']
t_120_loc = where(t > 120)[0][0]
data_list = [time_series['S']/model_input.ave_hh_by_class,
    time_series['E']/model_input.ave_hh_by_class,
    time_series['P']/model_input.ave_hh_by_class,
    time_series['I']/model_input.ave_hh_by_class,
    time_series['R']/model_input.ave_hh_by_class]

lgd=['S','E','P','I','R']

fig, (axis_C, axis_A, axis_E) = subplots(3,1, sharex=True)

cmap = get_cmap('tab20')
alpha = 0.5
for i in range(len(data_list)):
    axis_C.plot(
        t[:t_120_loc], data_list[i][:t_120_loc,0], label=lgd[i],
        color=cmap(i/len(data_list)), alpha=alpha)
axis_C.set_ylabel('Proportion of population')
axis_C.set_title('0-19 years old')
axis_C.legend(ncol=1, bbox_to_anchor=(1,0.50))

for i in range(len(data_list)):
    axis_A.plot(
        t[:t_120_loc], data_list[i][:t_120_loc,1], label=lgd[i],
        color=cmap(i/len(data_list)), alpha=alpha)
axis_A.set_ylabel('Proportion of population')
axis_A.set_title('20-64 years old')

for i in range(len(data_list)):
    axis_E.plot(
        t[:t_120_loc], data_list[i][:t_120_loc,2], label=lgd[i],
        color=cmap(i/len(data_list)), alpha=alpha)
axis_E.set_ylabel('Proportion of population')
axis_E.set_title('65+ years old')

fig.savefig('plots/ov_shielding/cases_by_class.png', bbox_inches='tight', dpi=300)

fig, ((ax_001, ax_011), (ax_101, ax_111)) = subplots(2,2, sharex=True)

ax_001.plot(t[:t_120_loc], P_001[:t_120_loc]+I_001[:t_120_loc])
ax_011.plot(t[:t_120_loc], P_011[:t_120_loc]+I_011[:t_120_loc])
ax_101.plot(t[:t_120_loc], P_101[:t_120_loc]+I_101[:t_120_loc])
ax_111.plot(t[:t_120_loc], P_111[:t_120_loc]+I_111[:t_120_loc])
ax_001.set_ylim([0, 0.4])
ax_011.set_ylim([0, 0.4])
ax_101.set_ylim([0, 0.4])
ax_111.set_ylim([0, 0.4])
ax_101.set_xlabel('Time in days')
ax_111.set_xlabel('Time in days')
ax_001.set_ylabel('Prevalence in 65+ class')
ax_101.set_ylabel('Prevalence in 65+ class')
ax_001.set_title('Single gen')
ax_011.set_title('65+ with 20-64')
ax_101.set_title('65+ with 0-19')
ax_111.set_title('Three gen')

fig.savefig('plots/ov_shielding/prev_by_gens.png')

fig, ((ax_001, ax_011), (ax_101, ax_111)) = subplots(2,2, sharex=True)

ax_001.plot(t[:t_120_loc], R_001[:t_120_loc])
ax_011.plot(t[:t_120_loc], R_011[:t_120_loc])
ax_101.plot(t[:t_120_loc], R_101[:t_120_loc])
ax_111.plot(t[:t_120_loc], R_111[:t_120_loc])
ax_001.set_ylim([0, 1])
ax_011.set_ylim([0, 1])
ax_101.set_ylim([0, 1])
ax_111.set_ylim([0, 1])
ax_101.set_xlabel('Time in days')
ax_111.set_xlabel('Time in days')
ax_001.set_ylabel('Cumulative prevalence\n in 65+ class')
ax_101.set_ylabel('Cumulative prevalence\n in 65+ class')
ax_001.set_title('Single gen')
ax_011.set_title('65+ with 20-64')
ax_101.set_title('65+ with 0-19')
ax_111.set_title('Three gen')

fig.savefig('plots/ov_shielding/cum_prev_by_gens.png')
