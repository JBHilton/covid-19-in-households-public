'''This plots the results from the exernal isolation working example.
'''
from os import mkdir
from os.path import isdir
from pickle import load
from matplotlib.pyplot import subplots, yscale
from matplotlib.cm import get_cmap

if isdir('plots/oohi') is False:
    mkdir('plots/oohi')


with open('outputs/oohi/results.pkl', 'rb') as f:
    model_input, I_OOHI,R_OOHI,Q_OOHI,OOHI_time,I_WHQ,R_WHQ,Q_WHQ,WHQ_time = load(f)
class_per_hh = model_input.ave_hh_by_class
lgd=['Children','Non-vulnerable adults','Vulnerable adults']

#clist=0.5*ones(10,3)
#clist(:,1)=(1/11)*(1:10)
#clist(:,3)=(1/11)*(10:-1:1)

fig, ax = subplots()

alpha = 0.5
ax.plot(
    WHQ_time, I_WHQ[:,2]/class_per_hh[2], label='Prevalence in vulnerable population under WHI',
    alpha=alpha, linewidth=2)
ax.plot(
    OOHI_time, I_OOHI[:,2]/class_per_hh[2], label='Prevalence in vulnerable population under OOHI',
    alpha=alpha, linewidth=2)
ax.plot(
    WHQ_time, Q_WHQ.sum(1)/model_input.ave_hh_size, label='Total quarantining population under WHI',
    alpha=alpha, linewidth=2)
ax.plot(
    OOHI_time, Q_OOHI.sum(1)/model_input.ave_hh_size, label='Total quarantining population under OOHI',
    alpha=alpha, linewidth=2)

ax.legend(ncol=1, bbox_to_anchor=(1,0.50))
ax.set_xlabel('Time in days')
ax.set_ylabel('Size of population')
# ax.set_ylim([0,1e-2])
yscale('log')

fig.savefig('plots/oohi/time_series.png', bbox_inches='tight', dpi=300)



fig, ax= subplots()
