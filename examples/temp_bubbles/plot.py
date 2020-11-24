'''This plots the bubble results
'''
from pickle import load
from numpy import hstack, where
from matplotlib.pyplot import subplots
from examples.temp_bubbles.common import DataObject

with open('switching_results.pkl', 'rb') as f:
    output_data = load(f)

baseline_dec_onwards = where(output_data.baseline_time>334)

fig, ax = subplots(1, 1, sharex=True)

lgd=['No bubbling','Bubble with single other hh for five days',
    'Bubble with single other hh for 1 day',
    'Bubble with new hh every day for 2 days',
    '... for 3 days',
    '... for 4 days',
    '... for 5 days']

ax.plot(output_data.baseline_time[baseline_dec_onwards], output_data.baseline_R[baseline_dec_onwards], label=lgd[0])
ax.plot(hstack((output_data.merge_t_5day,output_data.postmerge_t_5day)), hstack((output_data.merge_R_5day,output_data.postmerge_R_5day)), label=lgd[1])

for i in range(5):
    ax.plot(hstack((output_data.merge_t_switching[i],output_data.postmerge_t_switching[i])), hstack((output_data.merge_R_switching[i],output_data.postmerge_R_switching[i])), label=lgd[i+2])

ax.set_xlabel('Time in days')
ax.set_ylabel('Recovered')

ax.legend(ncol=1, bbox_to_anchor=(1,0.50))

fig.savefig('R_by_strategy.png', bbox_inches='tight', dpi=300)

fig, ax = subplots(1, 1, sharex=True)

lgd=['No bubbling','Bubble with single other hh for five days',
    'Bubble with single other hh for 1 day',
    'Bubble with new hh every day for 2 days',
    '... for 3 days',
    '... for 4 days',
    '... for 5 days']

ax.plot(output_data.baseline_time[baseline_dec_onwards], output_data.baseline_I[baseline_dec_onwards], label=lgd[0])

merge_5day_to_jan25 = where(output_data.postmerge_t_5day<390)
ax.plot(hstack((output_data.merge_t_5day,output_data.postmerge_t_5day[merge_5day_to_jan25])), hstack((output_data.merge_I_5day,output_data.postmerge_I_5day[merge_5day_to_jan25])), label=lgd[1])

for i in range(5):
    merge_temp_to_jan25 = where(output_data.postmerge_t_switching[i]<390)
    ax.plot(hstack((output_data.merge_t_switching[i],output_data.postmerge_t_switching[i][merge_temp_to_jan25])),
        hstack((output_data.merge_I_switching[i],output_data.postmerge_I_switching[i][merge_temp_to_jan25])), label=lgd[i+2])

ax.set_xlabel('Time in days')
ax.set_ylabel('Prevalence')
ax.set_xlim([359,389])

ax.legend(ncol=1, bbox_to_anchor=(1,0.50))

fig.savefig('peaks_by_strategy.png', bbox_inches='tight', dpi=300)
