'''This plots the bubble results
'''
from pickle import load
from numpy import atleast_2d, hstack, where
from matplotlib.pyplot import subplots
from examples.temp_bubbles.common import DataObject

with open('sweep_results_0.pkl', 'rb') as f:
    unmerged_population,baseline_H, baseline_time, baseline_S, baseline_E, baseline_I, baseline_R = load(f)

ave_hh_size = unmerged_population.composition_distribution.dot(atleast_2d(unmerged_population.composition_list))

fig, ax = subplots(1, 1, sharex=True)

ax.plot(baseline_time, baseline_E, label='E')
ax.plot(baseline_time, baseline_I, label='I')
ax.plot(baseline_time, baseline_R, label='R')
ax.legend(ncol=1, bbox_to_anchor=(1,0.50))
fig.savefig('sweep_baseline_epidemic.png', bbox_inches='tight', dpi=300)

with open('sweep_results_00.pkl', 'rb') as f:
    merged_population2, merged_population3, merged_output = load(f)

fig, ax = subplots(1, 1, sharex=True)

lgd=['No bubbling (continued lockdown)','Policy 1',
    'Policy 2',
    'Policy 3',
    'Policy 4',]


merge_I_1 = (1/3) * merged_output.H_merge_1.T.dot(
    merged_population3.states[:, 2] + merged_population3.states[:, 6] +
    merged_population3.states[:, 10])/ave_hh_size
postmerge_I_1 = merged_output.H_postmerge_1.T.dot(unmerged_population.states[:, 2])/ave_hh_size
merge_R_1 = (1/3) * merged_output.H_merge_1.T.dot(
    merged_population3.states[:, 3] + merged_population3.states[:, 7] +
    merged_population3.states[:,11])/ave_hh_size
postmerge_R_1 = merged_output.H_postmerge_1.T.dot(unmerged_population.states[:, 3])/ave_hh_size

merge_I_3 = (1/2) * merged_output.H_merge_3.T.dot(
    merged_population2.states[:, 2] + merged_population2.states[:, 6])/ave_hh_size
postmerge_I_3 = merged_output.H_postmerge_3.T.dot(unmerged_population.states[:, 2])/ave_hh_size
merge_R_3 = (1/2) * merged_output.H_merge_3.T.dot(
    merged_population2.states[:, 3] + merged_population2.states[:, 7])/ave_hh_size
postmerge_R_3 = merged_output.H_postmerge_3.T.dot(unmerged_population.states[:, 3])/ave_hh_size

ax.plot(baseline_time, baseline_R, label=lgd[0])
ax.plot(hstack((merged_output.t_merge_1,merged_output.t_postmerge_1)),hstack((merge_R_1,postmerge_R_1)), label=lgd[1])
ax.plot(hstack((merged_output.t_merge_3,merged_output.t_postmerge_3)),hstack((merge_R_3,postmerge_R_3)), label=lgd[2])

ax.set_xlabel('Time in days')
ax.set_ylabel('Recovered')

ax.legend(ncol=1, bbox_to_anchor=(1,0.50))

fig.savefig('R_by_strategy_sweep.png', bbox_inches='tight', dpi=300)

fig, ax = subplots(1, 1, sharex=True)

ax.plot(baseline_time, baseline_I, label=lgd[0])
ax.plot(hstack((merged_output.t_merge_1,merged_output.t_postmerge_1)),hstack((merge_I_1,postmerge_I_1)), label=lgd[1])
ax.plot(hstack((merged_output.t_merge_3,merged_output.t_postmerge_3)),hstack((merge_I_3,postmerge_I_3)), label=lgd[2])

ax.set_xlabel('Time in days')
ax.set_ylabel('Infectious')

ax.legend(ncol=1, bbox_to_anchor=(1,0.50))

fig.savefig('I_by_strategy_sweep.png', bbox_inches='tight', dpi=300)

# fig, ax = subplots(1, 1, sharex=True)
#
# ax.plot(output_data.baseline_time[baseline_dec_onwards], output_data.baseline_I[baseline_dec_onwards], label=lgd[0])
#
# merge_5day_to_jan25 = where(output_data.postmerge_t_5day<390)
# ax.plot(hstack((output_data.merge_t_5day,output_data.postmerge_t_5day[merge_5day_to_jan25])), hstack((output_data.merge_I_5day,output_data.postmerge_I_5day[merge_5day_to_jan25])), label=lgd[1])
#
# for i in range(3):
#     merge_temp_to_jan25 = where(output_data.postmerge_t_switching[i]<390)
#     ax.plot(hstack((output_data.merge_t_switching[i],output_data.postmerge_t_switching[i][merge_temp_to_jan25])),
#         hstack((output_data.merge_I_switching[i],output_data.postmerge_I_switching[i][merge_temp_to_jan25])), label=lgd[i+2])
#
# ax.set_xlabel('Time in days')
# ax.set_ylabel('Prevalence')
# ax.set_xlim([359,389])
#
# ax.legend(ncol=1, bbox_to_anchor=(1,0.50))
#
# fig.savefig('peaks_by_strategy_3hh.png', bbox_inches='tight', dpi=300)
