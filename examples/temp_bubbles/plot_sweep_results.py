'''This plots the bubble results
'''
from pickle import load
from numpy import array, atleast_2d, hstack, where, zeros
from matplotlib.pyplot import close, subplots
from examples.temp_bubbles.common import DataObject
from seaborn import heatmap

no_i_vals = 3
no_j_vals = 3

peaks_1 = zeros((no_i_vals,no_j_vals))
peaks_2 = zeros((no_i_vals,no_j_vals))
peaks_3 = zeros((no_i_vals,no_j_vals))
peaks_4 = zeros((no_i_vals,no_j_vals))

jan_antiprev_1 = zeros((no_i_vals,no_j_vals))
jan_antiprev_2 = zeros((no_i_vals,no_j_vals))
jan_antiprev_3 = zeros((no_i_vals,no_j_vals))
jan_antiprev_4 = zeros((no_i_vals,no_j_vals))

for i in range(no_i_vals):

    filename_stem = 'sweep_results_' + str(i)

    with open(filename_stem + '.pkl', 'rb') as f:
        unmerged_population,baseline_H, baseline_time, baseline_S, baseline_E, baseline_I, baseline_R = load(f)

    ave_hh_size = unmerged_population.ave_hh_size

    fig, ax = subplots(1, 1, sharex=True)

    ax.plot(baseline_time, baseline_E, label='E')
    ax.plot(baseline_time, baseline_I, label='I')
    ax.plot(baseline_time, baseline_R, label='R')
    ax.legend(ncol=1, bbox_to_anchor=(1,0.50))
    fig.savefig('sweep_baseline_epidemic' + str(i) +'.png', bbox_inches='tight', dpi=300)
    close()

    for j in range(no_j_vals):

        filename = filename_stem + str(j)
        with open(filename + '.pkl', 'rb') as f:
            merged_population2, merged_population3, merged_output = load(f)

        fig, ax = subplots(1, 1, sharex=True)

        lgd=['No bubbling','Policy 1',
            'Policy 2',
            'Policy 3',
            'Policy 4',]


        merge_I_1 = (1/3) * merged_output.H_merge_1.T.dot(
            merged_population3.states[:, 2] + merged_population3.states[:, 6] +
            merged_population3.states[:, 10])/ave_hh_size
        postmerge_I_1 = merged_output.H_postmerge_1.T.dot(unmerged_population.states[:, 2])/ave_hh_size
        peaks_1[i,j] = max(hstack((merge_I_1, postmerge_I_1)))
        merge_R_1 = (1/3) * merged_output.H_merge_1.T.dot(
            merged_population3.states[:, 3] + merged_population3.states[:, 7] +
            merged_population3.states[:,11])/ave_hh_size
        postmerge_R_1 = merged_output.H_postmerge_1.T.dot(unmerged_population.states[:, 3])/ave_hh_size
        jan_antiprev_1[i,j] = postmerge_R_1[-1]

        merge_I_2 = (1/2) * merged_output.H_merge_2.T.dot(
            merged_population2.states[:, 2] + merged_population2.states[:, 6])/ave_hh_size
        postmerge_I_2 = merged_output.H_postmerge_2.T.dot(unmerged_population.states[:, 2])/ave_hh_size
        peaks_2[i,j] = max(hstack((merge_I_2, postmerge_I_2)))
        merge_R_2 = (1/2) * merged_output.H_merge_2.T.dot(
            merged_population2.states[:, 3] + merged_population2.states[:, 7])/ave_hh_size
        postmerge_R_2 = merged_output.H_postmerge_2.T.dot(unmerged_population.states[:, 3])/ave_hh_size
        jan_antiprev_2[i,j] = postmerge_R_2[-1]

        merge_I_3 = (1/2) * merged_output.H_merge_3.T.dot(
            merged_population2.states[:, 2] + merged_population2.states[:, 6])/ave_hh_size
        postmerge_I_3 = merged_output.H_postmerge_3.T.dot(unmerged_population.states[:, 2])/ave_hh_size
        peaks_3[i,j] = max(hstack((merge_I_3, postmerge_I_3)))
        merge_R_3 = (1/2) * merged_output.H_merge_3.T.dot(
            merged_population2.states[:, 3] + merged_population2.states[:, 7])/ave_hh_size
        postmerge_R_3 = merged_output.H_postmerge_3.T.dot(unmerged_population.states[:, 3])/ave_hh_size
        jan_antiprev_3[i,j] = postmerge_R_3[-1]

        merge_I_4 = (1/2) * merged_output.H_merge_4.T.dot(
            merged_population2.states[:, 2] + merged_population2.states[:, 6])/ave_hh_size
        postmerge_I_4 = merged_output.H_postmerge_4.T.dot(unmerged_population.states[:, 2])/ave_hh_size
        peaks_4[i,j] = max(hstack((merge_I_4, postmerge_I_4)))
        merge_R_4 = (1/2) * merged_output.H_merge_4.T.dot(
            merged_population2.states[:, 3] + merged_population2.states[:, 7])/ave_hh_size
        postmerge_R_4 = merged_output.H_postmerge_4.T.dot(unmerged_population.states[:, 3])/ave_hh_size
        jan_antiprev_4[i,j] = postmerge_R_4[-1]

        ax.plot(baseline_time, 100*baseline_R, label=lgd[0])
        ax.plot(hstack((merged_output.t_merge_1,merged_output.t_postmerge_1)),100*hstack((merge_R_1,postmerge_R_1)), label=lgd[1])
        ax.plot(hstack((merged_output.t_merge_2,merged_output.t_postmerge_2)),100*hstack((merge_R_2,postmerge_R_2)), label=lgd[2])
        ax.plot(hstack((merged_output.t_merge_3,merged_output.t_postmerge_3)),100*hstack((merge_R_3,postmerge_R_3)), label=lgd[3])
        ax.plot(hstack((merged_output.t_merge_4,merged_output.t_postmerge_4)),100*hstack((merge_R_4,postmerge_R_4)), label=lgd[4])

        ax.set_xlabel('Time in days')
        ax.set_ylabel('Percentage recovered')
        ax.set_xlim([340,395])
        # ax.set_ylim([5.5,9.0])

        ax.legend(ncol=1, bbox_to_anchor=(1,0.50))

        fig.savefig('R_by_strategy_sweep' + str(i) + str(j) + '.png', bbox_inches='tight', dpi=300)
        close()

        fig, ax = subplots(1, 1, sharex=True)

        ax.plot(baseline_time, 100*baseline_I, label=lgd[0])
        ax.plot(hstack((merged_output.t_merge_1,merged_output.t_postmerge_1)),100*hstack((merge_I_1,postmerge_I_1)), label=lgd[1])
        ax.plot(hstack((merged_output.t_merge_2,merged_output.t_postmerge_2)),100*hstack((merge_I_2,postmerge_I_2)), label=lgd[2])
        ax.plot(hstack((merged_output.t_merge_3,merged_output.t_postmerge_3)),100*hstack((merge_I_3,postmerge_I_3)), label=lgd[3])
        ax.plot(hstack((merged_output.t_merge_4,merged_output.t_postmerge_4)),100*hstack((merge_I_4,postmerge_I_4)), label=lgd[4])

        ax.set_xlabel('Time in days')
        ax.set_ylabel('Percentage infectious')
        ax.set_xlim([340,395])
        # ax.set_ylim([0,1])

        ax.legend(ncol=1, bbox_to_anchor=(1,0.50))

        fig.savefig('I_by_strategy_sweep' + str(i) + str(j) + '.png', bbox_inches='tight', dpi=300)
        close()

pol_label = []
for pol in range(4):
    pol_label.append('Policy'+str(pol+1))

peaks_min = array([peaks_1.min(),peaks_2.min(),peaks_3.min(),peaks_4.min()]).min()
peaks_max = array([peaks_1.max(),peaks_2.max(),peaks_3.max(),peaks_4.max()]).max()

fig, ax = subplots(1,1,sharex=True)
heatmap(peaks_1,square=True, vmin=peaks_min, vmax=peaks_max)
ax.set_title(pol_label[0])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('peaks_1.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
heatmap(peaks_2,square=True, vmin=peaks_min, vmax=peaks_max)
ax.set_title(pol_label[1])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('peaks_2.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
heatmap(peaks_3,square=True, vmin=peaks_min, vmax=peaks_max)
ax.set_title(pol_label[2])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('peaks_3.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
heatmap(peaks_4,square=True, vmin=peaks_min, vmax=peaks_max)
ax.set_title(pol_label[3])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('peaks_4.png',bbox_inches='tight', dpi=300)
close()

antiprev_min = array([jan_antiprev_1.min(),jan_antiprev_2.min(),jan_antiprev_3.min(),jan_antiprev_4.min()]).min()
antiprev_max = array([jan_antiprev_1.max(),jan_antiprev_2.max(),jan_antiprev_3.max(),jan_antiprev_4.max()]).max()

fig, ax = subplots(1,1,sharex=True)
heatmap(jan_antiprev_1,square=True, vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[0])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('jan_antiprev_1.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
heatmap(jan_antiprev_2,square=True, vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[1])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('jan_antiprev_2.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
heatmap(jan_antiprev_3,square=True, vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[2])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('jan_antiprev_3.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
heatmap(jan_antiprev_4,square=True, vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[3])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('jan_antiprev_4.png',bbox_inches='tight', dpi=300)
close()
