'''This plots the bubble results
'''
from pickle import load
from numpy import array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import close, colorbar, imshow, set_cmap, subplots
from examples.temp_bubbles.common import DataObject
from seaborn import heatmap

no_AR_vals = 3
no_home_vals = 10
no_ext_vals = 10

AR = zeros((no_AR_vals,))

peaks = zeros((no_AR_vals,no_home_vals,no_ext_vals))

R_end = zeros((no_AR_vals,no_home_vals,no_ext_vals))

for i in range(no_AR_vals):

    filename_stem_i = 'mix_sweep_results_' + str(i)

    for j in range(no_home_vals):

        filename_stem_j = filename_stem_i +str(j)

        for k in range(no_ext_vals):

            filename = filename_stem_j + str(k)
            with open(filename + '.pkl', 'rb') as f:
                AR_now, household_population, results = load(f)

            H = results.H

            ave_hh_size = household_population.composition_distribution.dot(sum(household_population.composition_list, axis=1))

            fig, ax = subplots(1, 1, sharex=True)

            I = (H.T.dot(household_population.states[:, 3::5])).sum(axis=1)/ave_hh_size
            R = (H.T.dot(household_population.states[:, 4::5])).sum(axis=1)/ave_hh_size

            peaks[i,j,k] = 100 * max(I)
            R_end[i,j,k] = 100 * R[-1]
            AR[i] = AR_now

peak_min = peaks.min()
peak_max = peaks.max()
R_end_min = R_end.min()
R_end_max = R_end.max()
for i in range(3):
    fig, ax = subplots(1,1,sharex=True)
    imshow(peaks[i,:,:],origin='lower',extent=(0,100,0,100),vmin=peak_min,vmax=peak_max)
    ax.set_title('Peak prevalence')
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="Peak % prevalence",fraction=0.046, pad=0.04)

    fig.savefig('mixing_sweep_peak_AR_'+str(AR[i])+'.png',bbox_inches='tight', dpi=300)
    close()

    fig, ax = subplots(1,1,sharex=True)
    imshow(R_end[i,:,:],origin='lower',extent=(0,100,0,100),vmin=R_end_min,vmax=R_end_max)
    ax.set_title('End-of-projection immunity')
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="% immune after 30 days",fraction=0.046, pad=0.04)

    fig.savefig('mixing_sweep_immunity_AR_'+str(AR[i])+'.png',bbox_inches='tight', dpi=300)
    close()
