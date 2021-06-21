'''This plots the bubble results
'''
from pickle import load
from numpy import arange, array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import close, colorbar, imshow, set_cmap, subplots
from examples.temp_bubbles.common import DataObject
from seaborn import heatmap

with open('mixing_sweep_output.pkl','rb') as f:
    (growth_rate, peaks,
    R_end,
    ar_range,
    internal_mix_range,
    external_mix_range) = load(f)

ar_len = len(ar_range)
internal_mix_len = len(internal_mix_range)
external_mix_len = len(external_mix_range)

r_min = growth_rate.min()
r_max = growth_rate.max()
peak_min = peaks.min()
peak_max = peaks.max()
R_end_min = R_end.min()
R_end_max = R_end.max()
for i in range(ar_len):
    fig, ax = subplots(1,1,sharex=True)
    imshow(growth_rate[i,:,:],origin='lower',extent=(0,100,0,100),vmin=r_min,vmax=r_max)
    ttl = 'Secondary attack probability ' + str(ar_range[i])
    ax.set_title(ttl)
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="Growth rate",fraction=0.046, pad=0.04)

    fig.savefig('mixing_sweep_growth_rate_AR_par'+str(ar_range[i])+'.png',bbox_inches='tight', dpi=300)
    close()

    fig, ax = subplots(1,1,sharex=True)
    imshow(peaks[i,:,:],origin='lower',extent=(0,100,0,100),vmin=peak_min,vmax=peak_max)
    ttl = 'Secondary attack probability ' + str(ar_range[i])
    ax.set_title(ttl)
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="Peak % prevalence",fraction=0.046, pad=0.04)

    fig.savefig('mixing_sweep_peak_AR_par'+str(ar_range[i])+'.png',bbox_inches='tight', dpi=300)
    close()

    fig, ax = subplots(1,1,sharex=True)
    imshow(R_end[i,:,:],origin='lower',extent=(0,100,0,100),vmin=R_end_min,vmax=R_end_max)
    ax.set_title(ttl)
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="% immune at end of projection",fraction=0.046, pad=0.04)

    fig.savefig('mixing_sweep_immunity_AR_par'+str(ar_range[i])+'.png',bbox_inches='tight', dpi=300)
    close()
