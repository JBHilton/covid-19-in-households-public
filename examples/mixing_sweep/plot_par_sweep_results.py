'''This plots the bubble results
'''
from pickle import load
from numpy import arange, array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import close, colorbar, imshow, set_cmap, subplots
from examples.temp_bubbles.common import DataObject
from seaborn import heatmap

AR_range = array([0.3,0.45,0.6])
internal_mix_range = array([0.2,0.6])
external_mix_range = array([0.2,0.6])
AR_len = len(AR_range)
internal_mix_len = len(internal_mix_range)
external_mix_len = len(external_mix_range)

with open('mixing_sweep_output.pkl','rb') as f:
    beta_ext, peaks, R_end, params = load(f)

beta_min = beta_ext.min()
beta_max = beta_ext.max()
peak_min = peaks.min()
peak_max = peaks.max()
R_end_min = R_end.min()
R_end_max = R_end.max()
for i in range(3):
    fig, ax = subplots(1,1,sharex=True)
    imshow(beta_ext[i,:,:],origin='lower',extent=(0,100,0,100),vmin=beta_min,vmax=beta_max)
    ttl = 'Secondary attack probability ' + str(AR_range[i])
    ax.set_title(ttl)
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="Growth rate",fraction=0.046, pad=0.04)

    fig.savefig('mixing_sweep_beta_ext_AR_par'+str(AR_range[i])+'.png',bbox_inches='tight', dpi=300)
    close()

    fig, ax = subplots(1,1,sharex=True)
    imshow(peaks[i,:,:],origin='lower',extent=(0,100,0,100),vmin=peak_min,vmax=peak_max)
    ttl = 'Secondary attack probability ' + str(AR_range[i])
    ax.set_title(ttl)
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="Peak % prevalence",fraction=0.046, pad=0.04)

    fig.savefig('mixing_sweep_peak_AR_par'+str(AR_range[i])+'.png',bbox_inches='tight', dpi=300)
    close()

    fig, ax = subplots(1,1,sharex=True)
    imshow(R_end[i,:,:],origin='lower',extent=(0,100,0,100),vmin=R_end_min,vmax=R_end_max)
    ax.set_title(ttl)
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="% immune at end of projection",fraction=0.046, pad=0.04)

    fig.savefig('mixing_sweep_immunity_AR_par'+str(AR_range[i])+'.png',bbox_inches='tight', dpi=300)
    close()
