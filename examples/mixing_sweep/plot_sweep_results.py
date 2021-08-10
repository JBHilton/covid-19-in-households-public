'''This plots the bubble results
'''
from os import mkdir
from os.path import isdir
from pickle import load
from numpy import arange, array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import close, colorbar, imshow, set_cmap, subplots
from examples.temp_bubbles.common import DataObject
from seaborn import heatmap

if isdir('plots/mixing_sweep') is False:
    mkdir('plots/mixing_sweep')

with open('outputs/mixing_sweep/results.pkl','rb') as f:
    (growth_rate, peaks,
    R_end,
    hh_prop,
    attack_ratio,
    sip_range,
    internal_mix_range,
    external_mix_range) = load(f)

sip_len = len(sip_range)
internal_mix_len = len(internal_mix_range)
external_mix_len = len(external_mix_range)

r_min = growth_rate.min()
r_max = growth_rate.max()
peak_min = peaks.min()
peak_max = peaks.max()
R_end_min = R_end.min()
R_end_max = R_end.max()
hh_prop_min = hh_prop.min()
hh_prop_max = hh_prop.max()
attack_ratio_min = attack_ratio.min()
attack_ratio_max = attack_ratio.max()
for i in range(sip_len):)
    ttl = 'Secondary infection probability ' + str(sip_range[i])
    fig, ax = subplots(1,1,sharex=True)
    imshow(growth_rate[i,:,:],origin='lower',extent=(0,100,0,100),vmin=r_min,vmax=r_max
    ax.set_title(ttl)
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="Growth rate",fraction=0.046, pad=0.04)

    fig.savefig('plots/mixing_sweep/growth_rate_SIP_'+str(sip_range[i])+'.png',bbox_inches='tight', dpi=300)
    close()

    fig, ax = subplots(1,1,sharex=True)
    imshow(peaks[i,:,:],origin='lower',extent=(0,100,0,100),vmin=peak_min,vmax=peak_max)
    ax.set_title(ttl)
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="Peak % prevalence",fraction=0.046, pad=0.04)

    fig.savefig('plots/mixing_sweep/peak_SIP_'+str(sip_range[i])+'.png',bbox_inches='tight', dpi=300)
    close()

    fig, ax = subplots(1,1,sharex=True)
    imshow(R_end[i,:,:],origin='lower',extent=(0,100,0,100),vmin=R_end_min,vmax=R_end_max)
    ax.set_title(ttl)
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="% immune at end of projection",fraction=0.046, pad=0.04)

    fig.savefig('plots/mixing_sweep/immunity_SIP_'+str(sip_range[i])+'.png',bbox_inches='tight', dpi=300)
    close()

    fig, ax = subplots(1,1,sharex=True)
    imshow(hh_prop[i,:,:],origin='lower',extent=(0,100,0,100),vmin=hh_prop_min,vmax=hh_prop_max)
    ax.set_title(ttl)
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="% of households infected during projection",fraction=0.046, pad=0.04)

    fig.savefig('plots/mixing_sweep/hh_prop_SIP_'+str(sip_range[i])+'.png',bbox_inches='tight', dpi=300)
    close()

    fig, ax = subplots(1,1,sharex=True)
    imshow(attack_ratio[i,:,:],origin='lower',extent=(0,100,0,100),vmin=attack_ratio_min,vmax=attack_ratio_max)
    ax.set_title(ttl)
    ax.set_ylabel('% reduction in within-household transmission')
    ax.set_xlabel('% reduction in between-household transmission')
    set_cmap('bwr')
    cbar = colorbar(label="% attack rate in infected households",fraction=0.046, pad=0.04)

    fig.savefig('plots/mixing_sweep/attack_ratio_SIP_'+str(sip_range[i])+'.png',bbox_inches='tight', dpi=300)
    close()
