'''This plots the results of the parameter sweep for the long-term bubbling
example.
'''
from os import mkdir
from os.path import isdir
from pickle import load
from numpy import arange, array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import close, colorbar, imshow, set_cmap, subplots
from seaborn import heatmap

if isdir('plots/long_term_bubbles') is False:
    mkdir('plots/long_term_bubbles')

with open('outputs/long_term_bubbles/results.pkl','rb') as f:
    (growth_rate,
     peaks,
     R_end,
     hh_prop,
     attack_ratio,
     bubble_prob_range,
     external_mix_range) = load(f)

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

fig, ax = subplots(1,1,sharex=True)
imshow(growth_rate,origin='lower',extent=(0,100,0,100),vmin=r_min,vmax=r_max)
ax.set_xlabel('% reduction in between-household transmission')
ax.set_ylabel('% uptake of support bubbles')
set_cmap('bwr')
cbar = colorbar(label="Growth rate",fraction=0.046, pad=0.04)
fig.savefig('plots/long_term_bubbles/growth_rate.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(peaks,origin='lower',extent=(0,100,0,100),vmin=peak_min,vmax=peak_max)
ax.set_xlabel('% reduction in between-household transmission')
ax.set_ylabel('% uptake of support bubbles')
set_cmap('bwr')
cbar = colorbar(label="Peak % prevalence",fraction=0.046, pad=0.04)

fig.savefig('plots/long_term_bubbles/peak.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(R_end,origin='lower',extent=(0,100,0,100),vmin=R_end_min,vmax=R_end_max)
ax.set_xlabel('% reduction in between-household transmission')
ax.set_ylabel('% uptake of support bubbles')
set_cmap('bwr')
cbar = colorbar(label="% immune at end of projection",fraction=0.046, pad=0.04)

fig.savefig('plots/long_term_bubbles/immunity.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(hh_prop,origin='lower',extent=(0,100,0,100),vmin=hh_prop_min,vmax=hh_prop_max)
ax.set_ylabel('% reduction in within-household transmission')
ax.set_xlabel('% reduction in between-household transmission')
set_cmap('bwr')
cbar = colorbar(label="% of households infected during projection",fraction=0.046, pad=0.04)

fig.savefig('plots/mixing_sweep/hh_prop.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(attack_ratio,origin='lower',extent=(0,100,0,100),vmin=attack_ratio_min,vmax=attack_ratio_max)
ax.set_ylabel('% reduction in within-household transmission')
ax.set_xlabel('% reduction in between-household transmission')
set_cmap('bwr')
cbar = colorbar(label="% attack rate in infected households",fraction=0.046, pad=0.04)

fig.savefig('plots/mixing_sweep/attack_ratio.png',bbox_inches='tight', dpi=300)
close()
