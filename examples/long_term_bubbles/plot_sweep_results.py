'''This plots the results of the parameter sweep for the long-term bubbling
example.
'''
from os import mkdir
from os.path import isdir
from pickle import load
from numpy import arange, array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import axes, close, colorbar, imshow, set_cmap, subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
imshow(growth_rate,origin='lower',extent=(0,1,0,1),vmin=r_min,vmax=r_max)
ax.set_xlabel('% reduction in between-household transmission')
ax.set_ylabel('% uptake of support bubbles')

cbar = colorbar(label="Growth rate",fraction=0.046, pad=0.04)
fig.savefig('plots/long_term_bubbles/growth_rate.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(peaks,origin='lower',extent=(0,1,0,1),vmin=peak_min,vmax=peak_max)
ax.set_xlabel('% reduction in between-household transmission')
ax.set_ylabel('% uptake of support bubbles')

cbar = colorbar(label="Peak % prevalence",fraction=0.046, pad=0.04)

fig.savefig('plots/long_term_bubbles/peak.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(R_end,origin='lower',extent=(0,1,0,1),vmin=R_end_min,vmax=R_end_max)
ax.set_xlabel('% reduction in between-household transmission')
ax.set_ylabel('% uptake of support bubbles')

cbar = colorbar(label="% immune at end of projection",fraction=0.046, pad=0.04)

fig.savefig('plots/long_term_bubbles/immunity.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(hh_prop,origin='lower',extent=(0,1,0,1),vmin=hh_prop_min,vmax=hh_prop_max)
ax.set_xlabel('% reduction in between-household transmission')
ax.set_ylabel('% uptake of support bubbles')

cbar = colorbar(label="% of households infected during projection",fraction=0.046, pad=0.04)

fig.savefig('plots/long_term_bubbles/hh_prop.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(attack_ratio,origin='lower',extent=(0,1,0,1),vmin=attack_ratio_min,vmax=attack_ratio_max)
ax.set_xlabel('% reduction in between-household transmission')
ax.set_ylabel('% uptake of support bubbles')

cbar = colorbar(label="% attack rate in infected households",fraction=0.046, pad=0.04)

fig.savefig('plots/long_term_bubbles/attack_ratio.png',bbox_inches='tight', dpi=300)
close()

fig, ((ax1, ax2), (ax3, ax4)) = subplots(2, 2)
axim=ax1.imshow(peaks,
                origin='lower',
                extent=(0,1,0,1),
                vmin=peak_min,
                vmax=peak_max)
ax1.set_ylabel('% reduction in\n within-household\n transmission')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim, label="Peak % prevalence", cax=cax)

axim=ax2.imshow(R_end,
                origin='lower',
                extent=(0,1,0,1),
                vmin=R_end_min,
                vmax=R_end_max)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim, label="Cumulative % prevalence", cax=cax)

axim=ax3.imshow(hh_prop,
                origin='lower',
                extent=(0,1,0,1),
                vmin=hh_prop_min,
                vmax=hh_prop_max)
ax3.set_ylabel('% uptake of support bubbles')
ax3.set_xlabel('% reduction in\n between-household\n transmission')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim, label="% of households infected", cax=cax)

axim=ax4.imshow(attack_ratio,
                origin='lower',
                extent=(0,1,0,1),
                vmin=attack_ratio_min,
                vmax=attack_ratio_max)
ax4.set_xlabel('% reduction in\n between-household\n transmission')
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim, label="% attack ratio", cax=cax)

fig.savefig('plots/long_term_bubbles/grid_plot.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, ((ax1, ax2)) = subplots(1, 2)
axim=ax1.imshow(peaks,
                origin='lower',
                extent=(0,1,0,1),
                vmin=peak_min,
                vmax=peak_max)
ax1.set_xlabel('% reduction in\n between-household\n transmission')
ax1.set_ylabel('% reduction in\n within-household\n transmission')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("top", size="5%", pad=0.05)
cbar = colorbar(axim, label="Peak % prevalence", orientation='horizontal', cax=cax)
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

axim=ax2.imshow(R_end,
                origin='lower',
                extent=(0,1,0,1),
                vmin=R_end_min,
                vmax=R_end_max)
ax2.set_xlabel('% reduction in\n between-household\n transmission')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("top", size="5%", pad=0.05)
cbar = colorbar(axim, label="Cumulative % prevalence", orientation='horizontal', cax=cax)
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

fig.savefig('plots/long_term_bubbles/poster_plot.png',
            bbox_inches='tight',
            dpi=300)
close()
