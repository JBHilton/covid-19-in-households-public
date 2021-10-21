'''This plots the mixing sweep results
'''
from os import mkdir
from os.path import isdir
from pickle import load
from numpy import arange, array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import axes, close, colorbar, imshow, set_cmap, subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from seaborn import heatmap

if isdir('plots/mixing_sweep') is False:
    mkdir('plots/mixing_sweep')

with open('outputs/mixing_sweep/results.pkl','rb') as f:
    (growth_rate,
    peaks,
    R_end,
    hh_prop,
    attack_ratio,
    internal_mix_range,
    external_mix_range) = load(f)

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

fig, ax = subplots(1, 1, sharex=True)
axim=ax.imshow(growth_rate,
                origin='lower',
                extent=(0,1,0,1),
                vmin=r_min,
                vmax=r_max)
ax.set_ylabel('% reduction in\n within-household\n transmission')
ax.set_xlabel('% reduction in\n between-household\n transmission')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(axim, label="Growth rate", cax=cax)

fig.savefig('plots/mixing_sweep/growth_rate.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, ax = subplots(1, 1)
axim=ax.imshow(peaks,
                origin='lower',
                extent=(0,1,0,1),
                vmin=peak_min,
                vmax=peak_max)
ax.set_ylabel('% reduction in\n within-household\n transmission')
ax.set_xlabel('% reduction in\n between-household\n transmission')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(axim, label="Peak % prevalence", cax=cax)

fig.savefig('plots/mixing_sweep/peaks.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, ax = subplots(1, 1)
axim=ax.imshow(R_end,
                origin='lower',
                extent=(0,1,0,1),
                vmin=R_end_min,
                vmax=R_end_max)
ax.set_ylabel('% reduction in\n within-household\n transmission')
ax.set_xlabel('% reduction in\n between-household\n transmission')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(axim, label="% population immunity at end of simulation", cax=cax)

fig.savefig('plots/mixing_sweep/R_end.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, ax = subplots(1, 1)
axim=ax.imshow(hh_prop,
                origin='lower',
                extent=(0,1,0,1),
                vmin=hh_prop_min,
                vmax=hh_prop_max)
ax.set_ylabel('% reduction in\n within-household\n transmission')
ax.set_xlabel('% reduction in\n between-household\n transmission')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(axim, label="% of households infected", cax=cax)

fig.savefig('plots/mixing_sweep/hh_prop.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, ax = subplots(1, 1)
axim=ax.imshow(attack_ratio,
                origin='lower',
                extent=(0,1,0,1),
                vmin=attack_ratio_min,
                vmax=attack_ratio_max)
ax.set_ylabel('% reduction in\n within-household\n transmission')
ax.set_xlabel('% reduction in\n between-household\n transmission')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(axim, label="% attack ratio", cax=cax)

fig.savefig('plots/mixing_sweep/attack_ratio.png',
            bbox_inches='tight',
            dpi=300)
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
ax3.set_ylabel('% reduction in\n within-household\n transmission')
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

fig.savefig('plots/mixing_sweep/grid_plot.png',
            bbox_inches='tight',
            dpi=300)
close()
