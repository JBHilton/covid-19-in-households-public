'''This plots the bubble results
'''
from os import mkdir
from os.path import isdir
from pickle import load
from numpy import arange, array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import axes, close, colorbar, imshow, set_cmap, subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from examples.temp_bubbles.common import DataObject
from seaborn import heatmap

if isdir('plots/mixing_sweep') is False:
    mkdir('plots/mixing_sweep')

with open('outputs/mixing_sweep/results.pkl','rb') as f:
    (growth_rate,
    peaks,
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

fig, (ax1, ax2, ax3) = subplots(1, 3)
axim=ax1.imshow(growth_rate[0,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=r_min,
                vmax=r_max)
ax2.imshow(growth_rate[1,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=r_min,
                vmax=r_max)
ax3.imshow(growth_rate[2,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=r_min,
                vmax=r_max)
ax1.set_ylabel('% reduction in\n within-household transmission')
ax1.set_xlabel('% reduction in\n between-household\n transmission')
ax2.set_xlabel('% reduction in\n between-household\n transmission')
ax3.set_xlabel('% reduction in\n between-household\n transmission')
ax2.get_yaxis().set_ticks([])
ax3.get_yaxis().set_ticks([])
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim, label="Growth rate", cax=cax)
ax3.set_ylim(ax2.get_ylim())

fig.savefig('plots/mixing_sweep/growth_rate.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, (ax1, ax2, ax3) = subplots(1, 3)
axim=ax1.imshow(peaks[0,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=peak_min,
                vmax=peak_max)
ax2.imshow(peaks[1,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=peak_min,
                vmax=peak_max)
ax3.imshow(peaks[2,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=peak_min,
                vmax=peak_max)
ax1.set_ylabel('% reduction in\n within-household transmission')
ax1.set_xlabel('% reduction in\n between-household\n transmission')
ax2.set_xlabel('% reduction in\n between-household\n transmission')
ax3.set_xlabel('% reduction in\n between-household\n transmission')
ax2.get_yaxis().set_ticks([])
ax3.get_yaxis().set_ticks([])
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim, label="Peak % prevalence", cax=cax)
ax3.set_ylim(ax2.get_ylim())

fig.savefig('plots/mixing_sweep/peaks.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, (ax1, ax2, ax3) = subplots(1, 3)
axim=ax1.imshow(R_end[0,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=R_end_min,
                vmax=R_end_max)
ax2.imshow(R_end[1,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=R_end_min,
                vmax=R_end_max)
ax3.imshow(R_end[2,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=R_end_min,
                vmax=R_end_max)
ax1.set_ylabel('% reduction in\n within-household transmission')
ax1.set_xlabel('% reduction in\n between-household\n transmission')
ax2.set_xlabel('% reduction in\n between-household\n transmission')
ax3.set_xlabel('% reduction in\n between-household\n transmission')
ax2.get_yaxis().set_ticks([])
ax3.get_yaxis().set_ticks([])
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim, label="% population immunity at end of simulation", cax=cax)
ax3.set_ylim(ax2.get_ylim())

fig.savefig('plots/mixing_sweep/R_end.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, (ax1, ax2, ax3) = subplots(1, 3)
axim=ax1.imshow(hh_prop[0,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=hh_prop_min,
                vmax=hh_prop_max)
ax2.imshow(hh_prop[1,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=hh_prop_min,
                vmax=hh_prop_max)
ax3.imshow(hh_prop[2,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=hh_prop_min,
                vmax=hh_prop_max)
ax1.set_ylabel('% reduction in\n within-household transmission')
ax1.set_xlabel('% reduction in\n between-household\n transmission')
ax2.set_xlabel('% reduction in\n between-household\n transmission')
ax3.set_xlabel('% reduction in\n between-household\n transmission')
ax2.get_yaxis().set_ticks([])
ax3.get_yaxis().set_ticks([])
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim, label="% of households infected", cax=cax)
ax3.set_ylim(ax2.get_ylim())

fig.savefig('plots/mixing_sweep/hh_prop.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, (ax1, ax2, ax3) = subplots(1, 3)
axim=ax1.imshow(attack_ratio[0,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=attack_ratio_min,
                vmax=attack_ratio_max)
ax2.imshow(attack_ratio[1,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=attack_ratio_min,
                vmax=attack_ratio_max)
ax3.imshow(attack_ratio[2,:,:],
                origin='lower',
                extent=(0,1,0,1),
                vmin=attack_ratio_min,
                vmax=attack_ratio_max)
ax1.set_ylabel('% reduction in\n within-household transmission')
ax1.set_xlabel('% reduction in\n between-household\n transmission')
ax2.set_xlabel('% reduction in\n between-household\n transmission')
ax3.set_xlabel('% reduction in\n between-household\n transmission')
ax2.get_yaxis().set_ticks([])
ax3.get_yaxis().set_ticks([])
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim, label="% attack ratio", cax=cax)
ax3.set_ylim(ax2.get_ylim())

fig.savefig('plots/mixing_sweep/attack_ratio.png',
            bbox_inches='tight',
            dpi=300)
close()
