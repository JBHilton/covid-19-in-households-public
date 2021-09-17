'''This plots the results of the parameter sweep for the OOHI
example.
'''
from os import mkdir
from os.path import isdir
from pickle import load
from numpy import arange, array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import close, colorbar, imshow, set_cmap, subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from seaborn import heatmap

if isdir('plots/oohi') is False:
    mkdir('plots/oohi')

with open('outputs/oohi/results.pkl','rb') as f:
    (vuln_peaks,
     vuln_end,
     iso_peaks,
     cum_iso,
     iso_method_range,
     iso_rate_range,
     iso_prob_range) = load(f)

vp_min = vuln_peaks.min()
vp_max = vuln_peaks.max()
ve_min = vuln_end.min()
ve_max = vuln_end.max()
ip_min = iso_peaks.min()
ip_max = iso_peaks.max()
ci_min = cum_iso.min()
ci_max = cum_iso.max()

fig, (ax1, ax2) = subplots(1,2,sharex=True)
axim=ax1.imshow(vuln_peaks[0,:,:],
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[1],0,1),
           vmin=vp_min,
           vmax=vp_max)
ax1.set_xlabel('Detection rate')
ax1.set_ylabel('Adherence probability')
ax2.imshow(vuln_peaks[1,:,:],
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[1],0,1),
           vmin=vp_min,
           vmax=vp_max)
ax2.set_xlabel('Detection rate')

ax2.get_yaxis().set_ticks([])
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Peak % prevalence in vulnerable population",
                cax=cax)
fig.savefig('plots/oohi/vuln_peaks.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, (ax1, ax2) = subplots(1,2,sharex=True)
axim=ax1.imshow(vuln_end[0,:,:],
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[1],0,1),
           vmin=ve_min,
           vmax=ve_max)
ax1.set_xlabel('Detection rate')
ax1.set_ylabel('Adherence probability')
ax2.imshow(vuln_end[1,:,:],
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[1],0,1),
           vmin=ve_min,
           vmax=ve_max)
ax2.set_xlabel('Detection rate')

ax2.get_yaxis().set_ticks([])
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Cumulative % infected in vulnerable population",
                cax=cax)

fig.savefig('plots/oohi/cum_vuln_cases.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, (ax1, ax2) = subplots(1,2,sharex=True)
axim=ax1.imshow(iso_peaks[0,:,:],
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[1],0,1),
           vmin=ip_min,
           vmax=ip_max)
ax1.set_xlabel('Detection rate')
ax1.set_ylabel('Adherence probability')
ax2.imshow(iso_peaks[1,:,:],
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[1],0,1),
           vmin=ip_min,
           vmax=ip_max)
ax2.set_xlabel('Detection rate')

ax2.get_yaxis().set_ticks([])
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Peak % population isolating",
                cax=cax)

fig.savefig('plots/oohi/iso_peak.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, (ax1, ax2) = subplots(1,2,sharex=True)
axim=ax1.imshow(cum_iso[0,:,:],
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[1],0,1),
           vmin=ci_min,
           vmax=ci_max)
ax1.set_xlabel('Detection rate')
ax1.set_ylabel('Adherence probability')
ax2.imshow(cum_iso[1,:,:],
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[1],0,1),
           vmin=ci_min,
           vmax=ci_max)
ax2.set_xlabel('Detection rate')

ax2.get_yaxis().set_ticks([])
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Cumulative % isolating",
                cax=cax)

fig.savefig('plots/oohi/cum_iso.png',
            bbox_inches='tight',
            dpi=300)
close()
