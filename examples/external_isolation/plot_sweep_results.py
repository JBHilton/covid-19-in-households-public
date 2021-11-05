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

fig, ax = subplots(1,1,sharex=True)
axim=ax.imshow(vuln_peaks,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=vp_min,
           vmax=vp_max)
ax.set_xlabel('Detection rate')
ax.set_ylabel('Isolation probability')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Peak % prevalence in vulnerable population",
                cax=cax)
fig.savefig('plots/oohi/vuln_peaks.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
axim=ax.imshow(vuln_end,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=ve_min,
           vmax=ve_max)
ax.set_xlabel('Detection rate')
ax.set_ylabel('Isolation probability')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Cumulative % infected in vulnerable population",
                cax=cax)

fig.savefig('plots/oohi/cum_vuln_cases.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
axim=ax.imshow(iso_peaks,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=ip_min,
           vmax=ip_max)
ax.set_xlabel('Detection rate')
ax.set_ylabel('Isolation probability')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Peak % population isolating",
                cax=cax)

fig.savefig('plots/oohi/iso_peak.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
axim=ax.imshow(cum_iso,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=ci_min,
           vmax=ci_max)
ax.set_xlabel('Detection rate')
ax.set_ylabel('Isolation probability')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Cumulative % isolating",
                cax=cax)

fig.savefig('plots/oohi/cum_iso.png',
            bbox_inches='tight',
            dpi=300)
close()

peak_max = max(vp_max, ip_max)
end_max = max(ve_max, ci_max)

fig, ((ax1, ax3), (ax2, ax4)) = subplots(2,2,sharex=True)

axim=ax1.imshow(vuln_peaks,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=0,
           vmax=peak_max)
ax1.set_xlabel('Detection rate')
ax1.set_ylabel('Isolation probability')

# divider = make_axes_locatable(ax1)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = colorbar(axim,
#                 label="Peak % prevalence in vulnerable population",
#                 cax=cax)

axim=ax2.imshow(vuln_end,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=0,
           vmax=end_max)
ax2.set_xlabel('Detection rate')
ax2.set_ylabel('Isolation probability')

# divider = make_axes_locatable(ax2)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = colorbar(axim,
#                 label="Cumulative % infected in vulnerable population",
#                 cax=cax)

axim=ax3.imshow(iso_peaks,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=0,
           vmax=peak_max)
ax3.set_xlabel('Detection rate')
ax3.set_ylabel('Isolation probability')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = colorbar(axim,
                label="Peak %",
                cax=cax)

axim=ax4.imshow(cum_iso,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=0,
           vmax=end_max)
ax4.set_xlabel('Detection rate')
ax4.set_ylabel('Isolation probability')

divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = colorbar(axim,
                label="Cumulative %",
                cax=cax)

fig.savefig('plots/oohi/grid_plot.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, ((ax1, ax2)) = subplots(2,1,sharex=True)

axim=ax1.imshow(vuln_end,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=0,
           vmax=ve_max)
ax1.set_xlabel('Detection rate')
ax1.set_ylabel('Isolation probability')

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = colorbar(axim,
                label="Cumulative clinically\n vulnerable % prevalence",
                cax=cax)

axim=ax2.imshow(cum_iso,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=0,
           vmax=ci_max)
ax2.set_xlabel('Detection rate (1/days)')
ax2.set_ylabel('Isolation probability')

divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = colorbar(axim,
                label="Total % of population\n who isolate",
                cax=cax)

fig.savefig('plots/oohi/poster_plot.png',
            bbox_inches='tight',
            dpi=300)
close()
