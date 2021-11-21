'''This plots the bubble results
'''
from os import mkdir
from os.path import isdir
from pickle import dump, load
from numpy import array, atleast_2d, hstack, where, zeros
from matplotlib.pyplot import axes, close, colorbar, imshow, subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable

if isdir('plots/temp_bubbles') is False:
    mkdir('plots/temp_bubbles')

with open('outputs/temp_bubbles/results.pkl', 'rb') as f:
    (peak_data0,
     end_data0,
     ar_data0,
     hh_prop_data0,
     peak_data1,
     end_data1,
     ar_data1,
     hh_prop_data1,
     peak_data2,
     end_data2,
     ar_data2,
     hh_prop_data2,
     peak_data3,
     end_data3,
     ar_data3,
     hh_prop_data3,
     peak_data4,
     end_data4,
     ar_data4,
     hh_prop_data4,
     unmerged_exponents,
     merged_exponents) = load(f)

no_i_vals = 3
no_j_vals = 3

pol_label = []
for pol in range(5):
    pol_label.append('Policy'+str(pol))

peak0_min = peak_data0.min()
peak0_max = peak_data0.max()
end0_min = end_data0.min()
end0_max =end_data0.max()

peak_datamin = 100 * array([peak_data0.min(),
                            peak_data1.min(),
                            peak_data2.min(),
                            peak_data3.min(),
                            peak_data4.min()]).min()
peak_datamax = 100 * array([peak_data0.max(),
                            peak_data1.max(),
                            peak_data2.max(),
                            peak_data3.max(),
                            peak_data4.max()]).max()

fig, ((ax1, ax2), (ax3, ax4)) = subplots(2,2)
pd_1=ax1.imshow(100 * peak_data1, origin='lower',extent=(0, 1, 0, 1), vmin=peak_datamin, vmax=peak_datamax)
ax1.set_ylabel('Single household\n density exponent')

ax2.imshow(100 * peak_data2, origin='lower',extent=(0, 1, 0, 1), vmin=peak_datamin, vmax=peak_datamax)

ax3.imshow(100 * peak_data3, origin='lower',extent=(0, 1, 0, 1), vmin=peak_datamin, vmax=peak_datamax)
ax3.set_ylabel('Single household\n density exponent')
ax3.set_xlabel('Bubbled density exponent')

ax4.imshow(100 * peak_data4, origin='lower',extent=(0, 1, 0, 1), vmin=peak_datamin, vmax=peak_datamax)
ax4.set_xlabel('Bubbled density exponent')

cax = axes([0.9, 0.1, 0.05, 0.75])
fig.colorbar(pd_1, label="Peak % prevalence", cax=cax)

fig.savefig('plots/temp_bubbles/peak_data_1to4.png',bbox_inches='tight', dpi=300)
close()

end_datamin = 100 * array([end_data0.min(),
                            end_data1.min(),
                            end_data2.min(),
                            end_data3.min(),
                            end_data4.min()]).min()
end_datamax = 100 * array([end_data0.max(),
                            end_data1.max(),
                            end_data2.max(),
                            end_data3.max(),
                            end_data4.max()]).max()

fig, ((ax1, ax2), (ax3, ax4)) = subplots(2,2)
ed_1=ax1.imshow(100 * end_data1, origin='lower',extent=(0, 1, 0, 1), vmin=end_datamin, vmax=end_datamax)
ax1.set_ylabel('Single household\n density exponent')

ax2.imshow(100 * end_data2, origin='lower',extent=(0, 1, 0, 1), vmin=end_datamin, vmax=end_datamax)

ax3.imshow(100 * end_data3, origin='lower',extent=(0, 1, 0, 1), vmin=end_datamin, vmax=end_datamax)
ax3.set_ylabel('Single household\n density exponent')
ax3.set_xlabel('Bubbled density exponent')

ax4.imshow(100 * end_data4, origin='lower',extent=(0, 1, 0, 1), vmin=end_datamin, vmax=end_datamax)
ax4.set_xlabel('Bubbled density exponent')

cax = axes([0.9, 0.1, 0.05, 0.75])
fig.colorbar(ed_1, label="% population immunity", cax=cax)

fig.savefig('plots/temp_bubbles/end_data_1to4.png',bbox_inches='tight', dpi=300)
close()

ar_datamin = 100 * array([ar_data0.min(),
                            ar_data1.min(),
                            ar_data2.min(),
                            ar_data3.min(),
                            ar_data4.min()]).min()
ar_datamax = 100 * array([ar_data0.max(),
                            ar_data1.max(),
                            ar_data2.max(),
                            ar_data3.max(),
                            ar_data4.max()]).max()

fig, ((ax1, ax2), (ax3, ax4)) = subplots(2,2)
ard_1=ax1.imshow(100 * ar_data1, origin='lower',extent=(0, 1, 0, 1), vmin=ar_datamin, vmax=ar_datamax)
ax1.set_ylabel('Single household\n density exponent')

ax2.imshow(100 * ar_data2, origin='lower',extent=(0, 1, 0, 1), vmin=ar_datamin, vmax=ar_datamax)

ax3.imshow(100 * ar_data3, origin='lower',extent=(0, 1, 0, 1), vmin=ar_datamin, vmax=ar_datamax)
ax3.set_ylabel('Single household\n density exponent')
ax3.set_xlabel('Bubbled density exponent')

ax4.imshow(100 * ar_data4, origin='lower',extent=(0, 1, 0, 1), vmin=ar_datamin, vmax=ar_datamax)
ax4.set_xlabel('Bubbled density exponent')

cax = axes([0.9, 0.1, 0.05, 0.75])
fig.colorbar(ard_1, label="% attack ratio", cax=cax)

fig.savefig('plots/temp_bubbles/ar_data_1to4.png',bbox_inches='tight', dpi=300)
close()

hh_prop_datamin = 100 * array([hh_prop_data0.min(),
                            hh_prop_data1.min(),
                            hh_prop_data2.min(),
                            hh_prop_data3.min(),
                            hh_prop_data4.min()]).min()
hh_prop_datamax = 100 * array([hh_prop_data0.max(),
                            hh_prop_data1.max(),
                            hh_prop_data2.max(),
                            hh_prop_data3.max(),
                            hh_prop_data4.max()]).max()

fig, ax = subplots(1,1,sharex=True)
imshow(100 * hh_prop_data0, origin='lower',extent=(0, 1, 0, 1), vmin=hh_prop_datamin, vmax=hh_prop_datamax)

ax.set_ylabel('Single household\n density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.tight_layout()

fig.savefig('plots/temp_bubbles/hh_prop_data0.png',bbox_inches='tight', dpi=300)
close()

fig, ((ax1, ax2), (ax3, ax4)) = subplots(2,2)
hhd_1=ax1.imshow(100 * hh_prop_data1, origin='lower',extent=(0, 1, 0, 1), vmin=hh_prop_datamin, vmax=hh_prop_datamax)
ax1.set_ylabel('Single household\n density exponent')

ax2.imshow(100 * hh_prop_data2, origin='lower',extent=(0, 1, 0, 1), vmin=hh_prop_datamin, vmax=hh_prop_datamax)

ax3.imshow(100 * hh_prop_data3, origin='lower',extent=(0, 1, 0, 1), vmin=hh_prop_datamin, vmax=hh_prop_datamax)
ax3.set_ylabel('Single household\n density exponent')
ax3.set_xlabel('Bubbled density exponent')

ax4.imshow(100 * hh_prop_data4, origin='lower',extent=(0, 1, 0, 1), vmin=hh_prop_datamin, vmax=hh_prop_datamax)
ax4.set_xlabel('Bubbled density exponent')

cax = axes([0.9, 0.1, 0.05, 0.75])
fig.colorbar(hhd_1, label="% of households infected", cax=cax)

fig.savefig('plots/temp_bubbles/hh_prop_data_1to4.png',bbox_inches='tight', dpi=300)
close()

''' Now do single day strategy '''

fig, ((ax1, ax2)) = subplots(1,2,sharex=True)
fig.tight_layout()
axim = ax1.imshow(100 * peak_data0,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=100*peak0_min,
                  vmax=100*peak0_max)
ax1.set_ylabel('Single household\n density exponent')
ax1.set_xlabel('Bubbled density\n exponent')
ax1.text(-0.3, 1.3, 'a)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.25)
cbar = colorbar(axim,
                cax=cax,
                orientation='horizontal',
                label="Peak % prevalence")
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

axim = ax2.imshow(100 * end_data0,
                  origin='lower',
                  extent=(0, 1, 0, 1),
                  vmin=100*end0_min,
                  vmax=100*end0_max)
ax2.set_xlabel('Bubbled density\n exponent')
ax2.text(-0.15, 1.3, 'b)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax2)
cax = divider.append_axes("top", size="5%", pad=0.25)
cbar = colorbar(axim,
                cax=cax,
                orientation='horizontal',
                label="Cumulative % prevalence")
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

fig.savefig('plots/temp_bubbles/single_day_grid.png',bbox_inches='tight', dpi=300)
close()

fig, ((ax_0, ax_1),
      (ax_2, ax_3),
      (ax_4, ax_null)) = subplots(3, 2, sharex=True, sharey=True)

fig.tight_layout()

axim = ax_0.imshow(100 * peak_data0,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=100*peak_data0.min(),
                  vmax=100*peak_data0.max())
ax_0.set_ylabel('Single household\n density exponent')
ax_0.text(-1, 1, 'a)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_0)
cax = divider.append_axes("right", size="5%", pad=0.25)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Peak %\n prevalence")

axim = ax_1.imshow(100 * peak_data1,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=100*peak_data1.min(),
                  vmax=100*peak_data1.max())
ax_1.set_ylabel('Single household\n density exponent')
ax_1.text(-0.6, 1, 'b)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_1)
cax = divider.append_axes("right", size="5%", pad=0.25)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Peak %\n prevalence")

axim = ax_2.imshow(100 * peak_data2,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=100*peak_data2.min(),
                  vmax=100*peak_data2.max())
ax_2.set_ylabel('Single household\n density exponent')
ax_2.text(-1, 1, 'c)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_2)
cax = divider.append_axes("right", size="5%", pad=0.25)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Peak %\n prevalence")

axim = ax_3.imshow(100 * peak_data3,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=100*peak_data3.min(),
                  vmax=100*peak_data3.max())
ax_3.set_ylabel('Single household\n density exponent')
ax_3.set_xlabel('Bubbled density\n exponent')
ax_3.text(-0.6, 1, 'd)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_3)
cax = divider.append_axes("right", size="5%", pad=0.25)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Peak %\n prevalence")

axim = ax_4.imshow(100 * peak_data4,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=100*peak_data4.min(),
                  vmax=100*peak_data4.max())
ax_4.set_ylabel('Single household\n density exponent')
ax_4.set_xlabel('Bubbled density\n exponent')
ax_4.text(-1, 1, 'e)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_4)
cax = divider.append_axes("right", size="5%", pad=0.25)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Peak %\n prevalence")

ax_null.axis('off')

fig.savefig('plots/temp_bubbles/peak_prev_grid.png',bbox_inches='tight', dpi=300)
close()


fig, ((ax_0, ax_1),
      (ax_2, ax_3),
      (ax_4, ax_null)) = subplots(3, 2, sharex=True, sharey=True)

fig.tight_layout()

axim = ax_0.imshow(100 * end_data0,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=100*end_data0.min(),
                  vmax=100*end_data0.max())
ax_0.set_ylabel('Single household\n density exponent')
ax_0.text(-1, 1, 'a)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_0)
cax = divider.append_axes("right", size="5%", pad=0.25)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Cumulative %\n prevalence")

axim = ax_1.imshow(100 * end_data1,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=100*end_data1.min(),
                  vmax=100*end_data1.max())
ax_1.set_ylabel('Single household\n density exponent')
ax_1.text(-0.6, 1, 'b)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_1)
cax = divider.append_axes("right", size="5%", pad=0.25)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Cumulative %\n prevalence")

axim = ax_2.imshow(100 * end_data2,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=100*end_data2.min(),
                  vmax=100*end_data2.max())
ax_2.set_ylabel('Single household\n density exponent')
ax_2.text(-1, 1, 'c)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_2)
cax = divider.append_axes("right", size="5%", pad=0.25)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Cumulative %\n prevalence")

axim = ax_3.imshow(100 * end_data3,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=100*end_data3.min(),
                  vmax=100*end_data3.max())
ax_3.set_ylabel('Single household\n density exponent')
ax_3.set_xlabel('Bubbled density\n exponent')
ax_3.text(-0.6, 1, 'd)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_3)
cax = divider.append_axes("right", size="5%", pad=0.25)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Cumulative %\n prevalence")

axim = ax_4.imshow(100 * end_data4,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=100*end_data4.min(),
                  vmax=100*end_data4.max())
ax_4.set_ylabel('Single household\n density exponent')
ax_4.set_xlabel('Bubbled density\n exponent')
ax_4.text(-1, 1, 'e)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_4)
cax = divider.append_axes("right", size="5%", pad=0.25)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Cumulative %\n prevalence")

ax_null.axis('off')

fig.savefig('plots/temp_bubbles/cum_prev_grid.png',bbox_inches='tight', dpi=300)
close()
