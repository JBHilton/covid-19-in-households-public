'''This plots the bubble results
'''
from os import mkdir
from os.path import isdir
from pickle import dump, load
from numpy import array, atleast_2d, hstack, where, zeros
from matplotlib.pyplot import close, colorbar, imshow, subplots

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

peak_datamin = array([peak_data0.min(),peak_data1.min(),peak_data2.min(),peak_data3.min(),peak_data4.min()]).min()
peak_datamax = array([peak_data0.max(),peak_data1.max(),peak_data2.max(),peak_data3.max(),peak_data4.max()]).max()

fig, ax = subplots(1,1,sharex=True)
imshow(peak_data0, origin='lower',extent=(0, 1, 0, 1), vmin=peak_datamin, vmax=peak_datamax)
ax.set_title(pol_label[0])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/peak_data0.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(peak_data1, origin='lower',extent=(0, 1, 0, 1), vmin=peak_datamin, vmax=peak_datamax)
ax.set_title(pol_label[1])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/peak_data1.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(peak_data2, origin='lower',extent=(0, 1, 0, 1), vmin=peak_datamin, vmax=peak_datamax)
ax.set_title(pol_label[2])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')
cbar = colorbar(label="Peak prevalence",fraction=0.046, pad=0.04)

fig.savefig('plots/temp_bubbles/peak_data2.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(peak_data3, origin='lower',extent=(0, 1, 0, 1), vmin=peak_datamin, vmax=peak_datamax)
ax.set_title(pol_label[3])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/peak_data3.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(peak_data4, origin='lower',extent=(0, 1, 0, 1), vmin=peak_datamin, vmax=peak_datamax)
ax.set_title(pol_label[4])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/peak_data4.png',bbox_inches='tight', dpi=300)
close()

antiprev_min = array([end_data1.min(),end_data2.min(),end_data3.min(),end_data4.min()]).min()
antiprev_max = array([end_data1.max(),end_data2.max(),end_data3.max(),end_data4.max()]).max()

fig, ax = subplots(1,1,sharex=True)
imshow(end_data0, origin='lower',extent=(0, 1, 0, 1), vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[0])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/end_data0.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(end_data1, origin='lower',extent=(0, 1, 0, 1), vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[1])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/end_data1.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(end_data2, origin='lower',extent=(0, 1, 0, 1), vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[2])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')
cbar = colorbar(label="Population immunity by Feb 1st",fraction=0.046, pad=0.04)

fig.savefig('plots/temp_bubbles/end_data2.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(end_data3, origin='lower',extent=(0, 1, 0, 1), vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[3])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/end_data3.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(end_data4, origin='lower',extent=(0, 1, 0, 1), vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[4])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/end_data4.png',bbox_inches='tight', dpi=300)
close()

ar_min = array([ar_data1.min(),ar_data2.min(),ar_data3.min(),ar_data4.min()]).min()
ar_max = array([ar_data1.max(),ar_data2.max(),ar_data3.max(),ar_data4.max()]).max()

fig, ax = subplots(1,1,sharex=True)
imshow(ar_data0, origin='lower',extent=(0, 1, 0, 1), vmin=ar_min, vmax=ar_max)
ax.set_title(pol_label[0])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/ar_data0.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(ar_data1, origin='lower',extent=(0, 1, 0, 1), vmin=ar_min, vmax=ar_max)
ax.set_title(pol_label[1])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/ar_data1.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(ar_data2, origin='lower',extent=(0, 1, 0, 1), vmin=ar_min, vmax=ar_max)
ax.set_title(pol_label[2])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')
cbar = colorbar(label="Population immunity by Feb 1st",fraction=0.046, pad=0.04)

fig.savefig('plots/temp_bubbles/ar_data2.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(ar_data3, origin='lower',extent=(0, 1, 0, 1), vmin=ar_min, vmax=ar_max)
ax.set_title(pol_label[3])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/ar_data3.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(ar_data4, origin='lower',extent=(0, 1, 0, 1), vmin=ar_min, vmax=ar_max)
ax.set_title(pol_label[4])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/ar_data4.png',bbox_inches='tight', dpi=300)
close()

hh_prop_min = array([hh_prop_data1.min(),hh_prop_data2.min(),hh_prop_data3.min(),hh_prop_data4.min()]).min()
hh_prop_max = array([hh_prop_data1.max(),hh_prop_data2.max(),hh_prop_data3.max(),hh_prop_data4.max()]).max()

fig, ax = subplots(1,1,sharex=True)
imshow(hh_prop_data0, origin='lower',extent=(0, 1, 0, 1), vmin=hh_prop_min, vmax=hh_prop_max)
ax.set_title(pol_label[0])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/hh_prop_data0.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(hh_prop_data1, origin='lower',extent=(0, 1, 0, 1), vmin=hh_prop_min, vmax=hh_prop_max)
ax.set_title(pol_label[1])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/hh_prop_data1.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(hh_prop_data2, origin='lower',extent=(0, 1, 0, 1), vmin=hh_prop_min, vmax=hh_prop_max)
ax.set_title(pol_label[2])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')
cbar = colorbar(label="Population immunity by Feb 1st",fraction=0.046, pad=0.04)

fig.savefig('plots/temp_bubbles/hh_prop_data2.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(hh_prop_data3, origin='lower',extent=(0, 1, 0, 1), vmin=hh_prop_min, vmax=hh_prop_max)
ax.set_title(pol_label[3])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/hh_prop_data3.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(hh_prop_data4, origin='lower',extent=(0, 1, 0, 1), vmin=hh_prop_min, vmax=hh_prop_max)
ax.set_title(pol_label[4])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/hh_prop_data4.png',bbox_inches='tight', dpi=300)
close()
