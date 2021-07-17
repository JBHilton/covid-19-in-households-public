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
    (peaks_1,
         jan_antiprev_1,
         peaks_2,
         jan_antiprev_2,
         peaks_3,
         jan_antiprev_3,
         peaks_4,
         jan_antiprev_4,
         unmerged_exponents,
         merged_exponents) = load(f)

no_i_vals = 3
no_j_vals = 3

pol_label = []
for pol in range(4):
    pol_label.append('Policy'+str(pol+1))

peaks_min = array([peaks_1.min(),peaks_2.min(),peaks_3.min(),peaks_4.min()]).min()
peaks_max = array([peaks_1.max(),peaks_2.max(),peaks_3.max(),peaks_4.max()]).max()

fig, ax = subplots(1,1,sharex=True)
imshow(peaks_1, origin='lower',extent=(0, 1, 0, 1), vmin=peaks_min, vmax=peaks_max)
ax.set_title(pol_label[0])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/peaks_1.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(peaks_2, origin='lower',extent=(0, 1, 0, 1), vmin=peaks_min, vmax=peaks_max)
ax.set_title(pol_label[1])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')
cbar = colorbar(label="Peak prevalence",fraction=0.046, pad=0.04)

fig.savefig('plots/temp_bubbles/peaks_2.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(peaks_3, origin='lower',extent=(0, 1, 0, 1), vmin=peaks_min, vmax=peaks_max)
ax.set_title(pol_label[2])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/peaks_3.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(peaks_4, origin='lower',extent=(0, 1, 0, 1), vmin=peaks_min, vmax=peaks_max)
ax.set_title(pol_label[3])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/peaks_4.png',bbox_inches='tight', dpi=300)
close()

antiprev_min = array([jan_antiprev_1.min(),jan_antiprev_2.min(),jan_antiprev_3.min(),jan_antiprev_4.min()]).min()
antiprev_max = array([jan_antiprev_1.max(),jan_antiprev_2.max(),jan_antiprev_3.max(),jan_antiprev_4.max()]).max()

fig, ax = subplots(1,1,sharex=True)
imshow(jan_antiprev_1, origin='lower',extent=(0, 1, 0, 1), vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[0])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/jan_antiprev_1.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(jan_antiprev_2, origin='lower',extent=(0, 1, 0, 1), vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[1])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')
cbar = colorbar(label="Population immunity by Feb 1st",fraction=0.046, pad=0.04)

fig.savefig('plots/temp_bubbles/jan_antiprev_2.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(jan_antiprev_3, origin='lower',extent=(0, 1, 0, 1), vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[2])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/jan_antiprev_3.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots(1,1,sharex=True)
imshow(jan_antiprev_4, origin='lower',extent=(0, 1, 0, 1), vmin=antiprev_min, vmax=antiprev_max)
ax.set_title(pol_label[3])
ax.set_ylabel('Single household density exponent')
ax.set_xlabel('Bubbled density exponent')

fig.savefig('plots/temp_bubbles/jan_antiprev_4.png',bbox_inches='tight', dpi=300)
close()
