from os import mkdir
from os.path import isdir
from math import ceil, floor
from pickle import dump, load
from numpy import arange, array, atleast_2d, hstack, where, zeros
from matplotlib.pyplot import axes, close, colorbar, imshow, subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable

if isdir('plots/temp_bubbles') is False:
    mkdir('plots/temp_bubbles')

with open('outputs/temp_bubbles/baseline_ts.pkl', 'rb') as f:
    (baseline_ts) = load(f)
with open('outputs/temp_bubbles/ts_0.pkl', 'rb') as f:
    (ts_0) = load(f)
with open('outputs/temp_bubbles/ts_1.pkl', 'rb') as f:
    (ts_1) = load(f)
with open('outputs/temp_bubbles/ts_2.pkl', 'rb') as f:
    (ts_2) = load(f)
with open('outputs/temp_bubbles/ts_3.pkl', 'rb') as f:
    (ts_3) = load(f)
with open('outputs/temp_bubbles/ts_4.pkl', 'rb') as f:
    (ts_4) = load(f)

fig, ax = subplots()
ax.plot(baseline_ts[0],baseline_ts[2]+baseline_ts[3],label='bs')
ax.plot(ts_0[0],ts_0[2]+ts_0[3],label='ts_0')
ax.plot(ts_1[0],ts_1[2]+ts_1[3],label='ts_1')
ax.plot(ts_2[0],ts_2[2]+ts_2[3],label='ts_2')
ax.plot(ts_3[0],ts_3[2]+ts_3[3],label='ts_3')
ax.plot(ts_4[0],ts_4[2]+ts_4[3],label='ts_4')
ax.legend()
fig.savefig('plots/temp_bubbles/e_i_ts.png',bbox_inches='tight', dpi=300)
close()

fig, ax = subplots()
ax.plot(baseline_ts[0],baseline_ts[4],label='bs')
ax.plot(ts_0[0],ts_0[4],label='ts_0')
ax.plot(ts_1[0],ts_1[4],label='ts_1')
ax.plot(ts_2[0],ts_2[4],label='ts_2')
ax.plot(ts_3[0],ts_3[4],label='ts_3')
ax.plot(ts_4[0],ts_4[4],label='ts_4')
ax.legend()
fig.savefig('plots/temp_bubbles/r_ts.png',bbox_inches='tight', dpi=300)
close()
