'''This plots the UK-like model with a single set of parameters for 100 days
'''
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap

with open('carehome_results.pkl', 'rb') as f:
    t, H, P, I= load(f)

lgd=['Patients','Full-time staff','Agency workers']

#clist=0.5*ones(10,3)
#clist(:,1)=(1/11)*(1:10)
#clist(:,3)=(1/11)*(10:-1:1)

fig, (axis_det, axis_undet) = subplots(2,1, sharex=True)

cmap = get_cmap('tab20')
alpha = 0.5
for i in range(3):
    axis_det.plot(
        t, P[:,i], label=lgd[i], alpha=alpha)
axis_det.set_ylabel('Prodromal prevalence')


for i in range(3):
    axis_undet.plot(
        t, I[:,i], label=lgd[i], alpha=alpha)
axis_undet.set_xlabel('Time in days')
axis_undet.set_ylabel('Fully infectious prevalence')

axis_det.legend(ncol=1, bbox_to_anchor=(1,0.50))

fig.savefig('carehome_results.png', bbox_inches='tight', dpi=300)
