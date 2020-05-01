'''This plots the UK-like model with a single set of parameters for 100 days
''' 
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap

with open('uk_like.pkl', 'rb') as f:
    t, H, D, U, coarse_bds = load(f)

lgd=[
    'Age {} to {}'.format(coarse_bds[i], coarse_bds[i+1])
    for i, _ in enumerate(coarse_bds[:-1])]

#clist=0.5*ones(10,3)
#clist(:,1)=(1/11)*(1:10)
#clist(:,3)=(1/11)*(10:-1:1)

fig, (axis_det, axis_undet) = subplots(2,1, sharex=True)

cmap = get_cmap('tab20')
alpha = 0.5
for i, _ in enumerate(coarse_bds[:-1]):
    axis_det.plot(
        t, D[:,i], label=lgd[i],
        color=cmap(i/len(coarse_bds)), alpha=alpha)
axis_det.set_ylabel('Detected prevalence')


for i, _ in enumerate(coarse_bds[:-1]):
    axis_undet.plot(
        t, U[:,i], label=lgd[i],
        color=cmap(i/len(coarse_bds)), alpha=alpha)
axis_undet.set_xlabel('Time in days')
axis_undet.set_ylabel('Undetected prevalence')

axis_det.legend(ncol=1, bbox_to_anchor=(1,0.50))

fig.savefig('uk_like.png', bbox_inches='tight', dpi=300)

