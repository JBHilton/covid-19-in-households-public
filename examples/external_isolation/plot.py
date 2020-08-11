'''This plots the results from the exernal isolation working example.
'''
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap

with open('external-isolation-results.pkl', 'rb') as f:
    t, H, P, I, Q, children_per_hh, nonv_adults_per_hh, vuln_adults_per_hh = load(f)

class_per_hh = [children_per_hh, nonv_adults_per_hh, vuln_adults_per_hh]

print(class_per_hh)

lgd=['Children','Non-vulnerable adults','Vulnerable adults']

#clist=0.5*ones(10,3)
#clist(:,1)=(1/11)*(1:10)
#clist(:,3)=(1/11)*(10:-1:1)

fig, (axis_det, axis_undet, axis_isolate) = subplots(3,1, sharex=True)

cmap = get_cmap('tab20')
alpha = 0.5
for i in range(3):
    axis_det.plot(
        t,P[:,i]/class_per_hh[i], label=lgd[i],
        color=cmap(i/3), alpha=alpha)
axis_det.set_ylabel('Prodromal prevalence')


for i in range(3):
    axis_undet.plot(
        t, I[:,i]/class_per_hh[i], label=lgd[i],
        color=cmap(i/3), alpha=alpha)
axis_undet.set_ylabel('Infectious prevalence')

for i in range(3):
    axis_isolate.plot(
        t, Q[:,i]/class_per_hh[i], label=lgd[i],
        color=cmap(i/3), alpha=alpha)
axis_isolate.set_xlabel('Time in days')
axis_isolate.set_ylabel('Proportion in isolation')

axis_det.legend(ncol=1, bbox_to_anchor=(1,0.50))

fig.savefig('iso-plot.png', bbox_inches='tight', dpi=300)
