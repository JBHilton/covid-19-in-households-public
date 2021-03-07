'''This plots the UK-like model with a single set of parameters for 100 days
'''
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap

with open('uk_like.pkl', 'rb') as f:
    H, time_series, model_input = load(f)

t = time_series['time']
data_list = [time_series['S']/model_input.ave_hh_by_class,
    time_series['E']/model_input.ave_hh_by_class,
    time_series['P']/model_input.ave_hh_by_class,
    time_series['I']/model_input.ave_hh_by_class,
    time_series['R']/model_input.ave_hh_by_class]

lgd=['S','E','P','I','R']

fig, (axis_C, axis_A) = subplots(2,1, sharex=True)

cmap = get_cmap('tab20')
alpha = 0.5
for i in range(len(data_list)):
    axis_C.plot(
        t, data_list[i][:,0], label=lgd[i],
        color=cmap(i/len(data_list)), alpha=alpha)
axis_C.set_ylabel('Proportion of population')
axis_C.set_title('Children (0-19 years old)')
axis_C.legend(ncol=1, bbox_to_anchor=(1,0.50))

for i in range(len(data_list)):
    axis_A.plot(
        t, data_list[i][:,1], label=lgd[i],
        color=cmap(i/len(data_list)), alpha=alpha)
axis_A.set_ylabel('Proportion of population')
axis_A.set_title('Adults (20+ years old)')
axis_A.legend(ncol=1, bbox_to_anchor=(1,0.50))

fig.savefig('uk_like.png', bbox_inches='tight', dpi=300)
