'''This plots the long-term support bubbling output with a single set of parameters for 100 days
'''
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap

with open('support_bubble_output.pkl', 'rb') as f:
    baseline_time_series, bubbled_time_series, bubbled_model_input, baseline_model_input = load(f)

t = bubbled_time_series['time']
data_list = [bubbled_time_series['S']/bubbled_model_input.ave_hh_by_class,
    bubbled_time_series['E']/bubbled_model_input.ave_hh_by_class,
    bubbled_time_series['P']/bubbled_model_input.ave_hh_by_class,
    bubbled_time_series['I']/bubbled_model_input.ave_hh_by_class,
    bubbled_time_series['R']/bubbled_model_input.ave_hh_by_class]

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

fig.savefig('support_bubble_projection.png', bbox_inches='tight', dpi=300)

t = baseline_time_series['time']
data_list = [baseline_time_series['S']/baseline_model_input.ave_hh_by_class,
    baseline_time_series['E']/baseline_model_input.ave_hh_by_class,
    baseline_time_series['P']/baseline_model_input.ave_hh_by_class,
    baseline_time_series['I']/baseline_model_input.ave_hh_by_class,
    baseline_time_series['R']/baseline_model_input.ave_hh_by_class]

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

fig.savefig('baseline_projection.png', bbox_inches='tight', dpi=300)
