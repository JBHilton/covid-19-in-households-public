'''This plots the results from the simple example with a single set of
parameters for 30 days
'''
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap

with open('carehome_vacc_output.pkl', 'rb') as f:
    H, time_series, model_input = load(f)

lgd=['S','E','M', 'C', 'R', 'D']

t = time_series['time']
data_list = [time_series['S']/model_input.ave_hh_by_class,
    time_series['E']/model_input.ave_hh_by_class,
    time_series['M']/model_input.ave_hh_by_class,
    time_series['C']/model_input.ave_hh_by_class,
    time_series['R']/model_input.ave_hh_by_class,
    time_series['D']/model_input.ave_hh_by_class]

fig, (axis_P, axis_S, axis_A) = subplots(3,1, sharex=True)

cmap = get_cmap('tab20')
alpha = 0.5
for i in range(1,len(data_list)-2):
    axis_P.plot(
        t, data_list[i][:,0], label=lgd[i],
        color=cmap(i/len(data_list)), alpha=alpha)
axis_P.set_ylabel('Proportion of population')
axis_P.set_title('Patients')
axis_P.legend(ncol=1, bbox_to_anchor=(1,0.50))

for i in range(1,len(data_list)-2):
    axis_S.plot(
        t, data_list[i][:,1], label=lgd[i],
        color=cmap(i/len(data_list)), alpha=alpha)
axis_S.set_ylabel('Proportion of population')
axis_S.set_title('Permanent staff')
axis_S.legend(ncol=1, bbox_to_anchor=(1,0.50))

for i in range(1,len(data_list)-2):
    axis_A.plot(
        t, data_list[i][:,2], label=lgd[i],
        color=cmap(i/len(data_list)), alpha=alpha)
axis_A.set_ylabel('Proportion of population')
axis_A.set_title('Agency workers')
axis_A.legend(ncol=1, bbox_to_anchor=(1,0.50))

fig.savefig('carehome-plot.png', bbox_inches='tight', dpi=300)

fig, ax = subplots(1,1,sharex=True)
ax.plot(t,data_list[0][:,0] +
            data_list[1][:,0] +
            data_list[2][:,0] +
            data_list[3][:,0] +
            data_list[4][:,0])
fig.savefig('total_patients.png', bbox_inches='tight', dpi=300)
