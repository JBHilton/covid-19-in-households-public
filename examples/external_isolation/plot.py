'''This plots the results from the exernal isolation working example.
'''
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap


with open('outputs/oohi/results.pkl', 'rb') as f:
    model_input, I_OOHI,R_OOHI,Q_OOHI,OOHI_time,I_WHQ,R_WHQ,Q_WHQ,WHQ_time = load(f)
class_per_hh = model_input.ave_hh_by_class
# with open('isolation_data_1e-2_0pt6.pkl', 'rb') as f:
#     I_OOHI_1e2_0pt6,R_OOHI_1e2_0pt6,Q_OOHI_1e2_0pt6,OOHI_time_1e2_0pt6,I_WHQ_1e2_0pt6,R_WHQ_1e2_0pt6,Q_WHQ_1e2_0pt6,WHQ_time_1e2_0pt6 = load(f)
# with open('isolation_data_1e-2_1.pkl', 'rb') as f:
#     I_OOHI_1e2_1,R_OOHI_1e2_1,Q_OOHI_1e2_1,OOHI_time_1e2_1,I_WHQ_1e2_1,R_WHQ_1e2_1,Q_WHQ_1e2_1,WHQ_time_1e2_1 = load(f)
# with open('isolation_data_1e-1_0pt2.pkl', 'rb') as f:
#     I_OOHI_1e1_0pt2,R_OOHI_1e1_0pt2,Q_OOHI_1e1_0pt2,OOHI_time_1e1_0pt2,I_WHQ_1e1_0pt2,R_WHQ_1e1_0pt2,Q_WHQ_1e1_0pt2,WHQ_time_1e1_0pt2 = load(f)
# with open('isolation_data_1e-1_0pt6.pkl', 'rb') as f:
#     I_OOHI_1e1_0pt6,R_OOHI_1e1_0pt6,Q_OOHI_1e1_0pt6,OOHI_time_1e1_0pt6,I_WHQ_1e1_0pt6,R_WHQ_1e1_0pt6,Q_WHQ_1e1_0pt6,WHQ_time_1e1_0pt6 = load(f)
# with open('isolation_data_1e-1_1.pkl', 'rb') as f:
#     I_OOHI_1e1_1,R_OOHI_1e1_1,Q_OOHI_1e1_1,OOHI_time_1e1_1,I_WHQ_1e1_1,R_WHQ_1e1_1,Q_WHQ_1e1_1,WHQ_time_1e1_1 = load(f)
# with open('isolation_data_1_0pt2.pkl', 'rb') as f:
#     I_OOHI_1_0pt2,R_OOHI_1_0pt2,Q_OOHI_1_0pt2,OOHI_time_1_0pt2,I_WHQ_1_0pt2,R_WHQ_1_0pt2,Q_WHQ_1_0pt2,WHQ_time_1_0pt2 = load(f)
# with open('isolation_data_1_0pt6.pkl', 'rb') as f:
#     I_OOHI_1_0pt6,R_OOHI_1_0pt6,Q_OOHI_1_0pt6,OOHI_time_1_0pt6,I_WHQ_1_0pt6,R_WHQ_1_0pt6,Q_WHQ_1_0pt6,WHQ_time_1_0pt6 = load(f)
# with open('isolation_data_1_1.pkl', 'rb') as f:
#     I_OOHI_1_1,R_OOHI_1_1,Q_OOHI_1_1,OOHI_time_1_1,I_WHQ_1_1,R_WHQ_1_1,Q_WHQ_1_1,WHQ_time_1_1 = load(f)
lgd=['Children','Non-vulnerable adults','Vulnerable adults']

#clist=0.5*ones(10,3)
#clist(:,1)=(1/11)*(1:10)
#clist(:,3)=(1/11)*(10:-1:1)

fig, ax = subplots()

alpha = 0.5
ax.plot(
    WHQ_time, I_WHQ[:,2]/class_per_hh[2], label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
ax.plot(
    OOHI_time, I_OOHI[:,2]/class_per_hh[2], label='OOHI',
    color='b', alpha=alpha, linewidth=2)

ax.legend(ncol=1, bbox_to_anchor=(1,0.50))
ax.set_xlabel('Time in days')
ax.set_ylabel('Prevalence among\n vulnerable population')
ax.set_ylim([0,2e-5])

fig.savefig('isolation_plot.png', bbox_inches='tight', dpi=300)



fig, ax= subplots()
