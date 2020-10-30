'''This plots the results from the exernal isolation working example.
'''
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.cm import get_cmap

with open('class_per_hh.pkl','rb') as f:
    class_per_hh = load(f)
with open('isolation_data_1e-2_0pt2.pkl', 'rb') as f:
    I_baseline,R_baseline,Q_baseline,baseline_time,I_OOHI_1e2_0pt2,R_OOHI_1e2_0pt2,Q_OOHI_1e2_0pt2,OOHI_time_1e2_0pt2,I_WHQ_1e2_0pt2,R_WHQ_1e2_0pt2,Q_WHQ_1e2_0pt2,WHQ_time_1e2_0pt2 = load(f)
with open('isolation_data_1e-2_0pt6.pkl', 'rb') as f:
    I_OOHI_1e2_0pt6,R_OOHI_1e2_0pt6,Q_OOHI_1e2_0pt6,OOHI_time_1e2_0pt6,I_WHQ_1e2_0pt6,R_WHQ_1e2_0pt6,Q_WHQ_1e2_0pt6,WHQ_time_1e2_0pt6 = load(f)
with open('isolation_data_1e-2_1.pkl', 'rb') as f:
    I_OOHI_1e2_1,R_OOHI_1e2_1,Q_OOHI_1e2_1,OOHI_time_1e2_1,I_WHQ_1e2_1,R_WHQ_1e2_1,Q_WHQ_1e2_1,WHQ_time_1e2_1 = load(f)
with open('isolation_data_1e-1_0pt2.pkl', 'rb') as f:
    I_OOHI_1e1_0pt2,R_OOHI_1e1_0pt2,Q_OOHI_1e1_0pt2,OOHI_time_1e1_0pt2,I_WHQ_1e1_0pt2,R_WHQ_1e1_0pt2,Q_WHQ_1e1_0pt2,WHQ_time_1e1_0pt2 = load(f)
with open('isolation_data_1e-1_0pt6.pkl', 'rb') as f:
    I_OOHI_1e1_0pt6,R_OOHI_1e1_0pt6,Q_OOHI_1e1_0pt6,OOHI_time_1e1_0pt6,I_WHQ_1e1_0pt6,R_WHQ_1e1_0pt6,Q_WHQ_1e1_0pt6,WHQ_time_1e1_0pt6 = load(f)
with open('isolation_data_1e-1_1.pkl', 'rb') as f:
    I_OOHI_1e1_1,R_OOHI_1e1_1,Q_OOHI_1e1_1,OOHI_time_1e1_1,I_WHQ_1e1_1,R_WHQ_1e1_1,Q_WHQ_1e1_1,WHQ_time_1e1_1 = load(f)
with open('isolation_data_1_0pt2.pkl', 'rb') as f:
    I_OOHI_1_0pt2,R_OOHI_1_0pt2,Q_OOHI_1_0pt2,OOHI_time_1_0pt2,I_WHQ_1_0pt2,R_WHQ_1_0pt2,Q_WHQ_1_0pt2,WHQ_time_1_0pt2 = load(f)
with open('isolation_data_1_0pt6.pkl', 'rb') as f:
    I_OOHI_1_0pt6,R_OOHI_1_0pt6,Q_OOHI_1_0pt6,OOHI_time_1_0pt6,I_WHQ_1_0pt6,R_WHQ_1_0pt6,Q_WHQ_1_0pt6,WHQ_time_1_0pt6 = load(f)
with open('isolation_data_1_1.pkl', 'rb') as f:
    I_OOHI_1_1,R_OOHI_1_1,Q_OOHI_1_1,OOHI_time_1_1,I_WHQ_1_1,R_WHQ_1_1,Q_WHQ_1_1,WHQ_time_1_1 = load(f)

print('1e-2, 0.2, OOHI, delta_R/Q=',(R_OOHI_1e2_0pt2[-1,2]-R_baseline[-1,2])/(Q_OOHI_1e2_0pt2[:,1].max()))
print('1e-2, 0.6, OOHI, delta_R/Q=',(R_OOHI_1e2_0pt6[-1,2]-R_baseline[-1,2])/(Q_OOHI_1e2_0pt6[:,1].max()))
print('1e-2, 1, OOHI, delta_R/Q=',(R_OOHI_1e2_1[-1,2]-R_baseline[-1,2])/(Q_OOHI_1e2_1[:,1].max()))
print('1e-1, 0.2, OOHI, delta_R/Q=',(R_OOHI_1e1_0pt2[-1,2]-R_baseline[-1,2])/(Q_OOHI_1e1_0pt2[:,1].max()))
print('1e-1, 0.6, OOHI, delta_R/Q=',(R_OOHI_1e1_0pt6[-1,2]-R_baseline[-1,2])/(Q_OOHI_1e1_0pt6[:,1].max()))
print('1e-1, 1, OOHI, delta_R/Q=',(R_OOHI_1e1_1[-1,2]-R_baseline[-1,2])/(Q_OOHI_1e1_1[:,1].max()))
print('1, 0.2, OOHI, delta_R/Q=',(R_OOHI_1_0pt2[-1,2]-R_baseline[-1,2])/(Q_OOHI_1_0pt2[:,1].max()))
print('1, 0.6, OOHI, delta_R/Q=',(R_OOHI_1_0pt6[-1,2]-R_baseline[-1,2])/(Q_OOHI_1_0pt6[:,1].max()))
print('1, 1, OOHI, delta_R/Q=',(R_OOHI_1_1[-1,2]-R_baseline[-1,2])/(Q_OOHI_1_1[:,1].max()))

print('1e-2, 0.2, WHQ, delta_R/Q=',(R_WHQ_1e2_0pt2[-1,2]-R_baseline[-1,2])/(Q_WHQ_1e2_0pt2[:,1].max()))
print('1e-2, 0.6, WHQ, delta_R/Q=',(R_WHQ_1e2_0pt6[-1,2]-R_baseline[-1,2])/(Q_WHQ_1e2_0pt6[:,1].max()))
print('1e-2, 1, WHQ, delta_R/Q=',(R_WHQ_1e2_1[-1,2]-R_baseline[-1,2])/(Q_WHQ_1e2_1[:,1].max()))
print('1e-1, 0.2, WHQ, delta_R/Q=',(R_WHQ_1e1_0pt2[-1,2]-R_baseline[-1,2])/(Q_WHQ_1e1_0pt2[:,1].max()))
print('1e-1, 0.6, WHQ, delta_R/Q=',(R_WHQ_1e1_0pt6[-1,2]-R_baseline[-1,2])/(Q_WHQ_1e1_0pt6[:,1].max()))
print('1e-1, 1, WHQ, delta_R/Q=',(R_WHQ_1e1_1[-1,2]-R_baseline[-1,2])/(Q_WHQ_1e1_1[:,1].max()))
print('1, 0.2, WHQ, delta_R/Q=',(R_WHQ_1_0pt2[-1,2]-R_baseline[-1,2])/(Q_WHQ_1_0pt2[:,1].max()))
print('1, 0.6, WHQ, delta_R/Q=',(R_WHQ_1_0pt6[-1,2]-R_baseline[-1,2])/(Q_WHQ_1_0pt6[:,1].max()))
print('1, 1, WHQ, delta_R/Q=',(R_WHQ_1_1[-1,2]-R_baseline[-1,2])/(Q_WHQ_1_1[:,1].max()))

lgd=['Children','Non-vulnerable adults','Vulnerable adults']

#clist=0.5*ones(10,3)
#clist(:,1)=(1/11)*(1:10)
#clist(:,3)=(1/11)*(10:-1:1)

fig, ((axis_I_0pt2, axis_I_0pt6, axis_I_1) , (axis_Q_0pt2, axis_Q_0pt6, axis_Q_1)) = subplots(2,3, sharex=True)

alpha = 0.5

axis_I_0pt2.plot(
    baseline_time, I_baseline[:,2]/class_per_hh[2], label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_I_0pt2.plot(
    WHQ_time_1e2_0pt2, I_WHQ_1e2_0pt2[:,2]/class_per_hh[2], label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_I_0pt2.plot(
    OOHI_time_1e2_0pt2, I_OOHI_1e2_0pt2[:,2]/class_per_hh[2], label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_I_0pt2.legend(ncol=1, bbox_to_anchor=(1,0.50))
# axis_I_0pt2.set_xlabel('Time in days')
axis_I_0pt2.set_ylabel('Prevalence among\n vulnerable population')
axis_I_0pt2.title.set_text('20% external\n contact reduction')
axis_I_0pt2.set_ylim([0,2e-5])

axis_I_0pt6.plot(
    baseline_time, I_baseline[:,2]/class_per_hh[2], label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_I_0pt6.plot(
    WHQ_time_1e2_0pt6, I_WHQ_1e2_0pt6[:,2]/class_per_hh[2], label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_I_0pt6.plot(
    OOHI_time_1e2_0pt6, I_OOHI_1e2_0pt6[:,2]/class_per_hh[2], label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_I_0pt6.legend(ncol=1, bbox_to_anchor=(1,0.50))
# axis_I_0pt6.set_xlabel('Time in days')
# axis_I_0pt6.set_ylabel('Prevalence among vulnerable population')
axis_I_0pt6.set_yticklabels([])
axis_I_0pt6.title.set_text('60% external\n contact reduction')
axis_I_0pt6.set_ylim([0,2e-5])

axis_I_1.plot(
    baseline_time, I_baseline[:,2]/class_per_hh[2], label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_I_1.plot(
    WHQ_time_1e2_1, I_WHQ_1e2_1[:,2]/class_per_hh[2], label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_I_1.plot(
    OOHI_time_1e2_1, I_OOHI_1e2_1[:,2]/class_per_hh[2], label='OOHI',
    color='b', alpha=alpha, linewidth=2)

axis_I_1.legend(ncol=1, bbox_to_anchor=(1,0.50))
# axis_I_1.set_xlabel('Time in days')
# axis_I_1.set_ylabel('Prevalence among vulnerable population')
axis_I_1.set_yticklabels([])
axis_I_1.title.set_text('100% external\n contact reduction')
axis_I_1.set_ylim([0,2e-5])

axis_Q_0pt2.plot(
    baseline_time, sum(Q_baseline.T)/sum(class_per_hh), label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_Q_0pt2.plot(
    WHQ_time_1e2_0pt2, sum(Q_WHQ_1e2_0pt2.T)/sum(class_per_hh), label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_Q_0pt2.plot(
    OOHI_time_1e2_0pt2, sum(Q_OOHI_1e2_0pt2.T)/sum(class_per_hh), label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_I_0pt2.legend(ncol=1, bbox_to_anchor=(1,0.50))
axis_Q_0pt2.set_xlabel('Time in days')
axis_Q_0pt2.set_ylabel('Proportion of population\n in isolation')
axis_Q_0pt2.set_ylim([0,6e-5])

axis_Q_0pt6.plot(
    baseline_time, sum(Q_baseline.T)/sum(class_per_hh), label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_Q_0pt6.plot(
    WHQ_time_1e2_0pt6, sum(Q_WHQ_1e2_0pt6.T)/sum(class_per_hh), label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_Q_0pt2.plot(
    OOHI_time_1e2_0pt6, sum(Q_OOHI_1e2_0pt6.T)/sum(class_per_hh), label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_I_0pt6.legend(ncol=1, bbox_to_anchor=(1,0.50))
axis_Q_0pt6.set_xlabel('Time in days')
# axis_I_0pt6.set_ylabel('Prevalence among vulnerable population')
axis_Q_0pt6.set_yticklabels([])
axis_Q_0pt6.set_ylim([0,6e-5])

axis_Q_1.plot(
    baseline_time, sum(Q_baseline.T)/sum(class_per_hh), label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_Q_1.plot(
    WHQ_time_1e2_1, sum(Q_WHQ_1e2_1.T)/sum(class_per_hh), label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_Q_1.plot(
    OOHI_time_1e2_1, sum(Q_OOHI_1e2_1.T)/sum(class_per_hh), label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_Q_1.legend(ncol=1, bbox_to_anchor=(1,0.50))
axis_Q_1.set_xlabel('Time in days')
# axis_I_1.set_ylabel('Prevalence among vulnerable population')
axis_Q_1.set_yticklabels([])
axis_Q_1.set_ylim([0,6e-5])

fig.savefig('isolation_plot_1e-2.png', bbox_inches='tight', dpi=300)



fig, ((axis_I_0pt2, axis_I_0pt6, axis_I_1) , (axis_Q_0pt2, axis_Q_0pt6, axis_Q_1)) = subplots(2,3, sharex=True)

alpha = 0.5

axis_I_0pt2.plot(
    baseline_time, I_baseline[:,2]/class_per_hh[2], label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_I_0pt2.plot(
    WHQ_time_1e1_0pt2, I_WHQ_1e1_0pt2[:,2]/class_per_hh[2], label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_I_0pt2.plot(
    OOHI_time_1e1_0pt2, I_OOHI_1e1_0pt2[:,2]/class_per_hh[2], label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_I_0pt2.legend(ncol=1, bbox_to_anchor=(1,0.50))
# axis_I_0pt2.set_xlabel('Time in days')
axis_I_0pt2.set_ylabel('Prevalence among\n vulnerable population')
axis_I_0pt2.title.set_text('20% external\n contact reduction')
axis_I_0pt2.set_ylim([0,2e-5])

axis_I_0pt6.plot(
    baseline_time, I_baseline[:,2]/class_per_hh[2], label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_I_0pt6.plot(
    WHQ_time_1e1_0pt6, I_WHQ_1e1_0pt6[:,2]/class_per_hh[2], label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_I_0pt6.plot(
    OOHI_time_1e1_0pt6, I_OOHI_1e1_0pt6[:,2]/class_per_hh[2], label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_I_0pt6.legend(ncol=1, bbox_to_anchor=(1,0.50))
# axis_I_0pt6.set_xlabel('Time in days')
# axis_I_0pt6.set_ylabel('Prevalence among vulnerable population')
axis_I_0pt6.set_yticklabels([])
axis_I_0pt6.title.set_text('60% external\n contact reduction')
axis_I_0pt6.set_ylim([0,2e-5])

axis_I_1.plot(
    baseline_time, I_baseline[:,2]/class_per_hh[2], label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_I_1.plot(
    WHQ_time_1e1_1, I_WHQ_1e1_1[:,2]/class_per_hh[2], label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_I_1.plot(
    OOHI_time_1e1_1, I_OOHI_1e1_1[:,2]/class_per_hh[2], label='OOHI',
    color='b', alpha=alpha, linewidth=2)

axis_I_1.legend(ncol=1, bbox_to_anchor=(1,0.50))
# axis_I_1.set_xlabel('Time in days')
# axis_I_1.set_ylabel('Prevalence among vulnerable population')
axis_I_1.set_yticklabels([])
axis_I_1.title.set_text('100% external\n contact reduction')
axis_I_1.set_ylim([0,2e-5])

axis_Q_0pt2.plot(
    baseline_time, sum(Q_baseline.T)/sum(class_per_hh), label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_Q_0pt2.plot(
    WHQ_time_1e1_0pt2, sum(Q_WHQ_1e1_0pt2.T)/sum(class_per_hh), label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_Q_0pt2.plot(
    OOHI_time_1e1_0pt2, sum(Q_OOHI_1e1_0pt2.T)/sum(class_per_hh), label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_I_0pt2.legend(ncol=1, bbox_to_anchor=(1,0.50))
axis_Q_0pt2.set_xlabel('Time in days')
axis_Q_0pt2.set_ylabel('Proportion of population\n in isolation')
axis_Q_0pt2.set_ylim([0,2e-4])

axis_Q_0pt6.plot(
    baseline_time, sum(Q_baseline.T)/sum(class_per_hh), label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_Q_0pt6.plot(
    WHQ_time_1e1_0pt6, sum(Q_WHQ_1e1_0pt6.T)/sum(class_per_hh), label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_Q_0pt6.plot(
    OOHI_time_1e1_0pt6, sum(Q_OOHI_1e1_0pt6.T)/sum(class_per_hh), label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_I_0pt6.legend(ncol=1, bbox_to_anchor=(1,0.50))
axis_Q_0pt6.set_xlabel('Time in days')
# axis_I_0pt6.set_ylabel('Prevalence among vulnerable population')
axis_Q_0pt6.set_yticklabels([])
axis_Q_0pt6.set_ylim([0,2e-4])

axis_Q_1.plot(
    baseline_time, sum(Q_baseline.T)/sum(class_per_hh), label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_Q_1.plot(
    WHQ_time_1e1_1, sum(Q_WHQ_1e1_1.T)/sum(class_per_hh), label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_Q_1.plot(
    OOHI_time_1e1_1, sum(Q_OOHI_1e1_1.T)/sum(class_per_hh), label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_Q_1.legend(ncol=1, bbox_to_anchor=(1,0.50))
axis_Q_1.set_xlabel('Time in days')
# axis_I_1.set_ylabel('Prevalence among vulnerable population')
axis_Q_1.set_yticklabels([])
axis_Q_1.set_ylim([0,2e-4])

fig.savefig('isolation_plot_1e-1.png', bbox_inches='tight', dpi=300)



fig, ((axis_I_0pt2, axis_I_0pt6, axis_I_1) , (axis_Q_0pt2, axis_Q_0pt6, axis_Q_1)) = subplots(2,3, sharex=True)

alpha = 0.5

axis_I_0pt2.plot(
    baseline_time, I_baseline[:,2]/class_per_hh[2], label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_I_0pt2.plot(
    WHQ_time_1_0pt2, I_WHQ_1_0pt2[:,2]/class_per_hh[2], label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_I_0pt2.plot(
    OOHI_time_1_0pt2, I_OOHI_1_0pt2[:,2]/class_per_hh[2], label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_I_0pt2.legend(ncol=1, bbox_to_anchor=(1,0.50))
# axis_I_0pt2.set_xlabel('Time in days')
axis_I_0pt2.set_ylabel('Prevalence among\n vulnerable population')
axis_I_0pt2.title.set_text('20% external\n contact reduction')
axis_I_0pt2.set_ylim([0,2e-5])

axis_I_0pt6.plot(
    baseline_time, I_baseline[:,2]/class_per_hh[2], label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_I_0pt6.plot(
    WHQ_time_1_0pt6, I_WHQ_1_0pt6[:,2]/class_per_hh[2], label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_I_0pt6.plot(
    OOHI_time_1_0pt6, I_OOHI_1_0pt6[:,2]/class_per_hh[2], label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_I_0pt6.legend(ncol=1, bbox_to_anchor=(1,0.50))
# axis_I_0pt6.set_xlabel('Time in days')
# axis_I_0pt6.set_ylabel('Prevalence among vulnerable population')
axis_I_0pt6.set_yticklabels([])
axis_I_0pt6.title.set_text('60% external\n contact reduction')
axis_I_0pt6.set_ylim([0,2e-5])

axis_I_1.plot(
    baseline_time, I_baseline[:,2]/class_per_hh[2], label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_I_1.plot(
    WHQ_time_1_1, I_WHQ_1_1[:,2]/class_per_hh[2], label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_I_1.plot(
    OOHI_time_1_1, I_OOHI_1_1[:,2]/class_per_hh[2], label='OOHI',
    color='b', alpha=alpha, linewidth=2)

axis_I_1.legend(ncol=1, bbox_to_anchor=(1,0.50))
# axis_I_1.set_xlabel('Time in days')
# axis_I_1.set_ylabel('Prevalence among vulnerable population')
axis_I_1.set_yticklabels([])
axis_I_1.title.set_text('100% external\n contact reduction')
axis_I_1.set_ylim([0,2e-5])

axis_Q_0pt2.plot(
    baseline_time, sum(Q_baseline.T)/sum(class_per_hh), label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_Q_0pt2.plot(
    WHQ_time_1_0pt2, sum(Q_WHQ_1_0pt2.T)/sum(class_per_hh), label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_Q_0pt2.plot(
    OOHI_time_1_0pt2, sum(Q_OOHI_1_0pt2.T)/sum(class_per_hh), label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_I_0pt2.legend(ncol=1, bbox_to_anchor=(1,0.50))
# axis_Q_0pt2.set_xlabel('Time in days')
axis_Q_0pt2.set_ylabel('Proportion of population\n in isolation')
axis_Q_0pt2.set_ylim([0,2e-4])

axis_Q_0pt6.plot(
    baseline_time, sum(Q_baseline.T)/sum(class_per_hh), label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_Q_0pt6.plot(
    WHQ_time_1_0pt6, sum(Q_WHQ_1_0pt6.T)/sum(class_per_hh), label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_Q_0pt6.plot(
    OOHI_time_1_0pt6, sum(Q_OOHI_1_0pt6.T)/sum(class_per_hh), label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_I_0pt6.legend(ncol=1, bbox_to_anchor=(1,0.50))
# axis_Q_0pt6.set_xlabel('Time in days')
# axis_I_0pt6.set_ylabel('Prevalence among vulnerable population')
axis_Q_0pt6.set_yticklabels([])
axis_Q_0pt6.set_ylim([0,2e-4])

axis_Q_1.plot(
    baseline_time, sum(Q_baseline.T)/sum(class_per_hh), label='No intervention',
    color='k', alpha=alpha, linewidth=2)
axis_Q_1.plot(
    WHQ_time_1_1, sum(Q_WHQ_1_1.T)/sum(class_per_hh), label='Within-household isolation',
    color='r', alpha=alpha, linewidth=2)
axis_Q_1.plot(
    OOHI_time_1_1, sum(Q_OOHI_1_1.T)/sum(class_per_hh), label='OOHI',
    color='b', alpha=alpha, linewidth=2)

# axis_Q_1.legend(ncol=1, bbox_to_anchor=(1,0.50))
# axis_Q_1.set_xlabel('Time in days')
# axis_I_1.set_ylabel('Prevalence among vulnerable population')
axis_Q_1.set_yticklabels([])
axis_Q_1.set_ylim([0,2e-4])

fig.savefig('isolation_plot_1.png', bbox_inches='tight', dpi=300)
