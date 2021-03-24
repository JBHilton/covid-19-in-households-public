'''This plots the results from the vaccine efficacy sweep
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
xticklabels = ['0.0','0.2','0.4','0.6','0.8','1.0']
yticklabels= ['0.0','0.2','0.4','0.6','0.8','1.0']
out1 = np.array([[0.10995843, 0.09935811, 0.08773471, 0.0736153,  0.05298196, 0.01343789],
  [0.10612607, 0.09607337, 0.08497362, 0.07138105, 0.05137669, 0.01309041],
  [0.10171467, 0.09226986, 0.08173868, 0.06870564, 0.0493861,  0.01265149],
  [0.09628123, 0.08754191, 0.0776502,  0.06522866, 0.0466981,  0.01204953],
  [0.08876208, 0.08088543, 0.07173798, 0.0600057,  0.04249238, 0.01108654],
  [0.07645057, 0.06955102, 0.0611788,  0.05022487, 0.03444488, 0.00913675]])

fig, ax = plt.subplots(1,  1, sharex=True)
sns.heatmap(out1,vmax=.13, vmin=0, square=True, cbar_kws={'label': 'Expected cases per home after 30 days'}, ax=ax)
ax.invert_yaxis()
ax.set_ylim(0,6)
ax.set_xticklabels(xticklabels)
ax.set_yticklabels(yticklabels)
plt.title("b=0.1")
ax.set(xlabel='Efficacy in staff', ylabel='Efficacy in residents')
fig.savefig('vacc_eff_b_is_0pt1.png', bbox_inches='tight', dpi=300)

out2 = np.array([[0.12823899, 0.1172589,  0.10596901, 0.09318963, 0.07457758, 0.01666329],
  [0.12393788, 0.11352122, 0.10275425, 0.09046668, 0.07235433, 0.01632297],
  [0.11909388, 0.10931994, 0.09913186, 0.08735342, 0.06968405, 0.01588078],
  [0.11318363, 0.10420531, 0.09470751, 0.08348029, 0.06616831, 0.01526018],
  [0.10478248, 0.09695733, 0.08842269, 0.07787973, 0.06083218, 0.01430789],
  [0.08879462, 0.08325077, 0.07660706, 0.06737496, 0.05094656, 0.01275724]])

fig, ax = plt.subplots(1,  1, sharex=True)
sns.heatmap(out2,vmax=.13, vmin=0, square=True, cbar_kws={'label': 'Expected cases per home after 30 days'}, ax=ax)
ax.invert_yaxis()
ax.set_ylim(0,6)
ax.set_xticklabels(xticklabels)
ax.set_yticklabels(yticklabels)
plt.title("b=1")
ax.set(xlabel='Efficacy in staff', ylabel='Efficacy in residents')
fig.savefig('vacc_eff_b_is_1.png', bbox_inches='tight', dpi=300)

out3 = np.array([[0.12901551, 0.11826108, 0.10724358, 0.09480611, 0.07652076, 0.0162551],
  [0.12473139, 0.11451933, 0.10401136, 0.09206566, 0.07429877, 0.01594045],
  [0.11991809, 0.11032284, 0.1003783,  0.08894701, 0.07166744, 0.01554948],
  [0.11405839, 0.1052204,  0.09594204, 0.08507033, 0.06823213, 0.01501743],
  [0.10568874, 0.09793046, 0.08955722, 0.07935792, 0.0629105,  0.01418039],
  [0.087977,   0.08246125, 0.07589805, 0.06693078, 0.05128922, 0.01248648]])

fig, ax = plt.subplots(1,  1, sharex=True)
sns.heatmap(out3,vmax=.13, vmin=0, square=True, cbar_kws={'label': 'Expected cases per home after 30 days'}, ax=ax)
ax.invert_yaxis()
ax.set_ylim(0,6)
ax.set_xticklabels(xticklabels)
ax.set_yticklabels(yticklabels)
plt.title("b=10")
ax.set(xlabel='Efficacy in staff', ylabel='Efficacy in residents')
fig.savefig('vacc_eff_b_is_10.png', bbox_inches='tight', dpi=300)
