import numpy as np
import matplotlib as mpl

def figsize(scale):
    fig_width_pt = 510.0
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width_pt * inches_per_pt * golden_mean * 0.5
    fig_size = [fig_width, fig_height]
    return fig_size

mpl.use('pgf')
pgf_with_custom_preamble = {
        "font.family": "sansserif",
        "font.serif": ['Computer Modern Roman'],
        "font.size": 5,
        "figure.figsize": figsize(0.15),
        "text.usetex": True,
        "axes.linewidth": 0.1,
        "pgf.rcfonts": False,
        "pgf.preamble": [
            "\\usepackage[cm]{sfmath}",
            "\\usepackage{units}",
            ]
        }
mpl.rcParams.update(pgf_with_custom_preamble)

import matplotlib.pyplot as plt
from matplotlib import gridspec

# set up figure
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1,6], hspace=0.0)

ax = plt.subplot(gs[1, 0])
ax2 = plt.subplot(gs[0, 0], sharex=ax)


############## Spike times of all neurons #############

# prepare data
spikes = np.load('spike_times.txt') # load dataset
spikes = spikes[spikes[:,0] > 854e-6] # choose desired time frame
spikes = spikes[spikes[:,0] < 957e-6] # choose desired time frame
spikes_vis = spikes[spikes[:,1] < 4] # get visible spike times
spikes_hid = spikes[spikes[:,1] >= 4] # get hidden spike times

# set positions of neurons on y-axis 
spikes_vis[spikes_vis[:,1] == 0, 1] = 7.5 
spikes_vis[spikes_vis[:,1] == 1, 1] = 6.5
spikes_vis[spikes_vis[:,1] == 2, 1] = 5.5
spikes_vis[spikes_vis[:,1] == 3, 1] = 4.5
spikes_hid[spikes_hid[:,1] == 4, 1] = 3.5
spikes_hid[spikes_hid[:,1] == 5, 1] = 3
spikes_hid[spikes_hid[:,1] == 6, 1] = 2.5
spikes_hid[spikes_hid[:,1] == 7, 1] = 2
spikes_hid[spikes_hid[:,1] == 8, 1] = 1.5
spikes_hid[spikes_hid[:,1] == 9, 1] = 1
spikes_hid[spikes_hid[:,1] == 10, 1] = 0.5
spikes_hid[spikes_hid[:,1] == 11, 1] = 0.

# gray lines indicate every 5th readout
x_positions_gray = np.arange(854, 964, 10)
for x_pos in x_positions_gray:
    ax.vlines(x=x_pos, ymin=3.75, ymax=8.05, color='gray', alpha=0.2, linewidth=0.3)

# small black lines indicate each readout
x_positions_black = np.arange(854, 958, 2)
for x_pos in x_positions_black:
    ax.vlines(x=x_pos, ymin=7.95, ymax=8.05, color='black', linewidth=0.2)

# solid lines at neuron spike times
ax.vlines(spikes_vis[:,0]*1e6, ymin=spikes_vis[:,1] - 0.45, ymax=spikes_vis[:,1] + 0.45, color='orange', linewidth=0.5)
ax.vlines(spikes_hid[:,0]*1e6, ymin=spikes_hid[:,1] - 0.15, ymax=spikes_hid[:,1] + 0.15, color='green', linewidth=0.5)

# refractory times are about 1e-5 for all neurons 
taurefs = np.ones(np.shape(spikes_vis)[0]) * 1e-5
taurefs_h = np.ones(np.shape(spikes_hid)[0]) * 1e-5

# shading for refractory states after each spike
ax.fill([spikes_vis[:,0]*1e6, (spikes_vis[:,0] + taurefs[:])*1e6, (spikes_vis[:,0] + taurefs[:])*1e6, spikes_vis[:,0]*1e6], [spikes_vis[:,1] - 0.45, spikes_vis[:,1] - 0.45, spikes_vis[:,1] + 0.45, spikes_vis[:,1] + 0.45], color='orange', alpha=0.3, linewidth=0.1)

ax.fill([spikes_hid[:,0]*1e6, (spikes_hid[:,0] + taurefs_h[:])*1e6, (spikes_hid[:,0] + taurefs_h[:])*1e6, spikes_hid[:,0]*1e6], [spikes_hid[:,1] - 0.15, spikes_hid[:,1] - 0.15, spikes_hid[:,1] + 0.15, spikes_hid[:,1] + 0.15], color='green', alpha=0.3, linewidth=0.1)


######### potential evolution ###############

# prepare dataset
voltages = np.load('evolution_neuron0.npy') # load data
voltages = voltages[voltages[:,0] > 854e-6] # choose desired time frame
voltages = voltages[voltages[:,0] < 957e-6]

# neurons spike when threshold is crossed, so threshold = maximum potential
threshold = np.max(voltages)

# plot time evolution
ax2.plot(voltages[:,0]*1e6, voltages[:,1], linewidth=0.3)


############ finalize plot ###############
ax.set_xticks([854, 904, 954])
ax.set_xticklabels(['0','50','100'])

ax.set_xlim(852, 957)
ax2.set_xlim(852, 957)

ax.set_yticks([0,1,2,3,4.5,5.5,6.5,7.5])
ax.set_yticklabels(['12','10','8','6','4','3','2','1'])
ax.set_ylim(-0.2, 8)
ax2.set_yticks([240,340])

ax.set_xlabel(r'Time $\left[\mu s\right]$', labelpad=0.7)
ax.set_ylabel(r'Neuron ID', labelpad=1.5)
ax2.set_ylabel(r'Voltage 1', labelpad=1.5)

ax.tick_params(axis="x", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax2.tick_params(axis="x", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax2.tick_params(axis="y", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax.tick_params(axis="y", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)

plt.setp(ax2.get_xticklabels(), visible=False)

plt.savefig('Fig1_c.pdf', bbox_inches='tight')
