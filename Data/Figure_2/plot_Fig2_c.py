import matplotlib as mpl
import os
import numpy as np

def figsize(scale):
    fig_width_pt = 510.
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
        "figure.figsize": figsize(0.25),
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

# define arrays to store mean and standard deviation of Bell observable for different r
Bell_plot = []
Bell_plot_std = []

# loop over all r values, pW is needed for the file names
pW = ['0', '01', '02', '03', '04', '05', '06', '07', '08', '09', '1']

for q in pW: 

    # load data
    p = np.load('../Werner_State_Data/learning_nvis4_nhid20_25reps_{}_p.txt'.format(q))[:, -200:]

    # array to append Bell observables
    Bell = []

    for i in range(np.shape(p)[1]):
        # evaluate correlation in x- and z-basis, which corresponds to Theta=pi/4
        corrx = (8. * p[5,i] - 4. * (p[6,i] + p[7,i] + p[9,i] + p[13,i]) + 2. * (p[10,i] + p[11,i] + p[14,i] + p[15,i]))
        corrz = (9. * p[0,i] - 3. * (p[1,i] + p[2,i] + p[3,i] + p[4,i] + p[8,i] + p[12,i]) + p[5,i] + p[6,i] + p[7,i] + p[9,i] + p[10,i] + p[11,i] + p[13,i] + p[14,i] + p[15,i])

        # append Bell observable
        Bell = np.append(Bell, np.sqrt(2.) * (corrx + corrz))
        
    Bell_plot = np.append(Bell_plot, np.mean(Bell))
    Bell_plot_std = np.append(Bell_plot_std, np.std(Bell))

######## Set up figure ##########
plt.plot(np.arange(0, 1.1, 0.1), np.sqrt(2.) * np.arange(2., -0.1, -0.2), linewidth=0.7, color='k')

x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]

# define arrays for blue data points and plot them
Bell2 = np.array([Bell_plot[0], Bell_plot[1], Bell_plot[2], Bell_plot[4], Bell_plot[5], Bell_plot[6], Bell_plot[7], Bell_plot[8], Bell_plot[9]])
Bell2_std = np.array([Bell_plot_std[0], Bell_plot_std[1], Bell_plot_std[2], Bell_plot_std[4], Bell_plot_std[5], Bell_plot_std[6], Bell_plot_std[7], Bell_plot_std[8], Bell_plot_std[9]])

l1 = plt.errorbar(x, Bell2[::-1], yerr=Bell2_std[::-1], marker='o', markerfacecolor='none', markeredgecolor='C0', markersize=2., linestyle='None', linewidth=0.7, markeredgewidth=0.5, color='C0')
l1_1 = plt.plot(x, Bell2[::-1], 'o', markerfacecolor='C0', markeredgecolor='none', markersize=2., linewidth=0.3, alpha=0.5)

# plot red data point at r=1
plt.errorbar(0, Bell_plot[-1], yerr=Bell_plot_std[-1], marker='o', markerfacecolor='none', markeredgecolor='C3', markersize=2., linestyle='None', linewidth=0.7, markeredgewidth=0.5, color='C3')
plt.plot(0, Bell_plot[-1], 'o', markerfacecolor='C3', markeredgecolor='none', markersize=2., linewidth=0.3, alpha=0.5)

# plot green data point at r=0.3
plt.errorbar(0.7, Bell_plot[3], yerr=Bell_plot_std[3], marker='o', markerfacecolor='none', markeredgecolor='C2', markersize=2., linestyle='None', linewidth=0.7, markeredgewidth=0.5, color='C2')
plt.plot(0.7, Bell_plot[3], 'o', markerfacecolor='C2', markeredgecolor='none', markersize=2., linewidth=0.3, alpha=0.5)

plt.hlines(y = 2, xmin = -0.1, xmax=1.1, linestyle='dashed', linewidth=0.5, color='gray')

plt.xlabel(r'$1-p$', labelpad=0.7)
plt.ylabel(r'$\mathcal{B}\left(\pi/4\right)$', labelpad=0.7)

plt.xticks([0,0.2,0.4,0.6,0.8,1])

plt.xlim(-0.05, 1.05)

plt.tick_params(axis="x", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
plt.tick_params(axis="y", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)

plt.axvspan(xmin=2./3., xmax=1.05, facecolor='xkcd:apple green', alpha=0.15)
plt.axvspan(xmin=-0.05, xmax=2./3., facecolor='xkcd:chestnut', alpha=0.15)

plt.savefig('Fig2_c.pdf', bbox_inches='tight')
    
