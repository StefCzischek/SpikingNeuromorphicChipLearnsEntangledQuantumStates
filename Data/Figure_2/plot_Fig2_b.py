import matplotlib as mpl
import numpy as np

def figsize(scale):
    fig_width_pt = 510.
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0) - 1.0)/2.
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width_pt * inches_per_pt * golden_mean * 0.5
    fig_size = [fig_width, fig_height]
    return fig_size


mpl.use('pgf')
pgf_with_custom_preamble = {
        "font.family": "sansserif",
        "font.serif": ['Computer Modern Roman'],
        "font.size": 5,
        "figure.figsize": figsize(0.4),
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
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

# transform the measurement angle into the POVM basis
def Q_eval(theta):
    Q0 = 3. * np.cos(theta)
    Q1 = - np.cos(theta) + 2. * np.sqrt(2.) * np.sin(theta)
    Q2 = - np.cos(theta) - np.sqrt(2.) * np.sin(theta)
    Q3 = - np.cos(theta) - np.sqrt(2.) * np.sin(theta)
    Q = np.array([Q0, Q1, Q2, Q3])

    return Q

# evaluate the correlation expectation value in the POVM basis
def corr_exp(theta1, theta2, P):
    Q1 = Q_eval(theta1)
    Q2 = Q_eval(theta2)
    exp = 0.
    for i in range(4):
        for j in range(4):
            l = 4 * i + j
            exp += Q1[i] * Q2[j] * P[l]

    return exp

# evaluate the magnetization expectation value in the POVM basis
def mag_exp(theta, spin, P):
    exp = 0.
    Q = Q_eval(theta)
    for i in range(4):
        for j in range(4):
            l = 4 * i + j
            if spin == 0:
                exp += Q[i] * P[l]
            else:
                exp += Q[j] * P[l]

    return exp

# arrays to store Bell observables evaluated on exact and simulated data
Bell_res = [] # for the Bell state
Bell_res_sim = []
Bell_res_sim_std = []

Werner_res = [] # for the Werner state at r=0.3
Werner_res_sim = []
Werner_res_sim_std = []

# load data for Werner state at r=0.3
p2_w = np.load('../Werner_State_Data/learning_nvis4_nhid20_25reps_03_p.txt')[:,-200:] # experimental data
p_w = np.load('../Targetstates/targetstates_Werner03.txt') # exact target data

# load data for Bell state
p2_training = np.load('../Bell_State_Data/two_layer_sparse/learning_nvis4_nhid20_25reps_p.txt')[:,:] # experimental data entire training
p2 = np.copy(p2_training[:,-200:]) # experimental data converged
p = np.load('../Targetstates/targetstates_Bell.txt') # exact target data

# evaluate the Bell observable for Theta=pi/4 during the training progress
Bell_learn = corr_exp(0., np.pi/4., p2_training) - mag_exp(0., 0, p2_training) * mag_exp(np.pi/4., 1, p2_training)
Bell_learn += corr_exp(0., -np.pi/4., p2_training) - mag_exp(0., 0, p2_training) * mag_exp(-np.pi/4., 1, p2_training)
Bell_learn += corr_exp(np.pi / 2., np.pi/4., p2_training) - mag_exp(np.pi / 2., 0, p2_training) * mag_exp(np.pi/4., 1, p2_training)
Bell_learn -= (corr_exp(np.pi / 2., -np.pi/4., p2_training) - mag_exp(np.pi/2., 0, p2_training) * mag_exp(-np.pi/4., 1, p2_training))

# evaluate the Bell observable scanning Theta averaged over the last 200 training epochs
for i, theta in enumerate(np.arange(0, 2. * np.pi, 0.01)):

    # exact Bell state
    Bell = corr_exp(0., theta, p) - mag_exp(0., 0, p) * mag_exp(theta, 1, p)
    Bell += corr_exp(0., -theta, p) - mag_exp(0., 0, p) * mag_exp(-theta, 1, p)
    Bell += corr_exp(2. * theta, theta, p) - mag_exp(2. * theta, 0, p) * mag_exp(theta, 1, p)
    Bell -= (corr_exp(2. * theta, -theta, p) - mag_exp(2. * theta, 0, p) * mag_exp(-theta, 1, p))

    # exact Werner state at r=0.3
    Werner = corr_exp(0., theta, p_w) - mag_exp(0, 0, p_w) * mag_exp(theta, 1, p_w)
    Werner += corr_exp(0., -theta, p_w) - mag_exp(0., 0, p_w) * mag_exp(-theta, 1, p_w)
    Werner += corr_exp(2. * theta, theta, p_w) - mag_exp(2. * theta, 0, p_w) * mag_exp(theta, 1, p_w)
    Werner -= (corr_exp(2. * theta, -theta, p_w) - mag_exp(2. * theta, 0, p_w) * mag_exp(-theta, 1, p_w))
    
    # append results to arrays
    Bell_res = np.append(Bell_res, Bell)
    Werner_res = np.append(Werner_res, Werner)

    # only evaluate experimental data every 10 steps
    if i%10 == 0:

        # create arrays to collect observables of last 200 training epochs
        Bell_2 = []
        Werner_2 = []

        for i in range(np.shape(p2)[1]):

            # Bell state
            Bell_2_state = (corr_exp(0., theta, p2[:,i]) - mag_exp(0., 0, p2[:,i]) * mag_exp(theta, 1, p2[:,i]))
            Bell_2_state += (corr_exp(0., -theta, p2[:,i]) - mag_exp(0., 0, p2[:,i]) * mag_exp(-theta, 1, p2[:,i]))
            Bell_2_state += (corr_exp(2. * theta, theta, p2[:,i]) - mag_exp(2. * theta, 0, p2[:,i]) * mag_exp(theta, 1, p2[:,i]))
            Bell_2_state  -= (corr_exp(2. * theta, -theta, p2[:,i]) - mag_exp(2. * theta, 0, p2[:,i]) * mag_exp(-theta, 1, p2[:,i]))

            Bell_2 = np.append(Bell_2, Bell_2_state)
            
            # Werner state
            Werner_2_state = corr_exp(0., theta, p2_w[:,i]) - mag_exp(0., 0, p2_w[:,i]) * mag_exp(theta, 1, p2_w[:,i])
            Werner_2_state += corr_exp(0., -theta, p2_w[:,i]) - mag_exp(0., 0, p2_w[:,i]) * mag_exp(-theta, 1, p2_w[:,i])
            Werner_2_state += corr_exp(2. * theta, theta, p2_w[:,i]) - mag_exp(2. * theta, 0, p2_w[:,i]) * mag_exp(theta, 1, p2_w[:,i])
            Werner_2_state -= (corr_exp(2. * theta, -theta, p2_w[:,i]) - mag_exp(2. * theta, 0, p2_w[:,i]) * mag_exp(-theta, 1, p2_w[:,i]))

            Werner_2 = np.append(Werner_2, Werner_2_state)
            
        # store mean and standard deviation
        Bell_res_sim = np.append(Bell_res_sim, np.mean(Bell_2))
        Bell_res_sim_std = np.append(Bell_res_sim_std, np.std(Bell_2))

        Werner_res_sim = np.append(Werner_res_sim, np.mean(Werner_2))
        Werner_res_sim_std = np.append(Werner_res_sim_std, np.std(Werner_2))


########## Set up figure ##########
fig, ax1 = plt.subplots()
ax2 = inset_axes(ax1, width="37%", height="37%", loc=1, bbox_to_anchor=(-.325,-.06,1.045,1.05), bbox_transform=ax1.transAxes)

ax2.plot(Bell_learn, linewidth=0.5, color='C3', zorder=1)
ax2.hlines(y=2.*np.sqrt(2.), xmin=0, xmax=2000, color='k', linestyle='dashed', linewidth=0.7, zorder=2)
ax2.set_xlabel(r'Training Epochs', labelpad=0.7)
ax2.set_ylabel(r'$\mathcal{B}\left(\pi/4\right)$', labelpad=0.7)

ax1.vlines(x=np.pi/4., ymin=-3.2, ymax=3.2, color='C7', linestyle='dashed', linewidth=0.7)

l1 = ax1.plot(np.arange(0, 2. * np.pi, 0.01), Bell_res, linewidth=0.7, color='k')
l2 = ax1.errorbar(np.arange(0, 2. * np.pi, 0.1), Bell_res_sim, yerr=Bell_res_sim_std, marker='o', markerfacecolor='none', markeredgecolor='C3', color='C3', markersize=2., linestyle='None', linewidth=0.7, markeredgewidth=0.5)
l2_1 = ax1.plot(np.arange(0, 2. * np.pi, 0.1), Bell_res_sim, 'o', markerfacecolor='C3', markeredgecolor='none', markersize=2., linewidth=0.3, alpha=0.5)

ax1.plot(np.arange(0, 2. * np.pi, 0.01), Werner_res, linewidth=0.7, color='k')
ax1.errorbar(np.arange(0, 2. * np.pi, 0.1), Werner_res_sim, yerr=Werner_res_sim_std, marker='o', markerfacecolor='none', markeredgecolor='C2', markersize=2., linestyle='None', linewidth=0.7, markeredgewidth=0.5, color='C2')
ax1.plot(np.arange(0, 2. * np.pi, 0.1), Werner_res_sim, 'o', markerfacecolor='C2', markeredgecolor='none', markersize=2., linewidth=0.3, alpha=0.5)

ax1.set_xticks([0, np.pi/4., np.pi/2., 3.*np.pi/4., np.pi, 5.*np.pi/4., 3.*np.pi/2., 7.*np.pi/4., 2.*np.pi])
ax1.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$', r'$2\pi$'])

ax1.set_xlabel(r'$\Theta$', labelpad=0.7)
ax1.set_ylabel(r'$\mathcal{B}\left(\Theta\right)$', labelpad=0.7)

ax1.tick_params(axis="x", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax1.tick_params(axis="y", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax2.tick_params(axis="x", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax2.tick_params(axis="y", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)

ax1.axhspan(ymin=-3.3, ymax=-2, facecolor='gray', alpha=0.2)
ax1.axhspan(ymin=2, ymax=3.3, facecolor='gray', alpha=0.2)

ax1.set_ylim(-3.3, 3.3)

plt.savefig('Fig2_b.pdf', bbox_inches='tight')
       
