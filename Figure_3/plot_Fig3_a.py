import numpy as np
import matplotlib as mpl
import scipy.linalg as lg

def figsize(scale):
    fig_width_pt = 246.0
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean / scale
    fig_size = [fig_width, fig_height]
    return fig_size

mpl.use('pgf')
pgf_with_custom_preamble = {
        "font.family": "sansserif",
        "font.serif": ['Computer Modern Roman'],
        "font.size": 5,
        "figure.figsize": figsize(0.57),
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

# evaluate the fidelity
def q_fidelity(h, r):
    # target density matrix
    rho = np.array(((1,0,0,1),(0,0,0,0),(0,0,0,0),(1,0,0,1))) * 0.5

    # define operators for mapping into POVM basis
    M0 = 0.5 * np.array(((1,0),(0,0)))
    M1 = 1./6. * np.array(((1, np.sqrt(2.)), (np.sqrt(2.), 2.)))
    M2 = 1./12. * np.array(((2., -np.sqrt(2.)-1j*np.sqrt(6.)),(-np.sqrt(2.)+1j*np.sqrt(6.), 4.)))
    M3 = 1./12. * np.array(((2., -np.sqrt(2.)+1j*np.sqrt(6.)),(-np.sqrt(2.)-1j*np.sqrt(6.), 4.)))
    M = np.array((M0, M1, M2, M3))
    T = np.zeros((16, 16), dtype=complex)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    m = 4 * i + j
                    n = 4 * k + l
                    T[m, n] = np.trace(np.matmul(np.kron(M[i], M[j]), np.kron(M[k], M[l]))) 
    
    T_inv = np.linalg.inv(T)

    # load experimental and target data
    q = np.load('../Bell_State_Data/two_layer_sparse/learning_nvis4_nhid{}_{}reps_p.txt'.format(h, r))[:,:]
    p = np.load('../Targetstates/targetstates_Bell.txt')

    # array to store experimental density matrix
    sigma = np.zeros((4, 4, np.shape(q)[1]), dtype=complex)

    # array to store fidelities
    f = np.zeros(np.shape(q)[1], dtype=complex)
    for s in range(np.shape(q)[1]):              
        for i in range(16):
            for j in range(4):
                for k in range(4):
                    l = 4 * j + k
                    sigma[:,:,s] += T_inv[i, l] * np.kron(M[j], M[k]) * q[i,s]
 
        f[s] += np.trace(lg.sqrtm(np.matmul(lg.sqrtm(rho), np.matmul(sigma[:,:,s], lg.sqrtm(rho)))))
    
    return f

########### main file ##########
fig, ax1 = plt.subplots()
ax2 = inset_axes(ax1, width="52%", height="50%", loc=4, bbox_to_anchor=(.02,0.1,.95,1.0), bbox_transform=ax1.transAxes)

# indicate last 200 epochs since these are evaluated
ax1.fill_between(np.arange(1800, 2000), 0.785, 1.03, color='gray', alpha=0.4, linewidth=0)

# define colormap for different curves
norm = mpl.colors.Normalize(vmin=0, vmax=6)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.plasma)
cmap.set_array([])

# define number of hidden neurons that should be plotted
hiddens = [5, 10, 15, 20, 25, 30]
reps = 25

# plot DKL and fidelity
for i, h in enumerate(hiddens):

    # load DKL data
    dkl_out = np.load('../Bell_State_Data/two_layer_sparse/learning_nvis4_nhid{}_{}reps_dkl.txt'.format(h, reps))[:]
    
    dkl_mean = []
    dkl_std = []
    for k in range(50, 2000):
        dkl_mean = np.append(dkl_mean, np.mean(dkl_out[k-50:k]))
        dkl_std = np.append(dkl_std, np.std(dkl_out[k-50:k]))

    # plot running average and shade fluctuations in inset
    ax2.fill_between(np.arange(55, 2000), np.maximum(dkl_mean[5:] - np.absolute(dkl_std[5:]), 0.001), dkl_mean[5:] + np.absolute(dkl_std[5:]), color=cmap.to_rgba(i), alpha=0.2, linewidth=0)
    ax2.plot(np.arange(50, 2000), dkl_mean, linewidth=0.3, c=cmap.to_rgba(i))

    # evaluate fidelity and plot in main figure
    fid = q_fidelity(h, reps) # this is a complex number, but the imaginary part is 0
    ax1.plot(np.real(fid), linewidth=0.3, c=cmap.to_rgba(i))

    
############# Set up figure #############
ax2.set_yscale('log')
ax2.set_xlabel('Training Epochs', labelpad=0.7)
ax2.set_ylabel(r'$D_{KL}$', labelpad=0.7)
ax1.set_xlabel('Training Epochs', labelpad=0.7)
ax1.set_ylabel('Fidelity', labelpad=0.7)

ax1.set_xticks([0, 1000, 2000])
ax2.set_xticks([0, 2000])
ax2.set_yticks([1, 0.1, 0.01, 0.001])

ax2.set_ylim(bottom=0.001, top=0.3)
ax1.set_xlim(0, 2250)

ax1.tick_params(axis="x", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax1.tick_params(axis="y", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax2.tick_params(axis="x", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax2.tick_params(axis="y", which="major", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax2.tick_params(axis="y", which="minor", left=False)

plt.savefig('Fig3_a.pdf', bbox_inches='tight')
