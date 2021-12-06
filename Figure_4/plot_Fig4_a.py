import numpy as np
import matplotlib as mpl
import scipy.linalg as lg

def figsize(scale):
    fig_width_pt = 246.0
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean / scale * 0.4
    fig_size = [fig_width, fig_height]
    return fig_size

mpl.use('pgf')
pgf_with_custom_preamble = {
        "font.family": "sansserif",
        "font.serif": ['Computer Modern Roman'],
        "font.size": 5,
        "figure.figsize": figsize(1.1),
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

# evaluate the quantum fidelity
def q_fidelity(h, r):

    # define exact density matrix
    rho = np.zeros((4,4))
    rho[0,0] = 0.5
    rho[0,-1] = 0.5
    rho[-1,0] = 0.5
    rho[-1,-1] = 0.5

    # define operators for mapping to POVMs
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

    # load experimental data with sparse and full connections
    q_sparse = np.load('../Bell_State_Data/two_layer_sparse/learning_nvis4_nhid{}_{}reps_p.txt'.format(h, r))[:,-200:]
    q_full = np.load('../Bell_State_Data/two_layer_dense/learning_nvis4_nhid{}_{}reps_p.txt'.format(h, r))[:,-200:]

    # load exact data
    p = np.load('../Targetstates/targetstates_Bell.txt')

    # calculate experimental density matricies and fidelities
    sigma_sparse = np.zeros((4, 4, np.shape(q_sparse)[1]), dtype=complex)
    sigma_full = np.zeros((4, 4, np.shape(q_full)[1]), dtype=complex)
    f_sparse = np.zeros(np.shape(q_sparse)[1], dtype=complex)
    f_full = np.zeros(np.shape(q_full)[1], dtype=complex)
    for s in range(np.shape(q_sparse)[1]):              
        for j in range(4):
            for k in range(4):
                l = 4 * j + k
                for i in range(16):
                    sigma_sparse[:,:,s] += T_inv[i, l] * np.kron(M[j], M[k]) * q_sparse[i,s]
                    sigma_full[:,:,s] += T_inv[i, l] * np.kron(M[j], M[k]) * q_full[i,s]            
 
        f_sparse[s] += np.trace(lg.sqrtm(np.matmul(lg.sqrtm(rho), np.matmul(sigma_sparse[:,:,s], lg.sqrtm(rho)))))
        f_full[s] += np.trace(lg.sqrtm(np.matmul(lg.sqrtm(rho), np.matmul(sigma_full[:,:,s], lg.sqrtm(rho)))))

    return (f_sparse, f_full)


################ Main function ################

fig, ax1 = plt.subplots()

norm = mpl.colors.Normalize(vmin=0, vmax=6)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.plasma)
cmap.set_array([])

colors = ['C0', 'C1', 'C2']
shapes = ['o', 'd', 'H']

# hidden neuron numbers and reps
hiddens = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
reps = 25

# create arrays to fill with mean and standard deviation of fidelities
fid_sparse = []
fid_sparse_std = []
fid_full = []
fid_full_std = []

# loop over number of hidden neurons
for i, h in enumerate(hiddens):

    # evaluate fidelities
    (fid_save_sparse, fid_save_full) = np.real(q_fidelity(h, reps))

    # append results 
    fid_sparse = np.append(fid_sparse, np.mean(fid_save_sparse))
    fid_sparse_std = np.append(fid_sparse_std, np.std(fid_save_sparse))
    fid_full = np.append(fid_full, np.mean(fid_save_full))
    fid_full_std = np.append(fid_full_std, np.std(fid_save_full))

# plot fidelities
ax1.errorbar(hiddens, fid_sparse, yerr=fid_sparse_std, linestyle='None', marker=shapes[0], markerfacecolor='none', markeredgecolor=colors[0], markersize=3., linewidth=0.7, markeredgewidth=0.5, color=colors[0], zorder=0)
ax1.plot(hiddens, fid_sparse, marker=shapes[0], markerfacecolor=colors[0], markeredgecolor='none', markersize=3., linewidth=0.3, alpha=0.5, linestyle='None', zorder=0)

ax1.errorbar(hiddens, fid_full, yerr=fid_full_std, linestyle='None', marker=shapes[1], markerfacecolor='none', markeredgecolor=colors[1], markersize=3., linewidth=0.7, markeredgewidth=0.5, color=colors[1], zorder=1)
ax1.plot(hiddens, fid_full, marker=shapes[1], markerfacecolor=colors[1], markeredgecolor='none', markersize=3., linewidth=0.3, alpha=0.5, linestyle='None', zorder=1)

############### Set up figure ############
ax1.set_xlabel(r'$\#$ Hidden Neurons $M$', labelpad=0.7)
ax1.set_ylabel('Fidelity', labelpad=0.7)

ax1.tick_params(axis="x", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax1.tick_params(axis="y", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)

plt.savefig('Fig4_a.pdf', bbox_inches='tight')
