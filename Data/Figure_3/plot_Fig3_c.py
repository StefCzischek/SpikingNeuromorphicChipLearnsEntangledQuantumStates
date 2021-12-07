import numpy as np
import matplotlib as mpl
import scipy.linalg as lg

def figsize(scale):
    fig_width_pt = 246.0
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean / scale * 0.5
    fig_size = [fig_width, fig_height]
    return fig_size

mpl.use('pgf')
pgf_with_custom_preamble = {
        "font.family": "sansserif",
        "font.serif": ['Computer Modern Roman'],
        "font.size": 5,
        "figure.figsize": figsize(0.5),
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

# define arrays of numbers of hidden neurons, they differ for 2/3/4 qubits
hiddens_2 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58]
hiddens_3 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 56]
hiddens_4 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
reps = 25

# evaluate the quantum fidelity
def q_fidelity(h, h2, h3, r):

    # define the exact density matrices for 2/3/4 qubits
    rho_2 = np.zeros((4,4))
    rho_2[0,0] = 0.5
    rho_2[0,-1] = 0.5
    rho_2[-1,0] = 0.5
    rho_2[-1,-1] = 0.5

    rho_3 = np.zeros((8,8))
    rho_3[0,0] = 0.5
    rho_3[0,-1] = 0.5
    rho_3[-1,0] = 0.5
    rho_3[-1,-1] = 0.5

    rho_4 = np.zeros((16, 16))
    rho_4[0,0] = 0.5
    rho_4[0,-1] = 0.5
    rho_4[-1,0] = 0.5
    rho_4[-1,-1] = 0.5

    # define operators for the transformation into the POVM basis for 2/3/4 qubits
    M0 = 0.5 * np.array(((1,0),(0,0)))
    M1 = 1./6. * np.array(((1, np.sqrt(2.)), (np.sqrt(2.), 2.)))
    M2 = 1./12. * np.array(((2., -np.sqrt(2.)-1j*np.sqrt(6.)),(-np.sqrt(2.)+1j*np.sqrt(6.), 4.)))
    M3 = 1./12. * np.array(((2., -np.sqrt(2.)+1j*np.sqrt(6.)),(-np.sqrt(2.)-1j*np.sqrt(6.), 4.)))
    M = np.array((M0, M1, M2, M3))
    T_2 = np.zeros((16, 16), dtype=complex)
    T_3 = np.zeros((64, 64), dtype=complex)
    T_4 = np.zeros((256, 256), dtype=complex)

    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    m = 4 * i + j
                    n = 4 * k + l
                    T_2[m, n] = np.trace(np.matmul(np.kron(M[i], M[j]), np.kron(M[k], M[l])))
                    for s in range(4):
                        for t in range(4):
                            m = 16 * s + 4 * i + j
                            n = 16 * t + 4 * k + l
                            T_3[m, n] = np.trace(np.matmul(np.kron(M[s], np.kron(M[i], M[j])), np.kron(M[t], np.kron(M[k], M[l]))))  
                            for x in range(4):
                                for y in range(4):
                                    m = 64 * x + 16 * s + 4 * i + j
                                    n = 64 * y + 16 * t + 4 * k + l
                                    T_4[m, n] = np.trace(np.matmul(np.kron(M[x], np.kron(M[s], np.kron(M[i], M[j]))), np.kron(M[y], np.kron(M[t], np.kron(M[k], M[l])))))

    T_inv_2 = np.real(np.linalg.inv(T_2))
    T_inv_3 = np.real(np.linalg.inv(T_3))
    T_inv_4 = np.real(np.linalg.inv(T_4))

    # load experimental data
    q_2 = np.load('../Bell_State_Data/two_layer_sparse/learning_nvis4_nhid{}_{}reps_p.txt'.format(h_2, r))[:,-200:]
    q_3 = np.load('../GHZ3_State_Data/learning_nvis6_nhid{}_GHZ_CP_{}reps_p.txt'.format(h_3, r))[:,-200:]
    q_4 = np.load('../GHZ4_State_Data/learning_nvis8_nhid{}_GHZ_CP_45reps_p.txt'.format(h_4))[:,-200:]

    # load exact data
    p_2 = np.load('../Targetstates/targetstates_Bell.txt')
    p_3 = np.load('../Targetstates/targetstates_GHZ3.txt')
    p_4 = np.load('../Targetstates/targetstates_GHZ4.txt')
    f_2 = np.zeros(np.shape(q_2)[1], dtype=complex)
    f_3 = np.zeros(np.shape(q_3)[1], dtype=complex)
    f_4 = np.zeros(np.shape(q_4)[1], dtype=complex)

    # evaluate density matrices and fidelities
    for s in range(np.shape(q_2)[1]):
        sigma_2 = np.zeros((4, 4), dtype=complex)
        sigma_3 = np.zeros((8, 8), dtype=complex)
        sigma_4 = np.zeros((16, 16), dtype=complex)
        for j in range(4):
            for k in range(4):
                l = 4 * j + k
                for i in range(16):
                    sigma_2[:,:] += T_inv_2[i, l] * np.kron(M[j], M[k]) * q_2[i, s] 
                for t in range(4):
                    l = 16 * t + 4 * j + k
                    for i in range(64):
                        sigma_3[:,:] += T_inv_3[i, l] * np.kron(M[t], np.kron(M[j], M[k])) * q_3[i, s] 
                    for x in range(4):
                        l = 64 * x + 16 * t + 4 * j + k
                        for i in range(256):
                            sigma_4[:,:] += T_inv_4[i, l] * np.kron(M[x], np.kron(M[t], np.kron(M[j], M[k]))) * q_4[i, s]
    
        f_2[s] = np.sqrt(0.5 * (sigma_2[0,0] + sigma_2[0,-1] + sigma_2[-1,0] + sigma_2[-1,-1]))
        f_3[s] = np.sqrt(0.5 * (sigma_3[0,0] + sigma_3[0,-1] + sigma_3[-1,0] + sigma_3[-1,-1]))
        f_4[s] = np.sqrt(0.5 * (sigma_4[0,0] + sigma_4[0,-1] + sigma_4[-1,0] + sigma_4[-1,-1]))
    
    return (f_2, f_3, f_4)


fig, ax1 = plt.subplots()

norm = mpl.colors.Normalize(vmin=0, vmax=6)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.plasma)
cmap.set_array([])

# set up arrays to store fidelities and standard deviations
fid_2 = []
fid_2_std = []
fid_3 = []
fid_3_std = []
fid_4 = []
fid_4_std = []

# run over all hidden numbers
for i, h_2 in enumerate(hiddens_2):

    # they might differ for differen numbers of qubits
    h_3 = hiddens_3[i]
    h_4 = hiddens_4[i]

    # evaluate the fidelity
    (fid_2_save, fid_3_save, fid_4_save) = q_fidelity(h_2, h_3, h_4, reps)

    # append mean and standard deviations of outcomes
    fid_2 = np.append(fid_2, np.mean(fid_2_save))
    fid_2_std = np.append(fid_2_std, np.std(fid_2_save))
    fid_3 = np.append(fid_3, np.mean(fid_3_save))
    fid_3_std = np.append(fid_3_std, np.std(fid_3_save))
    fid_4 = np.append(fid_4, np.mean(fid_4_save))
    fid_4_std = np.append(fid_4_std, np.std(fid_4_save))


########## Set up figure ############

ax1.hlines(y=1./np.sqrt(2.), xmin=5, xmax=60, color='gray', linestyle='--', linewidth=0.7)

ax1.errorbar(hiddens_2, fid_2, yerr=fid_2_std, linestyle='None', marker='o', markerfacecolor='none', markeredgecolor='C0', markersize=2., linewidth=0.7, markeredgewidth=0.5, color='C0')
ax1.plot(hiddens, fid_2, 'o', markerfacecolor='C0', markeredgecolor='none', markersize=2., linewidth=0.3, alpha=0.5)

ax1.errorbar(hiddens_3, fid_3, yerr=fid4_std, linestyle='None', marker='d', markerfacecolor='none', markeredgecolor='C1', markersize=2., linewidth=0.7, markeredgewidth=0.5, color='C1')
ax1.plot(hiddens_3, fid_3, 'd', markerfacecolor='C1', markeredgecolor='none', markersize=2., linewidth=0.3, alpha=0.5)

ax1.errorbar(hiddens_4, fid_4, yerr=fid4_std, linestyle='None', marker='H', markerfacecolor='none', markeredgecolor='C2', markersize=2., linewidth=0.7, markeredgewidth=0.5, color='C2')
ax1.plot(hiddens_4, fid_4, 'H', markerfacecolor='C2', markeredgecolor='none', markersize=2., linewidth=0.3, alpha=0.5)

ax1.set_xlabel(r'$\#$ Hidden Neurons $M$', labelpad=0.7)
ax1.set_ylabel('Fidelity', labelpad=0.7)

ax1.tick_params(axis="x", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax1.tick_params(axis="y", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)

plt.savefig('Fid_large_new.pdf', bbox_inches='tight')
