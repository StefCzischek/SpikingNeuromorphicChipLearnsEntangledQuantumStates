import numpy as np
import matplotlib as mpl
import scipy.linalg as lg

def figsize(scale):
    fig_width_pt = 246.0
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean / scale * 0.3
    fig_size = [fig_width, fig_height]
    return fig_size

mpl.use('pgf')
pgf_with_custom_preamble = {
        "font.family": "sansserif",
        "font.serif": ['Computer Modern Roman'],
        "font.size": 5,
        "figure.figsize": figsize(0.45),
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

# evaluate probability distribution
def prob_calc(samples, n_hid, power_array):
    p = np.zeros(2 ** 4)
    state_vals_vis = int(np.dot(samples[:], power_array[:]) // 2**n_hid)
    p[state_vals_vis] += 1

    return p

# function to evaluate DKL
def dkl(p, p_t):
    dkl = 0.
    for i in range(16):
        if p[i] > 0:
            dkl += p_t[i] * np.log(p_t[i] / p[i])

    return dkl

# create array with different states as binary inputs
power_array = np.array([2 ** (23 - i) for i in range(24)], dtype = float)

# set number of hidden neurons and reps
h = 20
r = 25

# array to store evolution of DKL
DKL_evol = []

# load data
samps = np.load('samples.txt', allow_pickle=True)

# array to store measured probabilities
p = np.zeros((16, np.shape(samps)[0]))

# target probability
p_t = np.load('../Targetstates/targetstates_Bell.txt')

# array to store probability distribution
full_p = np.zeros(16)

# loop over the number of samples, evaluate DKL every time
for i in range(1, np.shape(samps)[0]+1):

    # get probability distribution underlying added configuration
    p_in = prob_calc(samps[i-1], 20, power_array)

    # add this distribution to the one from previous samples
    full_p += p_in

    # only output result of every 50 steps
    if (i-1)%50 == 0:

        # evaluate DKL
        dkl_val = dkl(full_p / i, p_t)

        # append DKL
        DKL_evol = np.append(DKL_evol, dkl_val)


########## Set up figure #########

fig, ax1 = plt.subplots()

x = np.arange(1, np.shape(samps)[0]+1, 50)
ax1.plot(x, 6. * DKL_evol[20] / x * 10**(2), color='gray', linestyle='--', linewidth=0.7)

ax1.plot(np.arange(0, np.shape(samps)[0], 50), DKL_evol, linewidth=0.7)

ax1.set_xlabel(r'\# Samples $S$', labelpad=0.7)
ax1.set_ylabel(r'$D_{\mathrm{KL}}\left(p^*\|p\right)$', labelpad=0.7)

ax1.set_xscale('log')

ax1.set_xlim(500, 255000)
ax1.set_ylim(0.007, 0.25)
ax1.set_yscale('log')

ax1.tick_params(axis="x", which="major", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax1.tick_params(axis="x", which="minor", bottom=False)
ax1.tick_params(axis="y", direction="in", length=1.5, width=0.3, labelsize=5, pad=1.)
ax1.tick_params(axis="y", which="minor", left=False)

plt.savefig('Fig3_b.pdf', bbox_inches='tight')

