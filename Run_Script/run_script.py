import hxsampler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import pylogging
from collections import Counter
import numba

pylogging.default_config(
        level=pylogging.LogLevel.TRACE,
        fname="",
        print_location=False,
        color=True,
        )
log = pylogging.get("sample")

################# Function definitions ################

# training function to update weights and biases
@numba.jit(nopython=True)
def update_weights(samples, n_hid, n, power_array, p_target, p_train):
    b = np.zeros(n)
    w = np.zeros((n, n))
    norm = 0.

    p_ratio = np.zeros(2 **(n - n_hid))
    for i in range(2**(n - n_hid)):
        if p_train[i] != 0:
            p_ratio[i] = 1. - p_target[i] / p_train[i]
        else:
            p_ratio[i] = 0.

    for i in range(np.shape(samples)[0]):
        vis_state = int(np.dot(samples[i,:], power_array[:]) // (2**n_hid))

        b[:] += - p_ratio[vis_state] * samples[i,:]
        w[:,:] += - p_ratio[vis_state] * np.outer(samples[i, :], samples[i, :])

    return (b, w)

# evaluate the state probabilities from sampled configurations
@numba.jit(nopython=True)
def calc_samps(samples, n_vis, n_hid, power_array):
    p = np.zeros(2 ** n_vis)

    for i in range(np.shape(samples)[0]):
        state_vals_vis = int(np.dot(samples[i, :], power_array[:]) // 2**n_hid)
        p[state_vals_vis] += 1.

    p /= np.shape(samples)[0]
    return p

# evaluate the DKL
@numba.jit(nopython=True)
def calc_dkl(samples, n_vis, p_target, p_train):
    dkl = 0.
    
    for vis_state in range(2**n_vis):
        if p_train[vis_state] != 0:
            dkl += p_target[vis_state] * np.log(p_target[vis_state] / p_train[vis_state])

    return dkl

# transform decimal to binary numbers
def dec_to_bin(x, n):
    
    binary = np.binary_repr(x, width = n)
    ret = np.array([int(y) for y in list(binary)])

    return ret

# transform binary to decimal numbers
def bin_to_dec(x, n):
    decimal = 0

    for i in range(n):
        decimal += x[i] * 2 ** (n - 1 - i)

    return decimal



################# Main function ##################

######### Set Network and Hardware Parameters ##########
n_vis = 4  # Number of visible Neurons
n_hid = 8  # Number of hidden Neurons

power_array = np.array([2 ** (n_vis + n_hid - 1 - i) for i in range(n_vis + n_hid)], dtype = float) # array enumerates states

noise_type = "On-chip" # Noise type: On-chip or Poisson
divider = np.load('calibration/divider.txt') # divider from chip calibration measurement

taurefs = np.zeros(n_vis + n_hid)  # Refractory times for the neurons
taurefs_read = np.load('calibration/taurefs.txt')  # from chip calibration measurement
taurefs[:] = taurefs_read[:n_vis+n_hid] / divider

duration = 1.0e-2  # Duration of measurement
repetitions = 25   # Measurement repetitions  
dt = 2e-6  # Time step in which states are read when sampling
epochs = 2000 # Epochs of training

p_target = np.load('../Data/Targetstates/targetstates_Bell.txt') # Load target distribution - This is a fixed distribution, no samples!

alpha_w = np.load('calibration/alpha_w.txt') / divider # Width of weight activation function from chip calibration measurement
alpha_b = np.load('calibration/alpha_b.txt') / divider # Width of bias activation function from chip calibration measurement
eta = 1.  # RBM learning rate


################# Setup Network ###########
n = n_vis + n_hid # Number of neurons

# Initialize connecting weights
w = np.random.randint(100, size = (n, n)) - np.full((n, n), 50)
w = 0.5 * (w + w.T)
w[n_vis:, n_vis:] = np.zeros((n_hid, n_hid))
w[:n_vis, :n_vis] = np.zeros((n_vis, n_vis))
np.fill_diagonal(w, 0.)

# Initialize biases
b = np.full(n, -50.) + np.random.randint(100, size=n)
b_read = np.load('calibration/biases.txt')
b[:n] += b_read[:n] / divider + 50.
b[:2] -= 20.
b[2:4] += 20.

# Choose between different noise setups (usually on-chip) and set parameters
if noise_type == "Poisson":
    noise_rate = 550000.
    noise_weight = 16.
    noise_multiplier = 1.
elif noise_type == "On-chip":
    noise_rate = 74000.
    noise_weight = 25.
    noise_multiplier = 4.

# Set up sampling on neuromorphic chip
hxsampler = hxsampler.HXSampler(n, np.copy(w), np.copy(b), noiseweight = noise_weight, noiserate = noise_rate, noisemultiplier = noise_multiplier, noisetype = noise_type)

# Calibrate if no calibration data given
if taurefs.all() == 0 and alpha_b == 0:
    hxsampler.measure_activation_functions(duration = duration, stepsize = 50)
    taurefs = hxsampler.measured_taurefs
    alpha_b = 0
    for i in range(n):
        alpha_b += np.sum(hxsampler.activation['fit'][i][1])
        b[i] = hxsampler.activation['fit'][i][0]
    alpha_b /= n
    print("taurefs={}, alpha_b={}, bstart={}".format(taurefs, alpha_b, b))
else:
    hxsampler.measured_taurefs = taurefs
    
if alpha_w == 0:
    hxsampler.measure_weight_activation(0, 1, duration = duration, stepsize = 5)
    alpha_w = hxsampler.weight_activation['fit'][1]

# Define learning rates for weights and biases from chip calibration measurement
eta_b = eta
eta_w = eta_b * alpha_w / alpha_b


# Define Some Arrays for Output 
# Output state probabilities, biases, weights, and DKL 
p_train = np.zeros((2 ** n_vis, epochs + 1))
b_train = np.zeros((n, epochs + 1))
w_train = np.zeros((n, n, epochs + 1))
dkl_train = np.zeros(epochs + 1)

# Store initial variables for output 
samples = hxsampler.get_samples(duration = duration, dt = dt, set_parameters=True, readout_neuron=0)

# Run sampling experiments
for reps in range(repetitions):
    samples = np.append(samples, hxsampler.get_samples(duration = duration, dt = dt), axis = 0)

# extract probabilities, biases, weights, and DKL
p_train[:, 0] = calc_samps(samples, n_vis, n_hid, power_array)
b_train[:,0] = b[:]
w_train[:,:,0] = w[:,:]
dkl_train[0] = calc_dkl(samples, n_vis, p_target, p_train[:, 0])


# Arrays to store updates
b_upd_mod_saver = np.zeros(n)
w_upd_mod_saver = np.zeros((n, n))
m_b = np.zeros(n)
m_w = np.zeros((n, n))
v_b = np.zeros(n)
v_w = np.zeros((n, n))

############### Start Training Procedure ##############
for e in range(epochs):

    # Update learning rates
    eta_b_u = max(eta_b * np.exp(- 0.001 * e), 0.001)
    eta_w_u = eta_b_u * alpha_w / alpha_b

    # Draw samples
    samples = hxsampler.get_samples(duration = duration, dt = dt, set_parameters=True)

    for reps in range(repetitions):
        samples = np.append(samples, hxsampler.get_samples(duration = duration, dt = dt), axis = 0)

    # get state probabilities
    p_train[:, e + 1] = calc_samps(samples, n_vis, n_hid, power_array)

    # get updates for weights and biases from training
    (b_upd_mod, w_upd_mod) = update_weights(samples, n_hid, n, power_array, p_target, p_train[:,e + 1])

    # get DKL
    dkl_train[e + 1] = calc_dkl(samples, n_vis, p_target, p_train[:, e + 1])

    b_train[:,e + 1] = b[:]
    w_train[:,:,e + 1] = w[:,:]
        
    ###### Calculate updates for weights and biases to write on hardware using Adam optimizer #######

    w_upd_mod[n_vis:, n_vis:] = np.zeros((n_hid, n_hid))
    w_upd_mod[:n_vis, :n_vis] = np.zeros((n_vis, n_vis))
    np.fill_diagonal(w_upd_mod, 0.)

    m_b = 0.9 * m_b + 0.1 * b_upd_mod / np.shape(samples)[0]
    m_w = 0.9 * m_w + 0.1 * w_upd_mod / np.shape(samples)[0]
    v_b = 0.999 * v_b + 0.001 * np.multiply(b_upd_mod, b_upd_mod) / np.shape(samples)[0]**2
    v_w = 0.999 * v_w + 0.001 * np.multiply(w_upd_mod, w_upd_mod) / np.shape(samples)[0]**2
    mh_b = m_b / (1. - 0.9**(e + 1))
    mh_w = m_w / (1. - 0.9**(e + 1))
    vh_b = v_b / (1. - 0.999**(e + 1))
    vh_w = v_w / (1. - 0.999**(e + 1))

    b_new = np.copy(b + eta_b_u * np.divide(mh_b, np.sqrt(vh_b) + 1e-8))
    w_new = np.copy(w + eta_w_u * np.divide(mh_w, np.sqrt(vh_w) + 1e-8))

    # update weights and biases
    for i in range(n):
        if b_new[i] <= 1023 and b_new[i] >= 0:
            hxsampler.logical_bias[i] = int(b_new[i])
            b[i] = np.copy(b_new[i])
        for j in range(n):
            if w_new[i, j] <= 63 and w_new[i,j] >= -63:
                hxsampler.logical_weight[i, j] = int(w_new[i, j])
                w[i, j] = np.copy(w_new[i, j])

    b_upd_mod_saver = b_upd_mod / np.shape(samples)[0]
    w_upd_mod_saver = w_upd_mod / np.shape(samples)[0]

    ######### Output state distributions, weights, biases, and DKL ########
    
    with open('learning_nvis{}_nhid{}_{}reps_p.txt'.format(n_vis, n_hid, repetitions), 'wb') as f:
        np.save(f, p_train)
    with open('learning_nvis{}_nhid{}_{}reps_b.txt'.format(n_vis, n_hid, repetitions), 'wb') as f:
        np.save(f, b_train)
    with open('learning_nvis{}_nhid{}_{}reps_w.txt'.format(n_vis, n_hid, repetitions), 'wb') as f:
        np.save(f, w_train)
    with open('learning_nvis{}_nhid{}_{}reps_dkl.txt'.format(n_vis, n_hid, repetitions), 'wb') as f:
        np.save(f, dkl_train)
