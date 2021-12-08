# Spiking Neuromorphic Chip Learns Entangled Quantum States

This repository contains the data and code used in the paper "Spiking neuromorphic chip learns entangled quantum states" by S. Czischek et al., [arXiv:2008.01039 [cs.ET]](https://arxiv.org/abs/2008.01039) (2020). 

The code is used to train the BrainScaleS-2 spiking neuromorphic chip to represent states of small quantum many-body systems and to draw samples of qubit configurations. The properties of the sampled configurations are evaluated via expectation values of different observables.

This repository contains an example script to train the neuromorphic chip to represent a given target state based on samples drawn from the hardware (`Run_Script`). It further contains the measured data and the scripts to create the plots used in the paper figures (`Data`).

## Run Script

The two scripts `bell_state.py` and `ghz_state.py` implement the training of a spiking neural network with an RBM-like structure to represent Bell and GHZ states.

If you want to execute it, you either need access to the BrainScaleS-2 hardware and the `hxsampler` module or replace `hxsampler.HXSampler` by another implementation of a sampler. The `HXSampler.get_samples` method generates a number of samples for a given hardware configuration and the member arrays `HXSampler.logical_bias` and `HXSampler.logical_weight` hold the current network configuration (bias vector and weight matrix). Since the set hardware parameters correspond to physical circuits their meaning in terms of the activity changes can be measured using the methods `HXSampler.measure_activation_functions` and `HXSampler.measure_weight_activation`. In case of a direct RBM implementation these can be omitted. The `hxsampler` module is currently not publicly available.

The parameters of the observed activation functions for all neurons of the neuromorphic chip can be found in `hardware_characteristics`. The folder contains the following files:
- `biases.txt`: central point of the activation function
- `alpha_b.txt`: width of the activation function as a function of the bias parameter
- `alpha_w.txt`: width of the activation function as a function of the connection strength of a synaptic input
- `divider.txt`: ratio between alpha_b and alpha_w
- `taurefs.txt`: measured refractory periods

There are a number of technical parameters that need to be set to get BrainScaleS-2 at an appropriate working point. The set of these parameters that has been extracted from calibration measurements and used for all experiments in the context of this work can be found in the folder `calibration`.

## Data
This folder contains all data that is used to create the plots in the paper figures. 

`Bell_State_Data` contains qubit configurations sampled from the neuromorphic hardware while training it to represent the two-qubit Bell state using either an RBM-structured network (`two_layer_sparse`), a fully connected two-layer network (`two_layer_dense`), or a three-layer network (`three_layer_sparse`). This data is used in Figs. 2(b), 3(a), 3(c), and 4.

`GHZ3_State_Data` contains that data received from training the neuromorphic hardware to represent a three-qubit GHZ state. Similarly, `GHZ4_State_Data` contains sampled configurations when representing a four-qubit GHZ state. This data is used in Fig. 3(c).

`Werner_State_Data` contains sampled qubit configurations when training the spiking hardware to represent a noisy Bell state (Werner state), with different amounts of noise, specified by the parameter *r* (see Fig. 2(c) in the paper). The amount of noise is denoted in the file name.

All folders with experimental data contain four files per measurement. The file names indicate the measurement setup, with number of visible neurons `nvis`, number of hidden neurons `nhid`, and number of measurement repetitions `reps`. For each setup the sampled configurations can be found in the file with the ending `_p.txt`. Additionally, the weights and biases of the network during training can be found in the files with the endings `_w.txt` and `_b.txt`, respectively. Files with the ending `_dkl.txt` contain the Kullback-Leibler divergence between the state represented on the trained network and the targetstate.

`Targetstates` contains the exact density matrices for the Bell state, the 3-qubit GHZ state, the 4-qubit GHZ state, and the Werner states with different amounts of noise *r*, as indicated in the file names.

The folders `Figure_x` contain the Python scripts that create the plots found in the paper figures from the provided data. 
