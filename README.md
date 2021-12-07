# Spiking Neuromorphic Chip Learns Entangled Quantum States

This repository contains the data and code used in the paper "Spiking neuromorphic chip learns entangled quantum states" by S. Czischek et al., [arXiv:2008.01039 [cs.ET]](https://arxiv.org/abs/2008.01039) (2020). 

The code is used to train the BrainScaleS-2 spiking neuromorphic chip to represent states of small quantum many-body systems and to draw samples of qubit configurations. The properties of the sampled configurations are evaluated via expectation values of different observables.

This repository contains an example script to train the neuromorphic chip to represent a given target state based on samples drawn from the hardware (```Run_Script```). It further contains the measured data and the scripts to create the plots used in the paper figures (```Data```).

## Run Script
The Python script ```run_script.py``` can be executed to train a neural network with RBM structure to represent a Bell state of two qubits. The module ```hxsampler``` provides a connection to the BrainScaleS-2 spiking neuromorphic hardware. Via this module the connecting weights and biases of the spiking hardware neurons can be adapted during the training process. Furthermore, neuron states and spike times can be read out, which is interpreted as sampling neuron configurations when representing quantum states. This module is not yet publicly available and should be replaced with a desired neuron sampler to run the script.

The used parameters for the neuromorphic hardware setup are extracted from calibration measurements and provided in the folder ```calibration```.

This script can further be modified to represent GHZ states on larger qubit systems (see Fig. 3 in the manuscript) and implement different network architectures, such as deep and fully connected networks, as they are used in Fig. 4 in the manuscript.

## Data
This folder contains all data that is used to create the plots in the paper figures. 

```Bell_State_Data``` contains qubit configurations sampled from the neuromorphic hardware while training it to represent the two-qubit Bell state using either an RBM-structured network (```two_layer_sparse```), a fully connected two-layer network (```two_layer_dense```), or a three-layer network (```three_layer_sparse```). This data is used in Figs. 2(b), 3(a), 3(c), and 4.

```GHZ3_State_Data``` contains that data received from training the neuromorphic hardware to represent a three-qubit GHZ state. Similarly, ```GHZ4_State_Data``` contains sampled configurations when representing a four-qubit GHZ state. This data is used in Fig. 3(c).

```Werner_State_Data``` contains sampled qubit configurations when training the spiking hardware to represent a noisy Bell state (Werner state), with different amounts of noise, specified by the parameter *r* (see Fig. 2(c) in the paper). The amount of noise is denoted in the file name.

All folders with experimental data contain four files per measurement. The file names indicate the measurement setup, with number of visible neurons ```nvis```, number of hidden neurons ```nhid```, and number of measurement repetitions ```reps```. For each setup the sampled configurations can be found in the file with the ending ```_p.txt```. Additionally, the weights and biases of the network during training can be found in the files with the endings ```_w.txt``` and ```_b.txt```, respectively. Files with the ending ```_dkl.txt``` contain the Kullback-Leibler divergence between the state represented on the trained network and the targetstate.

```Targetstates``` contains the exact density matrices for the Bell state, the 3-qubit GHZ state, the 4-qubit GHZ state, and the Werner states with different amounts of noise *r*, as indicated in the file names.

The folders ```Figure_x``` contain the Python scripts that create the plots found in the paper figures from the provided data. 
