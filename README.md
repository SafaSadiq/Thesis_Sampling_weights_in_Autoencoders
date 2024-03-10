# Thesis_Sampling_weights_in_Autoencoders

This repository contains the coder for all the experiments coducted for my Master's Thesis "Sampling weights in Autoencoders"

The thesis aims to extend the use of data-driven sampling in neural networks to an autoencoder setting. The "Sampling Where It Matters (SWIM)" algorithm is used to sample the weights and biases of the network that are then fixed and need not be iteratively trained. However, simply sampling points from the input space does not necessarily consider the intrinsic structure and complexities in the data. Hence, kernel representation learning using a contrastive loss function is combined with data-driven sampling to learn an embedding that can be used to reconstruct the data accurately. The approach is tested on benchmark image datasets including MNIST and CIFAR-10 and shown to have promising results. 

Structure
------------
**swimnetworks/** implements the SWIM algorithm for sampling weights of neural networks, found in the paper: https://arxiv.org/abs/2306.16830

**Autoencoder_Experiments_and_Results/Modules** contains all the modules and helper functions needed

**Autoencoder_Experiments_and_Results/Autoencoder_Experiments.py** contains the code for all the experiments run to test the autoenocder with different datasets and hyperparameters

**Autoencoder_Experiments_and_Results/Autoencoder_comparison_to_other_approaches.py** contains the experiments for comparing the approach to other approaches

**Autoencoder_Experiments_and_Results/maniolds.ipynb** is a notebook to view results of classical manifold learning algorithms on manifolds from sklearn and testing our autoencoder on those manifolds

