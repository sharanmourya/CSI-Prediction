# CSI-Prediction

# Pytorch code for "Spectral Temporal Graph Neural Network for massive MIMO CSI Prediction"
(c) Sharan Mourya, email: sharanmourya7@gmail.com
## Introduction
This repository holds the pytorch implementation of the original models described in the paper

Sharan Mourya, Pavan Reddy, Sai Dhiraj Amuru, "Spectral Temporal Graph Neural Network for massive MIMO CSI Prediction"

## Requirements
- Python >= 3.7
- [PyTorch >= 1.2](https://pytorch.org/get-started/locally/)
- [Scipy >= 1.8.0](https://scipy.org/install/)
- PyTorch Geometric
- MATLAB


## Steps to follow

#### 1) Generate Dataset

Run the file **ChannelGeneration.m** with the require number of users **K** to generate the one-ring channel matrices. Now run the **GraphGeneration.py** to convert the channel matrices to graph type data compatible with Pytorch Geometric. This generates a wireless communication graph with sum rate distance as the edge weights as explained in the paper. 

Note: Make sure to edit the path names in both the files to store the dataset at a convenient location.

#### 2) Training the GNN
Set the number of users, **K** and the dataset path name in **k_clique_user_pairing.py** as defined in the previous step.

Now run the file **k_clique_user_pairing.py** to begin training and evaluation...

## Acknowledgement
This repository is built upon the the [Erdos Goes Neural Pipeline open source code](https://github.com/Stalence/erdos_neu). Thanks Stalence for the amazing work.
