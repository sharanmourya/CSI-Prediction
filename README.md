# CSI-Prediction

# Pytorch code for "Spectral Temporal Graph Neural Network for massive MIMO CSI Prediction"
(c) Sharan Mourya, email: sharanmourya7@gmail.com
## Introduction
This repository holds the pytorch implementation of the original models described in the paper

Sharan Mourya, Pavan Reddy, Sai Dhiraj Amuru, "Spectral Temporal Graph Neural Network for massive MIMO CSI Prediction"

## Requirements
- Python >= 3.7
- [PyTorch >= 1.7.1](https://pytorch.org/get-started/locally/)
- [Scipy >= 1.5.4](https://scipy.org/install/)
- Numpy >= 1.19.5
- Pandas >= 1.1.5


## Steps to follow

#### 1) Dataset 

We have provided the channel matrices for Urban Macro (UMa) scenario with code word dimensions of 128 and 256. The channel matrices are according to the 3GPP 3-D channel model and are stored in a **.csv** (for ex: **UMa_128.csv**) file for efficient access by the code. 

#### 2) Training
Set the window size, horizon, dataset name, and other training parameters in **main.py** before running it. After training is finished, store the predicted channel matrices in a convenient location by defining the path in the **handler.py** file. After trining the GNN, import the saved dataset into STNET repository by cloning [STNet](https://github.com/sharanmourya/Pytorch_STNet) and then running stnet.py to get the final decompressed channel matrices.

#### 3) Plotting Results
Run **TX_LSTM_RNN.ipynb**. This trains the RNN, LSTM, and transformer models and produces results directly comparing them with the STEM GNN. Only run after importing the decompressed channel matrices obtained in the last step into this file.
