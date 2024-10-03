# CSI-Prediction

# Pytorch code for "Spectral Temporal Graph Neural Network for massive MIMO CSI Prediction"
(c) Sharan Mourya, email: sharanmourya7@gmail.com
## Introduction
This repository holds the pytorch implementation of the original models described in the paper

Sharan Mourya, Pavan Reddy, Sai Dhiraj Amuru, ["Spectral Temporal Graph Neural Network for massive MIMO CSI Prediction"](https://ieeexplore.ieee.org/abstract/document/10457056)

## Requirements
- Python >= 3.7
- [PyTorch >= 1.7.1](https://pytorch.org/get-started/locally/)
- [Scipy >= 1.5.4](https://scipy.org/install/)
- Numpy >= 1.19.5
- Pandas >= 1.1.5


## Steps to follow to obtain Spectra Efficiency plots

#### 1) Dataset 

We have provided the compressed channel matrices for Urban Macro (UMa) scenario with code word dimensions of 128 and 256 (512 is provided in the google folder shared here due to its large size). The channel matrices are according to the 3GPP 3-D channel model as specified in 38.901. These matrices are compressed using [STNet](https://github.com/sharanmourya/Pytorch_STNet) and are stored in a **.csv** (for ex: **UMa_128.csv**) file for efficient access by the code. Decompressed channel matrices for various mobilities are available at [Folder](https://drive.google.com/drive/folders/1RPfxECfrHL2oumEzVRQxBvq635p_T2B1?usp=drive_link). However, you can generate your own compressed dataset by sending your channel matrices into the stnet.py file from the STNet repository (make sure to match the dimensions to 32x32x2).

#### 2) Training and Evaluation
Set the window size, horizon, dataset name, and other training parameters in **main.py** before running it. After training is finished, store the predicted channel matrices in a convenient location by defining the path in the **handler.py** file (lines 94-95). After training the GNN, import the saved dataset into STNet by running **compare.py** with appropriate file path (lines 442-443) to get the final decompressed channel matrices (note that you need to convert .csv to .pt before importing). Store them in a convenient location by changing lines 489-490.

#### 3) Plotting Results
Run **spectral_eficiency.ipynb** by importing the decompressed channel matrices obtained in the previous step. This step produces the spectral efficiency plots of STEM GNN for various mobilities.

## Direct Method (without training)

In the shared Google [Folder](https://drive.google.com/drive/folders/1RPfxECfrHL2oumEzVRQxBvq635p_T2B1?usp=drive_link), there are also predicted channel matrices that are obtained after the training is performed on both STEM GNN and STNet. These predicted matrices can also be directly imported through the file **spectral_eficiency.ipynb** to obtain the spectral efficieny plots of STEM GNN.
