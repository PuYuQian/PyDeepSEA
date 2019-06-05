# README for PyDeepSEA

DanQ is a hybrid convolutional and recurrent neural network model for predicting the function of DNA de novo from sequence. This implements by PyTorch again.

# Citing DeepSEA

Jian Zhou, Olga G. Troyanskaya. Predicting the Effects of Noncoding Variants with Deep learning-based Sequence Model. Nature Methods (2015).

# INSTALL

Considering your ease of use, I have included the most recent version numbers of the software packages for the configuration that worked for me. For the record, I am using Ubuntu Linux 16.04 LTS with an NVIDIA Titan 1080Ti GPU and 32GB RAM.

## Required

- [Python] (<https://www.python.org>) (3.6.8). The easiest way to install Python and all of the necessary dependencies is to download and install [Anaconda] (https://www.anaconda.com/download/) (4.6.14).

- [PyTorch] (https://pytorch.org/) (1.0.1).

## Optional

- [CUDA] (https://developer.nvidia.com/cuda-80-download-archive) (8.0)

- [cuDNN] (https://developer.nvidia.com/rdp/cudnn-download) (7.1.3)

# USAGE

You need to first download the training, validation, and testing sets from DeepSEA. You can download the datasets from [here] (http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz). After you have extracted the contents of the tar.gz file, move the 3 .mat files into the data/ folder.

Because of my RAM limited, I firstly transform the train.mat file to 10 .pt files. If you don't worry about this problem, you can fix the train-part code according to the valid-part code in DeepSEA_train.py file.

Then you can train the model **<u>*DeepSEA_train.py</u>*** initially. Don't forget to install **visdom** and fix the **save_model_time** parameter according to your needs. Due to safety concerns, I set many model-saving checkpoints, you can fix it flexibility.(For your convenience, I've already uploaded the my bestmodel in the hyperlink, and I am grateful that if you can update it.)

When you have trained successfully, you can use **<u>*DeepSEA_test.ipynb</u>* ** to evaluate the model.Because of the  flexibility of the jupyter notebook, I integrate the pred, ROC/PR curve and aus file together.

## OPTIONAL

For convenience，you can download my trained [bestmodel] (https://pan.baidu.com/s/1h_LJIwP5ozUoBfiXFSCSnQ) with the password 'u0t6' .

# REFERENCE

> [DeepSEA] (<http://deepsea.princeton.edu/job/analysis/create/>)