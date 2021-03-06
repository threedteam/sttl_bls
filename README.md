# The code of proposed model(CFB-BLS)

## Introduce
A Cascade Feature Block Broad Learning System (CFB-BLS) is proposed to improve the image classification performance of Broad Learning System (BLS). The proposed model contains three important parts: (1) A feature block (FB) based on squeeze-and-excitation technique is put forward for image feature extraction, and multiple feature blocks are cascaded so the model can learn discriminative features. (2) A so-called top-level dropout layer is added before the output layer of the proposed model to avoid overfitting. (3) Adam algorithm is applied to train the model so that the model can work in a way similar to Convolutional Neural Network (CNN), which is greatly beneficial to classification. With these parts, our model gets the advantages of both BLS and CNN. Experimental results show that the proposed model can achieve the accuracies of 96.25%, 87.95%, and 60.53% for dataset SVHN, CIFAR-10, and CIFAR-100 respectively. And the time complexity of the proposed model is quite low: the training time and testing time of our model are no more than 800 seconds and 6 seconds respectively if GPU is used. 

##Code for Propose Model

SVHN: cfb_bls_svhn.py

CIFAR-10: cfb_bls.py

CIFAR-100: cfb_bls_cifar100.py

## Requirements


## Runtime Environment 
Intel Xeon E5-2678 CPU with 128G memory

an NVIDIA TITAN Xp GPU

python 3.6

keras 2.2.4

## Public dataset

### SVHN

http://ufldl.stanford.edu/housenumbers/

### CIFAR-10 & CIFAR-100

http://www.cs.toronto.edu/~kriz/cifar.html
