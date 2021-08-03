###########
# --Imports
###########

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append('../..')

# -- Pytorch tools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# -- Dataloading tools
import torchvision
from torchvision import datasets, models, transforms

# -- Spyrit packages
from spyrit.learning.model_Had_DCAN import * # models
from spyrit.misc.metrics import * # psnr metrics
from spyrit.learning.nets import * # traning, load, visualization...

#########################################
# -- STL-10 (Loading the Compressed Data)
#########################################
# Loading and normalizing STL10 :
# The output of torchvision datasets are PILImage images of range [0, 1].
# RGB images transformed into grayscale images.

print("Loading STL-10 DATA")
batch_size = 256
img_size = 64  # Height-width dimension of the unknown image
n = img_size**2 # longueur de l'image à recupèrer (n = img_sizeximg_size pixels)

data_root = Path('/home/licho/Documentos/Stage/Codes/STL10')
transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = \
    torchvision.datasets.STL10(root=data_root,
                          split='train+unlabeled',
                          transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size= batch_size, shuffle = True)

testset = \
   torchvision.datasets.STL10(root=data_root, split='test', transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = False)

dataloaders = {'train': testloader, 'val': testloader}

#####################################################
# -- Precomputed data (Average and covariance matrix)
#####################################################
# -- Path to precomputed data (Average and covariance matrix -- for model)
precompute_root = Path('/home/licho/Documentos/Stage/Codes/Test')
Cov_had = np.load(precompute_root / "Cov_{}x{}.npy".format(img_size, img_size))
Mean_had = np.load(precompute_root / 'Average_{}x{}.npy'.format(img_size, img_size))
print("Average and covariance matrix loaded")

###########################
# -- Acquisition parameters
###########################
print('Loading Acquisition parameters ( Covariance and Mean matrices) ')
# -- Compressed Reconstruction via CNN (CR = 1/8)
CR = 512  # Number of patterns ---> M = 1024 measures
N0 = 50 # noise level (light intensity)

# -- Precomputed Noise Matrix Measurements
Cov_noise = np.load(precompute_root / 'Cov_noise_{}.npy'.format(CR))
####################
# -- Data parameters
####################
sig = 0.5  # std maximum total number of photons

#####################
# -- Model definition
#####################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --network architecture --> ['c0mp', 'comp','pinv', 'free'] --> [0, 1, 2, 3]
net_arch = 0

# -- Available models classes : [compNet, noiCompNet, DenoiCompNet, DenoiCompNetSigma]
model = DenoiCompNetSigma(n=img_size, M=CR, Mean=Mean_had, Cov=Cov_had, Noi=Cov_noise, variant=net_arch, N0=N0, sig=sig, Post=False)
model = model.to(device)
print(model)

meas = dataset_meas(dataloaders['val'], model, device) #dataloaders['train']
meas = np.array(meas)

#%%
n1 = 2; #2,12 or 2,7
n2 = 11;

sns.jointplot(meas[:, n1], meas[:, n2], kind='reg', ratio=2)#, xlim=[-2,2], ylim=[-5, 5])
plt.xlabel('Hadamard coefficient #{}'.format(n1))
plt.ylabel('Hadamard coefficient #{}'.format(n2))
plt.show()

