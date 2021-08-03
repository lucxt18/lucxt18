###########
# --Imports
###########

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd
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
from spyrit.learning.model_Had_DCAN_Iner import * # models
from spyrit.misc.metrics import * # psnr metrics
from spyrit.learning.nets import * # traning, load, visualization...

#########################################
# -- STL-10 (Loading the Compressed Data)
#########################################
# Loading and normalizing STL10 :
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]. Also
# RGB images transformed into grayscale images.

print("Loading STL-10 DATA")

img_size = 64  # Height-width dimension of the unknown image
n = img_size ** 2 # longueur de l'image à recupèrer (n = img_sizeximg_size pixels)
batch_size = 256

data_root = Path('/home/licho/Documentos/Stage/Codes/STL10/')
transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = \
    torchvision.datasets.STL10(root=data_root, split='train+unlabeled',download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)

testset = \
    torchvision.datasets.STL10(root=data_root, split='test',download=False, transform=transform)
testloader =  torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)

dataloaders = {'train':trainloader, 'val':testloader}
print('dataloaders are ready')

#####################################################
# -- Precomputed data (Average and covariance matrix)
#####################################################
# -- Path to precomputed data (Average and covariance matrix -- for model)
precompute_root = Path('/home/licho/Documentos/Stage/Codes/Test/')
Cov_had = np.load(precompute_root / "Cov_{}x{}.npy".format(img_size, img_size))
Mean_had = np.load(precompute_root / 'Average_{}x{}.npy'.format(img_size, img_size))
print("Average and covariance matrix loaded")

###########################
# -- Acquisition parameters
###########################
print('Loading Acquisition parameters ( Covariance and Mean matrices) ')
# -- Compressed Reconstruction via CNN (CR = 1/8)
CR = 512  # Number of patterns ---> M = 1024 measures
N0 = 50 # noise level (light intensity) used in model training
N0_test = 50 # noise level (light intensity) used in model test
N0_NVMS = 1 # Photon level for matrix noise stabilization

# -- Precomputed Noise Matrix Measurements
Noise_Variance = np.load(precompute_root / 'NVMS_{}_CR_{}_batch_size_{}.npy'.format(N0_NVMS, CR, batch_size))
print('loaded : NVMS_{}_CR_{}_batch_size_{}.npy'.format(N0_NVMS, CR, batch_size))
####################
# -- Data parameters
####################

sig = 0.5  # std maximum total number of photons
C = 1070 # \mu_{dark}
s = 55 # sigma^{2}_{dark}
K = 1.6  # Normalisation constant

# -- Conjugate gradient parameters
tau = 0.001
Niter = 5

#####################
# -- Model definition
#####################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --network architecture --> ['c0mp', 'comp','pinv', 'free'] --> [0, 1, 2, 3]
net_type = ['NET_c0mp', 'NET_comp','NET_pinv', 'NET_free']
net_arch = 0

# --denoise --> [0,1]
denoise = 1

# --inversion --> [0,1]
inversion_list = [0,1]

# -- Optimisation parameters

# Number of training epochs
num_epochs = 20

# Regularisation Parameter
reg = 1e-7

# Learning Rate
lr = 1e-3

# Scheduler Step Size
step_size = 10

# Scheduler Decrease Rate
gamma = 0.5

# -- Loading Neural Network
model_root = Path('/home/licho/Documentos/Stage/Codes/Train_models/')

# -- Available models classes : [compNet, noiCompNet, DenoiCompNet, DenoiCompNetSigma]

# -- This model take the Diagonal approximation

"""
model1 = DenoiCompNet(n=img_size, M=CR, Mean=Mean_had, Cov=Cov_had, NVMS=Noise_Variance, variant=net_arch, N0=N0_test, sig=sig)
suffix1 = '_N0_{}_sig_{}_NVMS_{}_Denoi_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, N0_NVMS, img_size, CR, num_epochs, lr, step_size, gamma, batch_size, reg)
title1 = model_root / (net_type[net_arch] + suffix1)

load_net(title1, model1, device)
model1 = model1.to(device)

print('Number of trainable parameters: {}'.format(count_trainable_param(model1)))
print('Total number of parameters: {}'.format(count_param(model1)))


# -- Models with full matrix inversion
model2 = DenoiCompNetNVMS(n=img_size, M=CR, Mean=Mean_had, Cov=Cov_had, NVMS=Noise_Variance, variant=net_arch, N0=N0_test, sig=sig)
suffix2 = '_N0_{}_sig_{}_DenoiFull_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, img_size, CR, num_epochs, lr, step_size, gamma, batch_size, reg)
title2 = model_root / (net_type[net_arch] + suffix2)

load_net(title2, model2, device)
model2 = model2.to(device)

print('Number of trainable parameters: {}'.format(count_trainable_param(model2)))
print('Total number of parameters: {}'.format(count_param(model2)))
"""

# -- Matrix approximation by Noise Variance Matrix Stabilization
model3 = DenoiCompNetNVMS(n=img_size, M=CR, Mean=Mean_had, Cov=Cov_had, NVMS=Noise_Variance, tau=tau, Niter=Niter, variant=net_arch, N0=N0, sig=sig, Post=False)
model3 = model3.to(device)
"""
suffix3 = '_N0_{}_sig_{}_NVMS_{}_DenoiNVMS_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, N0_NVMS, img_size, CR, num_epochs, lr, step_size, gamma, batch_size, reg)
title3 = model_root / (net_type[net_arch] + suffix4)

load_net(title3, model3, device)
"""

# -- Conjugate gradient descent

model4 = DenoiCompInertialNet(n=img_size, M=CR, Mean=Mean_had, Cov=Cov_had, NVMS=Noise_Variance, tau=tau, Niter=Niter, variant=net_arch, N0=N0, sig=sig)
suffix4 = '_N0_{}_sig_{}_NVMS_{}_DenoiInertie_Niter_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, N0_NVMS,Niter, img_size, CR, num_epochs, lr, step_size, gamma, batch_size, reg)
title4 = model_root / (net_type[net_arch] + suffix4)

load_net(title4, model4, device)
model4 = model4.to(device)

#############################
# -- We fix an image for test
#############################
# this is a ndarray  of size : (batch_size, 1, 64, 64) containing the STL-10 images
inputs, classes = next(iter(dataloaders['train']))

# We select an image from batch
N0_img = 187 #  test [18, 180, 78] train
img = inputs[N0_img, 0, :, :]

path = '/home/licho/anaconda3/lib/python3.8/site-packages/spyrit/'

Lena =plt.imread(path + "8-bit-256-x-256-Grayscale-Lena-Image.png")

# We transform the image in the normalized range [-1, 1]
Lena = 2 * Lena[:,:,0] - 1

# We take a patch of 64 x 64 pixels
# img = torch.tensor(Lena[100:164,100:164])

###################################
# -- Simulation of adquisition data
###################################
# 1. Image to vector reshape to measurement domain
f = torch.reshape(img, (n, 1)).to(device)
# 2. Simulation of poisson measures + normal samples of dark current (in counts)
#m1 = model1.forward_acquire(f, 1, 1, img_size, img_size)
# m2 = model2.forward_acquire(f, 1, 1, img_size, img_size)
m3 = model3.forward_acquire(f, 1, 1, img_size, img_size)
m4 = model4.forward_acquire(f, 1, 1, img_size, img_size)

###############################################################
# -- Displaying the results -- Models without precomputed noise
###############################################################
# Once all the networks have been loaded, we apply those networks on the loaded Compressed Data.
# Attention to general model configuration : 'Network architecture' +  'Model class' +  ' Reconstructor method'

# -- Methods :

## -- Model 1 : Denoising stage is performed and the diagonal approximation is taken (Paper approach)
# f_c0mp_diag = model1.forward_reconstruct(m1, 1, 1, img_size, img_size).cpu().detach().numpy()

## -- Model 2 : Denoising stage and a full matrix inversion are performed
# f_c0mp_full = model2.forward_reconstruct_fullinverse(m2, 1, 1, img_size, img_size).cpu().detach().numpy()
f_c0mp_NVMS = model3.forward_reconstruct(m3, 1, 1, img_size, img_size).cpu().detach().numpy()

f_c0mp_Inertie = model4.forward_reconstruct(m4, 1, 1, img_size, img_size).cpu().detach().numpy()

###############################################################
# -- Displaying the results -- Models without precomputed noise
###############################################################

# numpy ground-true
Gt = img.numpy()

fig, axes = plt.subplots(figsize=(15,15))
fig.suptitle("Images reconstructed with  a number of patterns (CR) of =%i" % CR)
plt.axis('off')
rows = 1
cols = 3

"""
f_diag = f_c0mp_diag[0,0,:,:]
fig.add_subplot(rows,cols,1)
plt.title("Diag approx, N0 =%i" %  N0_test)
plt.xlabel("PSNR =%.3f" %psnr(Gt, f_diag))
fig2 = plt.imshow(f_diag, cmap='gray')
plt.colorbar(fig2, shrink=0.32)

f_full = f_c0mp_full[0,0,:,:]
fig.add_subplot(rows,cols,2)
plt.title("Full inverse, N0 =%i" %  N0_test)
plt.xlabel("PSNR =%.3f" %psnr(Gt, f_full ))
fig3 = plt.imshow(f_full, cmap='gray')
plt.colorbar(fig3, shrink=0.32)
"""

f_NVMS = f_c0mp_NVMS[0,0,:,:]
fig.add_subplot(rows,cols,1)
plt.title("NVMS, N0 =%i" %  N0_test)
plt.xlabel("PSNR =%.3f" %psnr(Gt, f_NVMS ))
fig2 = plt.imshow(f_NVMS, cmap='gray')
plt.colorbar(fig2, shrink=0.32)

f_inertie = f_c0mp_Inertie[0,0,:,:]
fig.add_subplot(rows,cols,2)
plt.title("Inertie, N0 =%i" %  N0_test)
plt.xlabel("PSNR =%.3f" %psnr(Gt, f_inertie))
fig3 = plt.imshow(f_inertie, cmap='gray')
plt.colorbar(fig3, shrink=0.32)

fig.add_subplot(rows,cols,3)
plt.title("GT")
fig4 = plt.imshow(Gt, cmap='gray')
plt.colorbar(fig4, shrink=0.32)

fig.tight_layout()
plt.grid(False)
plt.show()

########################################################################################################################

train_path4 = model_root/('TRAIN_c0mp'+suffix4+'.pkl')
train_net_Inertia = read_param(train_path4)

plt.rcParams.update({'font.size': 12})

# Training Plot
fig1, ax = plt.subplots(figsize=(10,6))
plt.title('Comparison of loss curves for NVMS and inertial models with N0 =%i' % N0_test)
ax.set_xlabel('Time (epochs)')
ax.set_ylabel('Loss (MSE)')
# ax.plot(train_net_diag.val_loss,'g', linewidth=1.5)
# ax.plot(train_net_full.val_loss,'r', linewidth=1.5)
# ax.plot(train_net_NVMS.val_loss,'c', linewidth=1.5)
ax.plot(train_net_Inertia.val_loss,'c', linewidth=1.5)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.grid(True)
ax.legend(('Inertia'),  loc='upper right')

"""
# Load training history
train_path1 = model_root/('TRAIN_c0mp'+suffix1+'.pkl')
train_net_diag = read_param(train_path1)

train_path2 = model_root/('TRAIN_c0mp'+suffix2+'.pkl')
train_net_full = read_param(train_path2)


train_path4 = model_root/('TRAIN_c0mp'+suffix4+'.pkl')
train_net_NVMS = read_param(train_path4)

plt.rcParams.update({'font.size': 12})

# Training Plot
fig1, ax = plt.subplots(figsize=(10,6))
plt.title('Comparison of loss curves for diagonal approximation and inversion models with N0 =%i' % N0_test)
ax.set_xlabel('Time (epochs)')
ax.set_ylabel('Loss (MSE)')
ax.plot(train_net_diag.val_loss,'g', linewidth=1.5)
# ax.plot(train_net_full.val_loss,'r', linewidth=1.5)
ax.plot(train_net_NVMS.val_loss,'c', linewidth=1.5)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.grid(True)
ax.legend(('Diag approx', 'Inverse Taylor approx + NVMS'),  loc='upper right')
# ax.legend(('Diag approx : 78 m 23 s', 'Full inversion : 159m 38 s', 'Inverse Taylor approx + NVMS : 156m 30s'),  loc='upper right')

"""


#Boxplot



"""
psnr_NETdiag, psnr_diag = dataset_psnr(dataloaders['val'], model1, device);
print_mean_std(psnr_NETdiag, 'diagNET ')
print_mean_std(psnr_diag, 'diag ')

psnr_NETFull, psnr_Full = dataset_psnr(dataloaders['val'], model2, device)
print_mean_std(psnr_NETFull, 'NETfull ')
print_mean_std(psnr_Full, 'full ')

psnr_NVMS = dataset_psnr(dataloaders['val'], model4, device, Post=False)
# print_mean_std(psnr_NETNVMS, 'NVMS')
print_mean_std(psnr_NVMS, 'NVMS')

plt.rcParams.update({'font.size': 16})
plt.figure()
sns.set_style("whitegrid")
bx = sns.boxplot(data=pd.DataFrame([psnr_NETdiag, psnr_NVMS]).T)
# bx = sns.boxplot(data=pd.DataFrame([psnr_NETdiag, psnr_NETFull, psnr_NVMS]).T)
bx.set_xticklabels(['NETdiag', 'NVMS'])
# bx.set_xticklabels(['NETdiag', 'NETFull', 'NVMS'])
bx.set_ylabel('PSNR')

"""





