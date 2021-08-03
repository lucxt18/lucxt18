###########
# --Imports
###########

import numpy as np
import matplotlib.pyplot as plt
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

# -- Data load
import scipy.io as sio

# -- LED Lamp DATA
from LCD_LED import *

#########################################
# -- STL-10 (Loading the Compressed Data)
#########################################
# Loading and normalizing STL10 :
# The output of torchvision datasets are PILImage images of range [0, 1].
# RGB images transformed into grayscale images.

print("Loading STL-10 DATA")

img_size = 64  # Height-width dimension of the unknown image
n = img_size**2 # longueur de l'image à recupèrer (n = img_sizeximg_size pixels)

data_root = Path('/home/licho/Documentos/Stage/Codes/STL10/')
transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = datasets.STL10(root=data_root,
                          split='train+unlabeled',
                          transform=transform)
batch_size = 256
dataloader = \
    torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

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
N0 = 50 # noise level (light intensity)
sig = 0.5  # std maximum total number of photons
N0_NVMS = 1 # Photon level for matrix noise stabilization

# -- Precomputed Noise Matrix Measurements
Noise_Variance = np.load(precompute_root / 'NVMS_{}_CR_{}_batch_size_{}.npy'.format(N0_NVMS, CR, batch_size))
print('loaded : NVMS_{}_CR_{}_batch_size_{}.npy'.format(N0_NVMS, CR, batch_size))

#####################
# -- Model definition
#####################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --network architecture --> ['c0mp', 'comp','pinv', 'free'] --> [0, 1, 2, 3]
net_arch = 0

# -- Available models classes : [compNet, noiCompNet, DenoiCompNet, DenoiCompNetSigma]
model1 = DenoiCompNet(n=img_size, M=CR, Mean=Mean_had, Cov=Cov_had, NVMS=Noise_Variance, variant=net_arch, N0=N0, sig=sig, Post=False)
model1 = model1.to(device)

model2 = DenoiCompNetNVMS(n=img_size, M=CR, Mean=Mean_had, Cov=Cov_had, NVMS=Noise_Variance, variant=net_arch, N0=N0, sig=sig, Post=False)
model2 = model2.to(device)

################################################
# -- LED Lamp DATA (Loading the Compressed Data)
################################################

print('Loading precomputed experimetal data')
# Path to precomputed experimetal data
expe_root = "/home/licho/Documentos/Stage/Codes/Test/expe_2/"

# Data parameters
channel = 548
sig = 0.5  # std maximum total number of photons
C = 1070
s = 55
K = 1.6  # Normalisation constant

#  -- Precomputed data (Average and covariance matrix -- for experimental data)
my_average_file = Path(expe_root) / ('Average_{}x{}'.format(img_size, img_size) + '.mat')
my_cov_file = Path(expe_root) / ('Cov_{}x{}'.format(img_size, img_size) + '.mat')
print('Loading covariance and mean')
Mean_had_1 = sio.loadmat(my_average_file)
Cov_had_1 = sio.loadmat(my_cov_file)

# Hadamard matrix definition

my_transform_file = Path(expe_root) / ('transform_{}x{}'.format(img_size, img_size) + '.mat')
H = sio.loadmat(my_transform_file)
H = (1 / img_size) * H["H"]

# Normalisation of imported Mean and Covariance.

Mean_had_1 = Mean_had_1["mu"] - np.dot(H, np.ones((img_size ** 2, 1)))
Mean_had_1 = np.reshape(Mean_had_1, (img_size, img_size))
Mean_had_1 = np.amax(Mean_had) / np.amax(Mean_had_1) * Mean_had_1
Cov_had_1 = Cov_had_1["C"]
Cov_had_1 = np.amax(Cov_had) / np.amax(Cov_had_1) * Cov_had_1

Var = Cov2Var(Cov_had_1)
Perm = Permutation_Matrix(Var)

titles_expe = ["noObjectD_1_0.0_variance", "noObjectD_1_0.3_02_variance"] + \
              ["noObjectD_1_0.3_03_variance", "noObjectD_1_0.3_04_variance"] + \
              ["noObjectD_1_0.3_01_variance"] + \
              ["noObjectD_1_0.3_01_variance", "noObjectD_1_0.6_variance"] + \
              ["noObjectD_1_1.0_variance", "noObjectD_1_1.3_variance"]



nflip = [1 for i in range(len(titles_expe))]
expe_data = [expe_root + titles_expe[i] for i in range(len(titles_expe))]

m_list = load_data_list_index(expe_data, nflip, CR, K, Perm, img_size, num_channel=channel)

m_prim = []
m_prim.append(sum(m_list[:4]) + m_list[6])
m_prim.append(sum(m_list[:2]))
m_prim.append(m_list[0]);
m_prim.append(m_list[6] + m_list[8])
m_prim = m_prim + m_list[7:]
m_list = m_prim

# -- Loading Ground Truth
# We normalize the incoming data, so that it has the right functioning range for neural networks to work with.

GT = raw_ground_truth_list_index(expe_data, nflip, H, img_size, num_channel=channel)
# Good values 450 - 530 -  548 - 600
GT_prim = []
GT_prim.append(sum(GT[:4]) + GT[6])
GT_prim.append(sum(GT[:2]))
GT_prim.append(GT[0])
GT_prim.append(GT[6] + GT[8])
GT_prim = GT_prim + GT[7:]
GT = GT_prim
max_list = [np.amax(GT[i]) - np.amin(GT[i]) for i in range(len(GT))]
GT = [((GT[i] - np.amin(GT[i])) / max_list[i]) * 2 - 1 for i in range(len(GT))]
max_list = [max_list[i] / K for i in range(len(max_list))]

###########################
# -- Displaying the results
###########################

# Once all the networks have been loaded, we apply those networks on the loaded Compressed Data.
# Attention to general model configuration : 'Network architecture' +  'Model class' +  ' Reconstructor method'

plt.rcParams['figure.figsize'] = [30, 10]

from time import perf_counter
titles = ["GT", "Diag approx", "Full inverse", "Inverse approx + NVMS"]
title_lists = []
Additional_info = [["N0 = {}".format(round(max_list[i])) if j == 0 else "" for j in range(len(titles))] for i in
                   range(len(max_list))]
Ground_truth = torch.Tensor(GT[0]).view(1, 1, 1, img_size, img_size).repeat(1, len(titles), 1, 1, 1)
outputs = []

with torch.no_grad():
    for i in range(len(GT)):
        list_outs = []

        # -- Methods :

        # 1. Diag approx
        x_c0mp_diag = model1.forward_reconstruct(1 / K * m_list[i] * 4 , 1, 1, img_size, img_size)

        # 2. Fullinverse :
        x_c0mp_full = model2.forward_reconstruct_fullinverse(1 / K * m_list[i] * 4  , 1, 1, img_size, img_size)

        # 5. Inverse approx + NVMS
        x_c0mp_NVMS = model2.forward_reconstruct(1 / K * m_list[i] * 4 , 1, 1, img_size, img_size)

        # 6. model.forward_N0_reconstruct_approxinverse


        gt = torch.Tensor(GT[i]).to(device)
        gt = gt.view(1, 1, img_size, img_size)
        list_outs.append(gt)
        list_outs.append(x_c0mp_diag)
        list_outs.append(x_c0mp_full)
        list_outs.append(x_c0mp_NVMS)

        output = torch.stack(list_outs, axis=1)

        psnr = batch_psnr_vid(Ground_truth, output)

        outputs.append(torch2numpy(output))

        title_lists.append(["{} {},\n PSNR = {}".format(titles[j], Additional_info[i][j], round(psnr[j], 2)) for j in
                            range(len(titles))])



nb_disp_frames = len(titles);

outputs_0 = outputs[:1];
outputs_1 = outputs[1:4];
outputs_2 = outputs[4:];

title_lists_0 = title_lists[:1];
title_lists_1 = title_lists[1:4];
title_lists_2 = title_lists[4:];

compare_video_frames(outputs_0, nb_disp_frames, title_lists_0);
compare_video_frames(outputs_1, nb_disp_frames, title_lists_1);
compare_video_frames(outputs_2, nb_disp_frames, title_lists_2);

#############################################
# -- STL-10 Cat (Loading the Compressed Data)
#############################################

titles_expe = ["stl10_05_1.5_0.0_0{}_variance".format(i) for i in range(1,7)]+\
              ["stl10_05_1_0.3_variance", "stl10_05_1_0.6_variance"]

expe_data = [expe_root+titles_expe[i] for i in range(len(titles_expe))];
nflip = [1.5 for i in range(len(titles_expe))];
nflip[-2:] = [1 for i in range(len(nflip[-2:]))]
channel = 581;
m_list = load_data_list_index(expe_data, nflip, CR, K, Perm, img_size, num_channel = channel);
m_prim = [];
m_prim = [];
m_prim.append(sum(m_list[:7]));
m_prim.append(m_list[0]+m_list[1]);
m_prim.append(m_list[2]);
m_prim = m_prim+m_list[-2:];
m_list = m_prim;

#########################
# -- Loading Ground Truth
#########################

### We normalize the incoming data, so that it has the right functioning range for neural networks to work with.

GT=raw_ground_truth_list_index(expe_data, nflip, H, img_size, num_channel = channel);
GT_prim = [];
GT_prim.append(sum(GT[:7]));
GT_prim.append(GT[0]+GT[1]);
GT_prim.append(GT[2]);
GT_prim = GT_prim+GT[-2:];
GT = GT_prim;
max_list = [np.amax(GT[i])-np.amin(GT[i]) for i in range(len(GT))];

GT = [((GT[i]-np.amin(GT[i]))/max_list[i])*2-1 for i in range(len(GT))];
max_list = [max_list[i]/K for i in range(len(max_list))];

plt.rcParams['figure.figsize'] = [30, 10]

title_lists = [];
Additional_info = [["N0 = {}".format(round(max_list[i])) if j == 0 else "" for j in range(len(titles))] for i in
                   range(len(max_list))]
Ground_truth = torch.Tensor(GT[0]).view(1, 1, 1, img_size, img_size).repeat(1, len(titles), 1, 1, 1);
outputs = [];

with torch.no_grad():
    for i in range(len(GT)):
        list_outs = []

        # -- Methods :

        # 1. Diag approx
        x_c0mp_diag = model1.forward_reconstruct(1 / K * m_list[i] * 4 , 1, 1, img_size, img_size)

        # 2. Fullinverse :
        x_c0mp_full = model2.forward_reconstruct_fullinverse(1 / K * m_list[i] * 4 , 1, 1, img_size, img_size)

        # 5. Inverse approx + NVMS
        x_c0mp_NVMS = model2.forward_reconstruct(1 / K * m_list[i] * 4, 1, 1, img_size, img_size)

        # 6. model.forward_N0_reconstruct_approxinverse


        gt = torch.Tensor(GT[i]).to(device)
        gt = gt.view(1, 1, img_size, img_size)
        list_outs.append(gt)
        list_outs.append(x_c0mp_diag)
        list_outs.append(x_c0mp_full)
        list_outs.append(x_c0mp_NVMS)

        output = torch.stack(list_outs, axis=1)

        psnr = batch_psnr_vid(Ground_truth, output)

        outputs.append(torch2numpy(output))

        title_lists.append(["{} {},\n PSNR = {}".format(titles[j], Additional_info[i][j], round(psnr[j], 2)) for j in
                            range(len(titles))])



nb_disp_frames = len(titles);

outputs_0 = outputs[:1];
outputs_1 = outputs[1:4];
outputs_2 = outputs[4:];

title_lists_0 = title_lists[:1];
title_lists_1 = title_lists[1:4];
title_lists_2 = title_lists[4:];

compare_video_frames(outputs_0, nb_disp_frames, title_lists_0);
compare_video_frames(outputs_1, nb_disp_frames, title_lists_1);
compare_video_frames(outputs_2, nb_disp_frames, title_lists_2);