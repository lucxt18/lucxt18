# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

#from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from fht import *
from ..misc.pattern_choice import Hadamard, matrix2conv, split
from collections import OrderedDict
import cv2
from scipy.stats import rankdata
#from ..misc.disp import *
from itertools import cycle;

from ..misc.disp import *


#######################################################################
# 1. Determine the important Hadamard Coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Helps determining the statistical best 
# Hadamard patterns for a given image size
# 

def optim_had(dataloader, root):
    """ Computes image that ranks the hadamard coefficients
    """
    inputs, classes = next(iter(dataloader))
    inputs = inputs.cpu().detach().numpy();
    (batch_size, channels, nx, ny) = inputs.shape;

    tot_num = len(dataloader)*batch_size;
    Cumulated_had = np.zeros((nx, ny));
    # Iterate over data.
    for inputs,labels in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i,0,:,:];
            h_img = np.abs(fht2(img))/tot_num;
            Cumulated_had += h_img;
    
    Cumulated_had = Cumulated_had / np.max(Cumulated_had) * 255
    np.save(root+'{}x{}'.format(nx,ny)+'.npy', Cumulated_had)
    np.savetxt(root+'{}x{}'.format(nx,ny)+'.txt', Cumulated_had)
    cv2.imwrite(root+'{}x{}'.format(nx,ny)+'.png', Cumulated_had)
    return Cumulated_had 

def hadamard_opt_spc(M ,root, nx, ny):
    msk = np.ones((nx,ny))
    had_mat = np.load(root+'{}x{}'.format(nx,ny)+'.npy');
    had_comp = np.reshape(rankdata(-had_mat, method = 'ordinal'),(nx, ny));
    msk[np.absolute(had_comp)>M]=0;
    
    conv = Hadamard(msk); 

    return conv



def Stat_had(dataloader, root):
    """ 
        Computes Mean Hadamard Image over the whole dataset + 
        Covariance Matrix Amongst the coefficients
    """

    inputs, classes = next(iter(dataloader))
    inputs = inputs.cpu().detach().numpy();
    (batch_size, channels, nx, ny) = inputs.shape;
    tot_num = len(dataloader)*batch_size;

    Mean_had = np.zeros((nx, ny));
    for inputs,labels in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i,0,:,:];
            h_img = fht2(img);
            Mean_had += h_img;
    Mean_had = Mean_had/tot_num;

    Cov_had = np.zeros((nx*ny, nx*ny));
    for inputs,labels in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i,0,:,:];
            h_img = fht2(img);
            Norm_Variable = np.reshape(h_img-Mean_had, (nx*ny,1));
            Cov_had += Norm_Variable*np.transpose(Norm_Variable);
    Cov_had = Cov_had/(tot_num-1);


    
    np.save(root+'Cov_{}x{}'.format(nx,ny)+'.npy', Cov_had)
    np.savetxt(root+'Cov_{}x{}'.format(nx,ny)+'.txt', Cov_had)
    
    np.save(root+'Average_{}x{}'.format(nx,ny)+'.npy', Mean_had)
    np.savetxt(root+'Average_{}x{}'.format(nx,ny)+'.txt', Mean_had)
    cv2.imwrite(root+'Average_{}x{}'.format(nx,ny)+'.png', Mean_had)#Needs conversion to Uint8!
    return Mean_had, Cov_had 


def img2mask(Value_map, M):
    (nx, ny) = Value_map.shape;
    msk = np.ones((nx, ny));
    ranked_data = np.reshape(rankdata(-Value_map, method = 'ordinal'),(nx, ny));
    msk[np.absolute(ranked_data)>M]=0;
    return msk

def Cov2Var(Cov):
    """
    Extracts Variance Matrix from Covarience Matrix
    """
    (Nx, Ny) = Cov.shape;
    diag_index = np.diag_indices(Nx);
    Var = Cov[diag_index];
    Var = np.reshape(Var, (int(np.sqrt(Nx)),int(np.sqrt(Nx))) );
    return Var

def Permutation_Matrix_root(root):
    """
        Returns Permutaion Matrix For The Hadamard Coefficients that ranks
        The Coefficients according to the Matrix defined By root.
    """
    had_mat = np.load(root);
    (nx, ny) = had_mat.shape;
    Reorder = rankdata(-had_mat, method = 'ordinal');
    Columns = np.array(range(nx*ny));
    P = np.zeros((nx*ny, nx*ny));
    P[Reorder-1, Columns] = 1;
    return P


def Permutation_Matrix(had_mat):
    """
        Returns Permutation Matrix For The Hadamard Coefficients that ranks
        The Coefficients according to the Matrix defined By had_mat.
    """
    (nx, ny) = had_mat.shape;
    Reorder = rankdata(-had_mat, method = 'ordinal');
    Columns = np.array(range(nx*ny));
    P = np.zeros((nx*ny, nx*ny));
    P[Reorder-1, Columns] = 1;
    return P

def maximum_Variance_Pattern(Cov,H,M):
    """
        Returns the patterns corresponding to coefficient that have the maximun
        variance for a given image database
    """
    Var = Cov2Var(Cov)
    Perm = Permutation_Matrix(Var)
    Pmat = np.dot(Perm,H);
    Pmat = Pmat[:M,:];
    return Pmat, Perm
    
def Hadamard_Transform_Matrix(img_size):
    H = np.zeros((img_size**2, img_size**2))
    for i in range(img_size**2):
        base_function = np.zeros((img_size**2,1));
        base_function[i] = 1;
        base_function = np.reshape(base_function, (img_size, img_size));
        hadamard_function = fht2(base_function);
        H[i, :] = np.reshape(hadamard_function, (1,img_size**2));
    return H

def Hadamard_stat_completion_matrices(Cov_had, Mean_had, CR):
    img_size, ny = Mean_had.shape;
    
    # choice of patterns
    Var = Cov2Var(Cov_had)
    P = Permutation_Matrix(Var)
    H = Hadamard_Transform_Matrix(img_size);

    Sigma = np.dot(P,np.dot(Cov_had,np.transpose(P)))
    mu = np.dot(P, np.reshape(Mean_had, (img_size**2,1)))
    mu1 = mu[:CR];

    Sigma1 = Sigma[:CR,:CR]
    Sigma21 = Sigma[CR:,:CR]
    
    W_p = np.zeros((img_size**2,CR))
    W_p[:CR,:] = np.eye(CR);
    W_p[CR:,:] = np.dot(Sigma21, np.linalg.inv(Sigma1));
    
    W = np.dot(H,np.dot(np.transpose(P),W_p));
    b = np.dot(H,np.dot(np.transpose(P),mu));
    return W, b, mu1, P, H

def stat_completion_matrices(P, H, Cov_had, Mean_had, CR):
    img_size, ny = Mean_had.shape;

    Sigma = np.dot(P, np.dot(Cov_had, np.transpose(P)))
    mu = np.dot(P, np.reshape(Mean_had, (img_size ** 2, 1)))
    mu1 = mu[:CR];

    Sigma1 = Sigma[:CR, :CR]
    #Sigma21 = np.zeros(Sigma[CR:, :CR].shape)
    Sigma21 = Sigma[CR:, :CR]

    W_p = np.zeros((img_size ** 2, CR))
    W_p[:CR, :] = np.eye(CR);
    W_p[CR:, :] = np.dot(Sigma21, np.linalg.inv(Sigma1));

    W = np.dot(H, np.dot(np.transpose(P), W_p));
    b = np.dot(H, np.dot(np.transpose(P), mu));
    return W, b, mu1

def Hadamard_stat_completion_extract(img,CR, P, H):
    img_size, ny = img.shape;
    f = np.reshape(img, (img_size**2,1))
    y = np.dot(P, np.dot(H, f))
    m = y[:CR];
    return m


def Hadamard_stat_completion(W, b, mu1, m):
    nxny , col = b.shape;
    img_size = int(round(np.sqrt(nxny)));
    f_star = b + np.dot(W,(m-mu1))
    img_rec = np.reshape(f_star,(img_size,img_size));
    return img_rec;

def Hadamard_stat_completion_comp(Cov,Mean,img, CR):
    img_size, ny = img.shape;
    Var = Cov2Var(Cov)
    P = Permutation_Matrix(Var)
    H = Hadamard_Transform_Matrix(img_size);

    Sigma = np.dot(P,np.dot(Cov,np.transpose(P)))
    mu = np.dot(P, np.reshape(Mean, (img_size**2,1)))
    mu1 = mu[:CR];

    Sigma1 = Sigma[:CR,:CR]
    Sigma21 = Sigma[CR:,:CR]
    
    W_p = np.zeros((img_size**2,CR))
    W_p[:CR,:] = np.eye(CR);
    W_p[CR:,:] = np.dot(Sigma21, np.linalg.inv(Sigma1));
    
    W = np.dot(H,np.dot(np.transpose(P),W_p));
    b = np.dot(H,np.dot(np.transpose(P),mu));


    f = np.reshape(img, (img_size**2,1))
    y = np.dot(P, np.dot(H, f))
    m = y[:CR];
    f_star = b + np.dot(W,(m-mu1))
    img_rec = np.reshape(f_star,(img_size, img_size));

    return img_rec;

###############################################################################
# 2. NEW Convolutional Neural Network
###############################################################################
#==============================================================================
# A. NO NOISE
#==============================================================================    
class compNet(nn.Module):
    def __init__(self, n, M, Mean, Cov, NVMS, tau, Niter, variant=0, H=None, Post=True):
        super(compNet, self).__init__()
        
        self.n = n;
        self.M = M;

        # Inicialization parameters for the conjugate gradient descent
        self.tau = tau
        self.Niter = Niter

        # Indicates if post processing task is perform
        self.Post = Post
        
        self.even_index = range(0,2*M,2);
        self.uneven_index = range(1,2*M,2);
        
        #-- Hadamard patterns (full basis)
        if type(H)==type(None):
            H = Hadamard_Transform_Matrix(self.n)
        H = n*H; #fht hadamard transform needs to be normalized
        
        #-- Hadamard patterns (undersampled basis)
        Var = Cov2Var(Cov)
        Perm = Permutation_Matrix(Var)
        Pmat = np.dot(Perm,H);
        Pmat = Pmat[:M,:];
        Pconv = matrix2conv(Pmat);
        self.H1 = torch.tensor(np.float32(Pmat))

        #-- Denoising parameters 
        Sigma_had = np.dot(Perm,np.dot(Cov,np.transpose(Perm))); 
        # We keep the full covariance matrix
        self.Sigma_had = n**2/4*Sigma_had; #(H = nH donc Cov = n**2 Cov)!
        # Selection of the MxM matrix for denoising step
        self.Sigma1_had = self.Sigma_had[:M, :M]
        # Diagonal selection
        diag_index = np.diag_indices(n**2);
        Sigma = self.Sigma_had[diag_index]; # Selection of the diagonal elements
        Sigma = Sigma[:M];
        Sigma = torch.Tensor(Sigma);
        self.sigma = Sigma.view(1,1,M);

        # The inverse of variance stabilization approximation is precalculated
        self.NVMS = np.float32(np.diag(NVMS))
        self.NVMS_inv = np.float32(np.linalg.inv(self.Sigma1_had + np.float32(NVMS)))
        self.Prod =  np.float32(self.Sigma1_had @ self.NVMS_inv)
        print('loaded precalculated matrices : NVMS, NVMS_inv, Prod')

        P1 = np.zeros((n**2,1));
        P1[0] = n**2;
        mean = n*np.reshape(Mean,(self.n**2,1))+P1;
        mu = (1/2)*np.dot(Perm, mean);
        #mu = np.dot(Perm, np.reshape(Mean, (n**2,1)))
        mu1 = torch.Tensor(mu[:M]);
        self.mu_1 = mu1.view(1,1,M);

        #-- Measurement preprocessing
        self.Patt = Pconv;
        P, T = split(Pconv, 1);
        self.P = P;
        self.T = T;
        self.P.bias.requires_grad = False;
        self.P.weight.requires_grad = False;
        self.Patt.bias.requires_grad = False;
        self.Patt.weight.requires_grad = False;
        self.T.weight.requires_grad=False;
        self.T.weight.requires_grad=False;

        #-- Pseudo-inverse to determine levels of noise.
        Pinv = (1/n**2)*np.transpose(Pmat);
        self.Pinv = nn.Linear(M,n**2, False)
        self.Pinv.weight.data=torch.from_numpy(Pinv);
        self.Pinv.weight.data=self.Pinv.weight.data.float();
        self.Pinv.weight.requires_grad=False;


        #-- Measurement to image domain
        if variant==0:
            #--- Statistical Matrix completion (no mean)
            print("Measurement to image domain: statistical completion (no mean)")
            
            self.fc1 = nn.Linear(M,n**2, False)
            
            W, b, mu1 = stat_completion_matrices(Perm, H, Cov, Mean, M)
            W = (1/n**2)*W; 

            self.fc1.weight.data=torch.from_numpy(W);
            self.fc1.weight.data=self.fc1.weight.data.float();
            self.fc1.weight.requires_grad=False;
        
        if variant==1:
            #--- Statistical Matrix completion  
            print("Measurement to image domain: statistical completion")
            
            self.fc1 = nn.Linear(M,n**2)
            
            W, b, mu1 = stat_completion_matrices(Perm, H, Cov, Mean, M)
            W = (1/n**2)*W; 
            b = (1/n**2)*b;
            b = b - np.dot(W,mu1);
            self.fc1.bias.data=torch.from_numpy(b[:,0]);
            self.fc1.bias.data=self.fc1.bias.data.float();
            self.fc1.bias.requires_grad = False;
            self.fc1.weight.data=torch.from_numpy(W);
            self.fc1.weight.data=self.fc1.weight.data.float();
            self.fc1.weight.requires_grad=False;
        
        elif variant==2:
            #--- Pseudo-inverse
            print("Measurement to image domain: pseudo inverse")
            
            self.fc1 = self.Pinv;
       
        elif variant==3:
            #--- FC is learnt
            print("Measurement to image domain: free")
            
            self.fc1 = nn.Linear(M,n**2)
            
        #-- Image correction
        self.recon = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0)),
          ('relu2', nn.ReLU()),
          ('conv3', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
        ]));

    def forward(self, x):
        b,c,h,w = x.shape;
        x = self.forward_acquire(x, b, c, h, w);
        x = self.forward_reconstruct(x, b, c, h, w);
        return x

    #--------------------------------------------------------------------------
    # Forward functions (with grad)
    #--------------------------------------------------------------------------
    def forward_acquire(self, x, b, c, h, w):
        #--Scale input image
        x = (x+1)/2; 
        #--Acquisition
        x = x.view(b*c, 1, h, w);
        x = self.P(x);
        x = F.relu(x); ## x[:,:,1] = -1/N0 ????
        x = x.view(b*c,1, 2*self.M); 
        return x
    
    def forward_maptoimage(self, x, b, c, h, w):
        #- Pre-processing (use batch norm to avoid division by N0 ?)
        x = self.T(x);
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M));
        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w)
        return x
    
    def forward_postprocess(self, x, b, c, h, w):
        x = self.recon(x)
        x = x.view(b, c, h, w)
        return x
    
    #def forward_reconstruct(self, x, b, c, h, w):
     #   x = self.forward_maptoimage(x, b, c, h, w)
     #   x = self.forward_postprocess(x, b, c, h, w)
     #   return x
     
    
    #--------------------------------------------------------------------------
    # Evaluation functions (no grad)
    #--------------------------------------------------------------------------
    def acquire(self, x, b, c, h, w):
        with torch.no_grad():
            b,c,h,w = x.shape
            x = self.forward_acquire(x, b, c, h, w)
        return x
    
    def evaluate_fcl(self, x):
        with torch.no_grad():
           b,c,h,w = x.shape
           x = self.forward_acquire(x, b, c, h, w)
           x = self.forward_maptoimage(x, b, c, h, w)
        return x
     
    def evaluate_Pinv(self, x):
        with torch.no_grad():
           b,c,h,w = x.shape
           x = self.forward_Pinv(x, b, c, h, w)
        return x
    
    def evaluate(self, x):
        with torch.no_grad():
           x = self.forward(x)
        return x
    
    def reconstruct(self, x, b, c, h, w):
        with torch.no_grad():
            x = self.forward_reconstruct(x, b, c, h, w)
        return x
   
#==============================================================================    
# B. NOISY MEASUREMENTS (NOISE LEVEL IS VARYING)
#==============================================================================
class noiCompNet(compNet):
    def __init__(self, n, M, Mean, Cov, NVMS, tau, Niter, variant, N0, sig = 0.1, H=None, Post=True):
        super().__init__(n, M, Mean, Cov, NVMS, tau, Niter, variant, H, Post)
        self.N0 = N0;
        self.sig = sig;
        self.max = nn.MaxPool2d(kernel_size = n);
        print("Varying N0 = {:g} +/- {:g}".format(N0,sig*N0))
        
    def forward_acquire(self, x, b, c, h, w):
        #--Scale input image
        x = (self.N0*(1+self.sig*torch.randn_like(x)))*(x+1)/2;
        #--Acquisition
        x = x.view(b*c, 1, h, w);
        x = self.P(x);
        x = F.relu(x);     # x[:,:,1] = -1/N0 ????
        x = x.view(b*c,1, 2*self.M); # x[:,:,1] < 0??? 
        #--Measurement noise (Gaussian approximation of Poisson)
        x = x + torch.sqrt(x)*torch.randn_like(x);  
        return x
    
    def forward_maptoimage(self, x, b, c, h, w):
        #-- Pre-processing (use batch norm to avoid division by N0 ?)
        x = self.T(x);
        x = 2/self.N0*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x
         

    def forward_Pinv(self, x, b, c, h, w):
        #-- Pre-processing (use batch norm to avoid division by N0 ?)
        x = self.T(x);
        x = 2/self.N0*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        #--Projection to the image domain
        x = self.Pinv(x);
        x = x.view(b*c,1,h,w)
        return x
 
    def forward_N0_Pinv(self, x, b, c, h, w):
        #-- Pre-processing (use batch norm to avoid division by N0 ?)
        x = self.T(x);
        #--Projection to the image domain
        x = self.Pinv(x);
        x = x.view(b*c,1,h,w)
        N0_est = self.max(x);
        N0_est = N0_est.view(b*c,1,1,1);
        N0_est = N0_est.repeat(1,1,h,w);
        x = torch.div(x,N0_est);
        x=2*x-1; 
        return x
     
    def forward_N0_maptoimage(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        
        #-- Pre-processing(Estimating No and normalizing by No) 
        x_est = self.Pinv(x);
        x_est = x_est.view(b*c,1,h,w);
        N0_est = self.max(x_est);
        N0_est = N0_est.view(b*c,1,1);
        N0_est = N0_est.repeat(1,1,self.M);
        x = torch.div(x,N0_est);
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 

        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w)
        return x
    
    def forward_N0_reconstruct(self, x, b, c, h, w):
        x = self.forward_N0_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w)
        return x
 
    def forward_stat_comp(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = x/self.N0;
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 

        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x


# ==============================================================================
# C. NOISY MEASUREMENTS (NOISE LEVEL IS VARYING) + denoising architecture
# ==============================================================================
class DenoiCompNet(noiCompNet):
    def __init__(self, n, M, Mean, Cov, NVMS, tau, Niter, variant, N0, sig=0.1, H=None, mean_denoi=False, Post=True):
        super().__init__(n, M, Mean, Cov, NVMS, tau, Niter, variant, N0, sig, H, Post)
        print("Denoised Measurements with matrix diagonal approximation")
        if self.Post:
            print('Forward post processing is performed')
        else:
            print('Only forward reconstruction step is performed')

    def forward_maptoimage(self, x, b, c, h, w):
        # -- Pre-processing(Recombining positive and negatve values+normalisation)
        var = x[:, :, self.even_index] + x[:, :, self.uneven_index];
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];
        x = x / self.N0;
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b * c, 1, self.M));

        # --Denoising
        sigma = self.sigma.repeat(b * c, 1, 1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma + var / (self.N0) ** 2), x);

        # --Projection to the image domain
        x = self.fc1(x);
        x = x.view(b * c, 1, h, w)
        return x

    def forward_reconstruct(self, x, b, c, h, w):
        x = self.forward_maptoimage(x, b, c, h, w)
        if self.Post:
            x = self.forward_postprocess(x, b, c, h, w)

        return x

    def forward_maptoimage_2(self, x, b, c, h, w):
        # -- Pre-processing(Recombining positive and negatve values+normalisation)
        var = x[:, :, self.even_index] + x[:, :, self.uneven_index];
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];
        x = x / self.N0;
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b * c, 1, self.M));

        # --Denoising
        sigma = self.sigma.repeat(b * c, 1, 1).to(x.device);
        mu_1 = self.mu_1.repeat(b * c, 1, 1).to(x.device);
        x = mu_1 + torch.mul(torch.div(sigma, sigma + var / (self.N0) ** 2), x - mu_1);

        # --Projection to the image domain
        x = self.fc1(x);
        x = x.view(b * c, 1, h, w)
        return x

    def forward_denoised_Pinv(self, x, b, c, h, w):
        # -- Pre-processing(Recombining positive and negatve values+normalisation)
        var = x[:, :, self.even_index] + x[:, :, self.uneven_index];
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];
        x = x / self.N0;
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b * c, 1, self.M));

        # --Denoising
        sigma = self.sigma.repeat(b * c, 1, 1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma + 4 * var / (self.N0) ** 2), x);

        # --Projection to the image domain
        x = self.Pinv(x);
        x = x.view(b * c, 1, h, w)
        return x

    def forward_NO_maptoimage(self, x, b, c, h, w):
        # -- Pre-processing(Recombining positive and negatve values+normalisation)
        var = x[:, :, self.even_index] + x[:, :, self.uneven_index];
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];

        # -- Pre-processing(Estimating No and normalizing by No)
        x_est = self.Pinv(x);
        x_est = x_est.view(b * c, 1, h, w);
        N0_est = self.max(x_est);
        N0_est = N0_est.view(b * c, 1, 1);
        N0_est = N0_est.repeat(1, 1, self.M);
        x = torch.div(x, N0_est);

        # --Denoising
        sigma = self.sigma.repeat(b * c, 1, 1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma + torch.div(var, N0_est ** 2)), x);
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b * c, 1, self.M));

        # --Projection to the image domain
        x = self.fc1(x);
        x = x.view(b * c, 1, h, w)
        return x;

    def forward_N0_maptoimage_expe(self, x, b, c, h, w, C, s, g):
        # -- Pre-processing(Recombining positive and negatve values+normalisation)
        var = g ** 2 * (x[:, :, self.even_index] + x[:, :, self.uneven_index]) - 2 * C * g + 2 * s ** 2;
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];

        # -- Pre-processing(Estimating No and normalizing by No)
        x_est = self.Pinv(x);
        x_est = x_est.view(b * c, 1, h, w);
        N0_est = self.max(x_est);
        N0_est = N0_est.view(b * c, 1, 1);
        N0_est = N0_est.repeat(1, 1, self.M);
        x = torch.div(x, N0_est);

        # --Denoising
        sigma = self.sigma.repeat(b * c, 1, 1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma + torch.div(var, N0_est ** 2)), x);
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b * c, 1, self.M));

        # --Projection to the image domain
        x = self.fc1(x);
        x = x.view(b * c, 1, h, w)
        return x;

    def forward_N0_reconstruct_expe(self, x, b, c, h, w, C, s, g):
        x = self.forward_N0_maptoimage_expe(x, b, c, h, w, C, s, g)
        x = self.forward_postprocess(x, b, c, h, w)
        return x

    def forward_N0_maptoimage_expe_bis(self, x, b, c, h, w, C, s, g, N0):
        # -- Pre-processing(Recombining positive and negatve values+normalisation)
        var = g ** 2 * (x[:, :, self.even_index] + x[:, :, self.uneven_index]) - 2 * C * g + 2 * s ** 2;
        var = x[:, :, self.even_index] + x[:, :, self.uneven_index];
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];

        # -- Pre-processing(Estimating No and normalizing by No)
        x_est = self.Pinv(x);
        x_est = x_est.view(b * c, 1, h, w);
        N0_est = self.max(x_est);
        N0_est = N0_est.view(b * c, 1, 1);
        N0_est = N0_est.repeat(1, 1, self.M);
        sigma = self.sigma.repeat(b * c, 1, 1).to(x.device);
        print(N0_est)
        x = x / N0;
        #        x = torch.div(x,N0_est);

        x = torch.mul(torch.div(sigma, sigma + torch.div(var, N0_est ** 2)), x);
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b * c, 1, self.M));

        #        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        #        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        #        x = torch.div(x,N0_est);
        #        x = x/N0;
        #
        #        #--Denoising
        #        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        #        x = torch.mul(torch.div(sigma, sigma+var/(N0)**2), x);
        #        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M));
        #
        # --Projection to the image domain
        x = self.fc1(x);
        x = x.view(b * c, 1, h, w)
        return x;

    def forward_N0_reconstruct_expe_bis(self, x, b, c, h, w, C, s, g, N0):
        x = self.forward_N0_maptoimage_expe_bis(x, b, c, h, w, C, s, g, N0)
        x = self.forward_postprocess(x, b, c, h, w)
        return x


# ===========================================================================================================
# D. NOISY MEASUREMENTS (NOISE LEVEL IS VARYING) + denoising architecture + full covariance matrix processing
# ===========================================================================================================

class DenoiCompNetNVMS(noiCompNet):
    def __init__(self, n, M, Mean, Cov, NVMS, tau, Niter, variant, N0, sig = 0.1, H=None, mean_denoi=False, Post=True):
        super().__init__(n, M, Mean, Cov, NVMS, tau, Niter, variant, N0, sig, H, Post)
        print("Denoised Measurements with inverse matrix approximation (NVMS)")
        if self.Post:
            print('Forward post processing is performed')
        else:
            print('Only forward reconstruction step is performed')


    """
    This method computes the denoising of the raw data in the measurement
    domain, based on a precalculate a Noise Variance Matrix Stabilization (NVMS),
    which is a matrix that takes the mean of the variance of the noised measurements, 
    for a given photon level N0 on a batch of the STL-10 database. This method allows  
    to stabilize the signal dependent variance matrix in the denoising stage. 
    A first-order taylor development is taken also for tackle the matrix inversion.
    """

    def forward_maptoimage(self, x, b, c, h, w):
        # -- Variance approximation
        var = x[:, :, self.even_index] + x[:, :, self.uneven_index]
        correction = (var / self.N0 ** 2) - torch.tensor(self.NVMS).repeat(b * c, 1, 1).to(x.device)

        # -- Pre-processing(Recombining positive and negatve values+normalisation)
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];
        x = x / self.N0;
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b * c, 1, self.M));

        # -- Statistical matrices
        NVMS_inv = torch.tensor(self.NVMS_inv).repeat(b * c, 1, 1).to(x.device) #, dtype=torch.float32
        Prod = torch.Tensor(self.Prod).repeat(b * c, 1, 1).to(x.device)

        ####################
        # -- Denoising stage
        ####################
        # --Raw data denoising
        # 1. Sigma_inv.shape = [b * c, self.M, self.M] and x.shape =  [b * c, 1, self.M]
        #    @ allows to perform the broadcasting in the matrix multiplication case
        x = torch.matmul((Prod - torch.matmul(Prod, torch.mul(NVMS_inv, torch.transpose(correction, 1, 2)))) , torch.transpose(x, 1, 2))
        # 2. We recover the original size of x
        x = torch.reshape(x, ((b * c, 1, self.M)))

        # --Projection to the image domain
        x = self.fc1(x);
        x = x.view(b * c, 1, h, w)
        return x

    def forward_reconstruct(self, x, b, c, h, w):
        x = self.forward_maptoimage(x, b, c, h, w)
        if self.Post:
            x = self.forward_postprocess(x, b, c, h, w)

        return x

########################################################################################################################

    """
    This method computes the denoising of the raw data in the measurement
    domain, based on the vanilla equation (i.e a full matrix inversion is performed)
    """

    def forward_maptoimage_fullinverse(self, x, b, c, h, w):
        # -- Pre-processing(Recombining positive and negative values+normalisation)
        var = x[:, :, self.even_index] + x[:, :, self.uneven_index];
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];
        x = x / self.N0 ;
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b * c, 1, self.M));

        # --Denoising stage :
        # 1. We operates on whole the batch sample, thus we prepare the block to perform b * c matrix inversions
        Sigma1_had = torch.Tensor(self.Sigma1_had).repeat(b * c, 1, 1).to(x.device)
        # 2. Normalisation of the Noise vector
        var = torch.div(var, self.N0 ** 2)
        var = torch.diag_embed(var).view([b * c, self.M, self.M])
        # 3. Full matrix inversion is performed on the tensor data (matrix inversion along the batch dimensions is allowed)
        Sigma_inv = torch.linalg.inv(Sigma1_had + var)

        # --Raw data denoising
        # 1. Sigma_inv.shape = [b * c, self.M, self.M] and x.shape =  [b * c, 1, self.M]
        #    torch.matmul allows to perform the broadcasting in the matrix multiplication case
        x = Sigma1_had @ Sigma_inv @ torch.transpose(x, 1, 2)
        # 2. We recover the original size of x
        x = torch.reshape(x, ((b * c, 1, self.M)))

        # --Projection to the image domain
        x = self.fc1(x);
        x = x.view(b * c, 1, h, w)
        return x;

    def forward_reconstruct_fullinverse(self, x, b, c, h, w):
        x = self.forward_maptoimage_fullinverse(x, b, c, h, w)
        if self.Post:
            x = self.forward_postprocess(x, b, c, h, w)

        return x

########################################################################################################################

    """
    This method computes the denoising of the raw data in the measurement
    domain, based on the vanilla equation approximation, basically
    a first-order taylor development is taken for tackle the matrix inversion.
    """

    def forward_maptoimage_approxinverse(self, x, b, c, h, w):
        # -- Pre-processing(Recombining positive and negatve values+normalisation)
        var = x[:, :, self.even_index] + x[:, :, self.uneven_index];
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];
        x = torch.div(x, self.N0)
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b * c, 1, self.M));
        #var = torch.div(var, self.N0 ** 2)

        # --Denoising stage :
        # 1. Diagonal an not-diagonal extraction
        diag_index = np.diag_indices(self.M)
        T1 = self.Sigma1_had - np.diag(self.Sigma1_had[diag_index])

        # 2. We operates on whole the batch sample, thus we prepare the block to perform b * c matrix operations
        T1 = torch.Tensor(T1).repeat(b * c, 1, 1).to(x.device)
        D1 = self.sigma.repeat(b * c, 1, 1).to(x.device)

        # 3. Normalisation of the Noise vector
        var = torch.div(var, self.N0 ** 2)
        # 4. Diagonal inversion
        diag_compt1 = torch.div(D1, D1 + var)
        diag_compt2 = torch.div(1, (D1 + var))
        # 5. Visualisation of every tensor as a batch of diagonal matrices (to perform the matrix multiplication)
        compt1 = torch.diag_embed(diag_compt1).view([b * c, self.M, self.M])
        compt2 = torch.diag_embed(diag_compt2).view([b * c, self.M, self.M])
        var = torch.diag_embed(var).view([b * c, self.M, self.M])

        # 5. Block approximation
        M = compt1 + ( var @ compt2 @ T1  @compt2 )

        # --Raw data denoising
        # 1. Sigma_inv.shape = [b * c, self.M, self.M] and x.shape =  [b * c, 1, self.M]
        #    @ allows to perform the broadcasting in the matrix multiplication case
        x = M @ torch.transpose(x, 1, 2)
        # 2. We recover the original size of x
        x = torch.reshape(x, ((b * c, 1, self.M)))

        # --Projection to the image domain
        x = self.fc1(x);
        x = x.view(b * c, 1, h, w)
        return x

    def forward_reconstruct_approxinverse(self, x, b, c, h, w):
        x = self.forward_maptoimage_approxinverse(x, b, c, h, w)
        if self.Post:
            x = self.forward_postprocess(x, b, c, h, w)
        return x

########################################################################################################################

class DenoiCompInertialNet(noiCompNet):
    def __init__(self, n, M, Mean, Cov, NVMS, tau, Niter, variant, N0, sig = 0.1, H=None, mean_denoi=False):
        super().__init__(n, M, Mean, Cov, NVMS, tau, Niter, variant, N0, sig, H)
        print("Denoised Measurements in the conjugate gradient descent algorithm are performed")

    # model --> forward() in the training phase every model call is a forward method instance
    # In the training phase only <<forward_reconstruct ---> forward_maptoimage >> are to taking in count !!!
    # Otherwise : 'DenoiCompNetNVMS' object has no attribute 'forward_reconstruct'

    """
    This model perform an algorithm of conjugate gradient descent applied to the FCL
    """

    def forward_reconstruct(self, m_alpha, b, c, h, w):
        # We start the conjugate gradient descent for the batch x

        ###########################
        # -- Initialization (n = 0)
        ###########################
        # -- f(0)
        x = self.forward_maptoimage(m_alpha, [], b, c, h, w)
        #########
        # -- grad
        #########
        # -- Initial variation : we compute the tikhonov correction in the measurement space
        deltax = self.forward_maptoimage(m_alpha, x, b, c, h, w, correction=True)
        # -- "FCL-Gradient" calculation
        grad = (self.forward_postprocess(x + deltax, b, c, h, w) - x)
        # -- Inertie/momentum
        eta = -grad
        # -- Acceleration norm
        mod = torch.linalg.norm(grad,dim=(2,3)) ** 2

        for n in range(self.Niter):
            # -- Image & gradient update
            x = x + self.tau * eta
            deltax = self.forward_maptoimage(m_alpha, x, b, c, h, w, correction=True)
            grad =  (self.forward_postprocess(x + deltax, b, c, h, w) - x)
            ####################
            # -- Momentum update
            ####################
            mod_n = torch.linalg.norm(grad,dim=(2,3)) ** 2
            beta = torch.div(mod_n, mod)
            mod = mod_n.clone().detach()

            eta = -grad + torch.mul(beta.view(b * c, 1, 1, 1), eta)
            eta = eta.clone().detach()

        return x


    def forward_maptoimage(self, x, f, b, c, h, w, correction = False):
        # -- Pre-processing(Recombining positive and negatve values+normalisation)
        var = x[:, :, self.even_index] + x[:, :, self.uneven_index];
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];
        x = x / self.N0;
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b * c, 1, self.M));

        if correction:
            # -- correction in the measurement domain
            H1 = self.H1.to(x.device)
            x = x - torch.transpose(torch.matmul(H1, torch.reshape(f, (b * c, self.n ** 2, 1))), 1, 2)
        # --Denoising
        sigma = self.sigma.repeat(b * c, 1, 1).to(x.device)
        x = torch.mul(torch.div(sigma, sigma + var / (self.N0) ** 2), x);

        # --Projection to the image domain
        x = self.fc1(x);
        x = x.view(b * c, 1, h, w)
        return x


########################################################################################################################

########################################################################################################################

# 2. Define a custom Loss function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Creating custom loss function
# ---------------------------
# Just to make sure that all functions work the same way...   
# i.e., that they take the same number of arguments

class Weight_Decay_Loss(nn.Module):
    
    def __init__(self, loss):
        super(Weight_Decay_Loss,self).__init__()
        self.loss = loss;

    def forward(self,x,y, net):
        mse = self.loss(x,y);
        return mse


