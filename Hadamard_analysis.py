# -- packages
import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib import animation

from spyrit.learning.model_Had_DCAN import * # models

from pathlib import Path

############################
# -- Undersampling framework
############################

# image test size
img_size = 64
# Number of parameters to estimate
n = img_size ** 2

#####################################################
# -- Precomputed data (Average and covariance matrix)
#####################################################
# -- Path to precomputed data (Average and covariance matrix -- for model)

precompute_root = Path('/home/licho/Documentos/Stage/Codes/Test/')
Cov = np.load(precompute_root / "Cov_{}x{}.npy".format(img_size, img_size))

# -- Hadamard Matrix definition (fht hadamard transform needs to be normalized)
H = img_size * Hadamard_Transform_Matrix(img_size)
Var = Cov2Var(Cov)
Perm = Permutation_Matrix(Var)
Pmat = np.dot(Perm, H)
# -- Hadamard Matrix definition (fht hadamard transform needs to be normalized)
H = img_size * Hadamard_Transform_Matrix(img_size)

# Spectral Hadamard information
# W, V = LA.eig(H)
# U, S, VH = LA.svd(H)

# Pattern generaation
path = '/home/licho/anaconda3/lib/python3.8/site-packages/spyrit/'
Lena = plt.imread(path + "8-bit-256-x-256-Grayscale-Lena-Image.png")
Lena = Lena[:,:,0]
pattern = Lena[100:164,100:164] # np.random.uniform(low=0,high=1,size=(img_size,img_size))

# Measurement of Hadamard coefficients
# Signal reshape to parameter space
f = np.reshape(pattern, (n, 1))
# full coefficients adquisition
m = np.matmul(Pmat,f)

###################################
# --Inverse problem (Undersampling)
###################################
fig, axes = plt.subplots(figsize=(7,7))
plt.title("Pattern reconstructed with size =%i" % img_size)
plt.axis('off')
rows = 1
cols = 2

fig.add_subplot(rows,cols,1)
im = plt.imshow(np.zeros((img_size, img_size)), cmap='gray', vmin=0, vmax=1)

fig.add_subplot(rows,cols,2)
GT = plt.imshow(pattern, cmap='gray')

def init():
    im.set_data(np.zeros((img_size, img_size)))
    GT.set_data(pattern)

def animate(i):
    # Number of Hadamard measurements
    HT_star = Pmat.T[:, :i]
    m_star = m[:i]
    f_star = np.matmul(HT_star / n, m_star)
    pattern_star = np.reshape(f_star, (img_size, img_size))

    im.set_data(pattern_star)
    plt.title("Pattern reconstructed, number of Hadamard coefficients : %.i" % i)
    GT.set_data(pattern)

    return im, GT

# - interval : time transition
# - frames : number of images

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n + 1,
                               interval=1e-5)

fig.tight_layout()
plt.show()

""" 
fig.add_subplot(rows,cols,2)
a=np.empty((n+1,1))
am,=plt.plot(np.arange(n+1), a)
plt.ylim(0, 3)

am.set_ydata(np.empty((n+1, 1)))

# animate lines
a = am.get_ydata()
a[i] = LA.norm(f - f_star) 
am.set_ydata(a)
am.set_label("Pattern reconstructed, error level : %.3e" % LA.norm(f - f_star))
"""
