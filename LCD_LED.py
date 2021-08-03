# -- LED Lamp DATA (Loading the Compressed Data)

import numpy as np
import warnings

# -- Pytorch tools
import torch

# -- Data load
import scipy.io as sio

def read_mat_data_index(expe_data, nflip, lambda_i=548):
    F_pos = sio.loadmat(expe_data + "_{}_100_pos_data.mat".format(nflip))
    F_neg = sio.loadmat(expe_data + "_{}_100_neg_data.mat".format(nflip))
    F_spectro = F_pos["spec"][0][0][0]
    F_spectro = F_spectro[0, :]
    lambda_indices = np.where(np.abs(F_spectro - lambda_i) < 1)
    num_channel = lambda_indices[0][0]
    F_data_pos = F_pos["F_WT_lambda_pos"]
    F_data_neg = F_neg["F_WT_lambda_neg"]
    F_pos = F_data_pos[:, :, num_channel]
    F_neg = F_data_neg[:, :, num_channel]
    if (2 ** 16 - 1 in F_pos) or (2 ** 16 - 1 in F_neg):
        warnings.warn("Warning, Saturation!", UserWarning)
    F_pos = F_pos.astype("int64")
    F_neg = F_neg.astype("int64")
    return F_pos, F_neg


def load_data_list_index(expe_data, nflip, CR, K, Perm, img_size, num_channel=548):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    even_index = range(0, 2 * CR, 2)
    odd_index = range(1, 2 * CR, 2)
    m_list = []
    for i in range(len(nflip)):
        F_pos, F_neg = read_mat_data_index(expe_data[i], nflip[i], num_channel)
        F_pos = F_pos
        F_neg = F_neg
        f_pos = np.reshape(F_pos, (img_size ** 2, 1))
        f_neg = np.reshape(F_neg, (img_size ** 2, 1))
        f_re_pos = np.dot(Perm, f_pos)
        f_re_neg = np.dot(Perm, f_neg)
        m = np.zeros((2 * CR, 1))
        m[even_index] = f_re_pos[:CR]
        m[odd_index] = f_re_neg[:CR]
        m = torch.Tensor(m)
        m = m.view(1, 1, 2 * CR)
        m = m.to(device)
        m_list.append(m)
    return m_list


def raw_ground_truth_list_index(expe_data, nflip, H, img_size, num_channel=548):
    gt_list = [];
    for i in range(len(nflip)):
        F_pos, F_neg = read_mat_data_index(expe_data[i], nflip[i], num_channel)
        f_pos = np.reshape(F_pos, (img_size ** 2, 1))
        f_neg = np.reshape(F_neg, (img_size ** 2, 1))
        Gt = np.reshape((1 / img_size) * np.dot(H, f_pos - f_neg), (img_size, img_size))
        gt_list.append(Gt)
    return gt_list


def batch_flipud(vid):
    outs = vid
    for i in range(vid.shape[1]):
        outs[0, i, 0, :, :] = np.flipud(vid[0, i, 0, :, :])
    return outs