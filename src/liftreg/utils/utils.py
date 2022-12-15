import os
import random

import mermaid.finite_differences as fdt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch


def set_seed_for_demo():
    """ reproduce the training demo"""
    seed = 2021
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_jacobi_map(map, spacing, crop_boundary=True, use_01=False):
        """
        compute determinant jacobi on transformatiomm map,  the coordinate should be canonical.

        :param map: the transformation map
        :param crop_boundary: if crop the boundary, then jacobi analysis would only analysis on cropped map
        :param use_01: infer the input map is in[0,1]  else is in [-1,1]
        :return: the sum of absolute value of  negative determinant jacobi, the num of negative determinant jacobi voxels
        """
        if type(map) == torch.Tensor:
            map = map.detach().cpu().numpy()
        span = 1.0 if use_01 else 2.0
        spacing = spacing * span  # the disp coorindate is [-1,1]
        fd = fdt.FD_np(spacing)
        a = fd.dXc(map[:, 0])
        b = fd.dYc(map[:, 0])
        c = fd.dZc(map[:, 0])
        d = fd.dXc(map[:, 1])
        e = fd.dYc(map[:, 1])
        f = fd.dZc(map[:, 1])
        g = fd.dXc(map[:, 2])
        h = fd.dYc(map[:, 2])
        i = fd.dZc(map[:, 2])
        jacobi_det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
        
        if crop_boundary:
            crop_range = 5
            jacobi_det_croped = jacobi_det[:, crop_range:-crop_range, crop_range:-crop_range, crop_range:-crop_range]
            jacobi_abs = - np.sum(jacobi_det_croped[jacobi_det_croped < 0.])
            jacobi_num = np.sum(jacobi_det_croped < 0.)
        jacobi_abs = - np.sum(jacobi_det[jacobi_det < 0.])
        jacobi_num = np.sum(jacobi_det < 0.)
        jacobi_abs_mean = jacobi_abs / map.shape[0]
        jacobi_num_mean = jacobi_num / map.shape[0]

        return jacobi_abs_mean, jacobi_num_mean

def save_deformations(phis, idx, path):
    '''
    Save batched deformation.
    :param phi: BxDxWxH numpy
    :param idx: a list of the ids added to the saved file name
    :param path: the root folder where the deformations should be saved
    '''
    phis = (phis + 1.) / 2.  # normalize the phi into 0, 1
    for i in range(phis.shape[0]):
        phi = nib.Nifti1Image(phis[i], np.eye(4))
        nib.save(phi, os.path.join(path, idx[i]) + '_phi.nii.gz')
        np.save(os.path.join(path, idx[i])+"_phi.npy", phis[i])


def save_fig_3D(imgs, path, idx, suffix, spacing=[1.,1.,1.], save_as_np=True):
        """
        save 3d output, i.e. moving, target and warped images
        :param imgs: BxCx
        :return:
        """
    
        if type(imgs) == torch.Tensor:
            imgs = imgs.detach().cpu().numpy()
        
        for i in range(imgs.shape[0]):
            appendix = idx[i] + suffix

            if save_as_np:
                saving_file_path = path + '/' + appendix + ".npy"
                np.save(saving_file_path, imgs[i,...])

            saving_file_path = path + '/' + appendix + ".nii.gz"
            output = sitk.GetImageFromArray(imgs[i, ...])
            output.SetSpacing(np.flipud(spacing))
            sitk.WriteImage(output, saving_file_path)

def sigmoid_decay(ep, static =5, k=5):
    """
    factor  decease with epoch, factor = k/(k + exp(ep / k))
    :param ep: cur epoch
    :param static: at the first #  epoch, the factor keep unchanged
    :param k: the decay factor
    :return:
    """
    static = static
    if ep < static:
        return float(1.)
    else:
        ep = ep - static
        factor =  k/(k + np.exp(ep / k))
        return float(factor)

def t2np(v):
    """
    Takes a torch array and returns it as a numpy array on the cpu

    :param v: torch array
    :return: numpy array
    """

    if type(v) == torch.Tensor:
        return v.detach().cpu().numpy()
    else:
        try:
            return v.cpu().numpy()
        except:
            return v

def lift_to_dimension(A,dim):
    """
    Creates a view of A of dimension dim (by adding dummy dimensions if necessary).
    Assumes a numpy array as input

    :param A: numpy array
    :param dim: desired dimension of view
    :return: returns view of A of appropriate dimension
    """

    current_dim = len(A.shape)
    if current_dim>dim:
        raise ValueError('Can only add dimensions, but not remove them')

    if current_dim==dim:
        return A
    else:
        return A.reshape([1]*(dim-current_dim)+list(A.shape))
