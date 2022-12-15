import os

import numpy as np
import torch
from torch.nn import Module

dim=3

class Bilinear(Module):
    """
   Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
   """

    def __init__(self, zero_boundary=False, using_scale=True, mode="bilinear"):
        """
        Constructor

        :param ndim: (int) spatial transformation of the transform
        """
        super(Bilinear, self).__init__()
        self.zero_boundary = 'zeros' if zero_boundary else 'border'
        self.using_scale = using_scale
        self.mode = mode
        """ scale [-1,1] image intensity into [0,1], this is due to the zero boundary condition we may use here """

    def forward_stn(self, input1, input2):
        input2_ordered = torch.zeros_like(input2)
        input2_ordered[:, 0, ...] = input2[:, 2, ...]
        input2_ordered[:, 1, ...] = input2[:, 1, ...]
        input2_ordered[:, 2, ...] = input2[:, 0, ...]

        output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 4, 1]),
                                                 padding_mode=self.zero_boundary,
                                                 mode=self.mode,
                                                 align_corners=True)
        # output = torch.nn.functional.grid_sample(input1, input2.permute([0, 2, 3, 4, 1]),
        #                                          padding_mode=self.zero_boundary)
        return output

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform

        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """
        if self.using_scale:

            output = self.forward_stn((input1 + 1) / 2, input2)
            # print(STNVal(output, ini=-1).sum())
            return output * 2 - 1
        else:
            output = self.forward_stn(input1, input2)
            # print(STNVal(output, ini=-1).sum())
            return output


def identity_map(sz, dtype= np.float32):
    """
    Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0: sz[0]]
    elif dim == 2:
        id = np.mgrid[0: sz[0], 0: sz[1]]
    elif dim == 3:
        # id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        id = np.mgrid[0: sz[0], 0:sz[1], 0: sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')
    id = np.array(id.astype(dtype))
    if dim == 1:
        id = id.reshape(1, sz[0])  # add a dummy first index
    spacing = 1./ (np.array(sz)-1)

    for d in range(dim):
        id[d] *= spacing[d]
        id[d] = id[d]*2 - 1

    return torch.from_numpy(id.astype(np.float32)).cuda()


def not_normalized_identity_map(sz):
    """
    Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0: sz[0]]
    elif dim == 2:
        id = np.mgrid[0: sz[0], 0: sz[1]]
    elif dim == 3:
        # id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        id = np.mgrid[0: sz[0], 0:sz[1], 0: sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')
    # id= id*2-1
    return torch.from_numpy(id.astype(np.float32)).cuda()


def gen_identity_map(img_sz, resize_factor=1.,normalized=True):
    """
    given displacement field,  add displacement on grid field  todo  now keep for reproduce  this function will be disabled in the next release, replaced by spacing version
    """
    if isinstance(resize_factor, list):
        img_sz = [int(img_sz[i] * resize_factor[i]) for i in range(dim)]
    else:
        img_sz = [int(img_sz[i] * resize_factor) for i in range(dim)]
    if normalized:
        grid = identity_map(img_sz)
    else:
        grid = not_normalized_identity_map(img_sz)
    return grid

def resume_train(model_path, model, optimizer, lr_scheduler=None):
    """
    resume the training from checkpoint
    :param model_path: the checkpoint path
    :param model: the model to be set
    :param optimizer: the optimizer to be set
    :return:
    """
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')  # {'cuda:'+str(old_gpu):'cuda:'+str(cur_gpu)})
        start_epoch = 0
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print("the started epoch now is {}".format(start_epoch))
        else:
            start_epoch = 0
        
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        else:
            phases = ['train', 'val', 'debug']
            global_step = {x: 0 for x in phases}
        
        try:
            model.load_state_dict(checkpoint['state_dict'])
            print("=> succeed load model '{}'".format(model_path))
        except:
            ############### TODO  Currently not compatabile to enemble network ###############
            print("Warning !!! Meet error is reading the whole model, now try to read the part")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(" The incomplelet model is succeed load from '{}'".format(model_path))

        if 'optimizer' in checkpoint:
            if not isinstance(optimizer, tuple):
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
                    print("=> succeed load optimizer '{}'".format(model_path))
                    optimizer.zero_grad()
                except:
                    print("Warning !!! Meet error during loading the optimize, not externaly initialized")
        
        if 'lr_scheduler' in checkpoint and lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):
                state_dict = checkpoint['lr_scheduler']
                lr_scheduler.load_state_dict(state_dict)
                # step_size = checkpoint["lr_scheduler"].step_size
                # gamma = checkpoint["lr_scheduler"].gamma
                # last_epoch = checkpoint["lr_scheduler"].last_epoch
                # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch=last_epoch)
            else:
                # For some of the lr_scheduler, there is no state_dict() function defined.
                state_dict = getattr(checkpoint['lr_scheduler'], "state_dict", None)
                if state_dict is None:
                    state_dict = checkpoint['lr_scheduler']
                lr_scheduler.load_state_dict(state_dict)
        else:
            print("Warning !! lr_scheduler is not successfully loaded.")

        return start_epoch, global_step
    else:
        print("=> no checkpoint found at '{}'".format(model_path))


# get_test_model = resume_train


def save_model(epoch, network, global_step, save_path, prefix, is_best=False):
    if isinstance(network.optimizer, tuple):
        # for multi-optimizer cases
        optimizer_state = []
        for term in network.optimizer:
            optimizer_state.append(term.state_dict())
        optimizer_state = tuple(optimizer_state)
    else:
        optimizer_state = network.optimizer.state_dict()

    save_checkpoint({'epoch': epoch,
                     'state_dict': network.model.state_dict(),
                     'optimizer': optimizer_state,
                     'global_step': global_step,
                     'lr_scheduler':network.lr_scheduler.state_dict()}, is_best, save_path, prefix)


def save_checkpoint(state, is_best, save_path, prefix, filename='checkpoint'):
    """
    save checkpoint during training
    'epoch': epoch,'
    :param state_dict': {'epoch': epoch,'state_dict':  model.network.state_dict(),'optimizer': optimizer_state,
                  'best_score': best_score, 'global_step':global_step}
    :param is_best: if is the best model
    :param path: path to save the checkpoint
    :param prefix: prefix to add before the fname
    :param filename: filename
    :return:
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    prefix_save = os.path.join(save_path, prefix)
    name = '_'.join([prefix_save, filename]) + '.pth.tar'
    
    if is_best:
        torch.save(state, save_path + '/model_best.pth.tar')
    else:
        torch.save(state, name)
