import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layers import convBlock, FullyConnectBlock, GaussianSmoothing
from utils.net_utils import Bilinear, gen_identity_map
import numpy as np
from utils.sdct_projection_utils import backproj_grids_with_poses


class model(nn.Module):
    """
    Estimating the eigen value of pre-built subspace of the displacement.

    :param img_sz: Voxel shape.
    :param opt: setting for the network.
    """
    def __init__(self, img_sz, opt=None):
        super(model, self).__init__()
        
        enc_filters = [16, 32, 32, 32, 32, 32]
        self.input_channel = 2
        self.output_channel = 3
        self.img_sz = img_sz
        self.gaussian_smooth = GaussianSmoothing(4, 8, 2, dim=2)

        # Init flow net
        self.encoders = nn.ModuleList()
        self.bilinear = Bilinear(zero_boundary=True, using_scale=True)
        for i in range(len(enc_filters)):
            if i==0:
                self.encoders.append(convBlock(opt["drr_feature_num"]+1, enc_filters[i], stride=1, bias=True))
            else:
                self.encoders.append(convBlock(enc_filters[i-1], enc_filters[i], stride=2, bias=True))
        self.encoders.append(nn.Sequential(
            nn.Flatten(),
            FullyConnectBlock(32*5*5*5, 800),
            FullyConnectBlock(800, 256),
            FullyConnectBlock(256, opt["latent_dim"], nonlinear=None)
        ))

        # Load pca components
        self.pca_vectors = torch.from_numpy(np.load(f"{opt['pca_path']}/pca_vectors.npy").T).float().cuda()
        self.pca_mean = torch.from_numpy(np.load(f"{opt['pca_path']}/pca_mean.npy")).float().cuda()

        self.id_transform = gen_identity_map(self.img_sz, 1.0)
        self.backward_proj_grids = None


    def forward(self, input):
        # Parse input
        moving = input['source']
        target = input['target']
        target_proj = input["target_proj"]
        if 'source_label' in input:
            moving_seg = input['source_label']
            target_seg = input['target_label']
            moving_cp = (moving+1)*moving_seg-1
            target_cp = (target+1)*target_seg-1
        else:
            moving_cp = moving
            target_cp = target
        
        B,_,D,W,H = moving.shape
        target_poses = input['target_poses']

        coefs, disp_field = self._estimate_flow(moving, target_proj, target_poses)
        
        deform_field = disp_field + self.id_transform
        warped_source = self.bilinear(moving_cp, deform_field)

        model_output = {"warped": warped_source,
                        "phi": deform_field,
                        "params": disp_field,
                        "target": target_cp,
                        "pca_coefs": coefs,
                        "target_proj":target_proj,
                        "warped_proj":target_proj}
        return model_output
    
    def _estimate_flow(self, moving, target_proj, poses):
        batch_size, proj_num, proj_w, proj_h = target_proj.shape
        w, d, h = moving.shape[2:]
        B,_,D,W,H = moving.shape

        if self.backward_proj_grids is None:
            with torch.no_grad():
                self.backward_proj_grids = backproj_grids_with_poses(poses[0:1].cpu().numpy(), moving.shape[2:], target_proj.shape[2:], device=moving.device).permute(0,1,3,4,5,2)

        target_volume = F.grid_sample(
                target_proj.reshape(batch_size*proj_num, 1, proj_w, proj_h), 
                self.backward_proj_grids.expand(batch_size, -1, -1, -1, -1, -1).reshape(batch_size*proj_num, w*d, h, -1),
                align_corners=True,
                padding_mode="zeros").reshape(batch_size, proj_num, w, d, h).detach()

        x = torch.cat([moving,
                         target_volume],
                        dim = 1
                        )
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
        
        disp_field = F.linear(x, self.pca_vectors, self.pca_mean).reshape(B, 3, D, W, H)

        return x, disp_field

    def get_extra_to_plot(self):
        return None, None
    
    def get_disp(self):
        return None, ""



