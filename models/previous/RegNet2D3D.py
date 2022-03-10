"""
registration network described in voxelmorph
An experimental pytorch implemetation, the official tensorflow please refers to https://github.com/voxelmorph/voxelmorph

An Unsupervised Learning Model for Deformable Medical Image Registration
Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
CVPR 2018. eprint arXiv:1802.02604

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018. eprint arXiv:1805.04605
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layers import convBlock
from utils.general import get_class
from utils.net_utils import Bilinear, gen_identity_map
from utils.utils import gen_affine_identity_matrix, gen_affine_map
from layers.layers import convBlock, GaussianSmoothing
from utils.sdct_projection_utils import backproj_grids_with_poses
from utils.sdct_projection_utils import calc_relative_atten_coef_cuda, forward_grids_with_poses


class model(nn.Module):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras reg_model
    """
    def __init__(self, img_sz, opt=None):
        super(model, self).__init__()
        
        enc_filters = [16, 32, 32, 32, 32]
        #dec_filters = [32, 32, 32, 8, 8]
        dec_filters = [32, 32, 32, 32, 16, 16]
        self.enc_filter = enc_filters
        self.dec_filter = dec_filters
        input_channel =2
        output_channel= 3
        self.input_channel = 2
        self.output_channel = 3
        self.img_sz = img_sz

        # Init affine net
        affine_class = opt[('affine_class', "", 'if set, then use pretrained affine.')] if opt is not None else ""
        if affine_class != "":
            self.using_affine_init = True
            self.affine = self.init_affine_net(opt)
            self.id_transform = None
            self.affine_identity = gen_affine_identity_matrix()
        else:
            self.using_affine_init = False
            self.id_transform = gen_identity_map(self.img_sz, 1.0)

        # Init flow net
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.bilinear = Bilinear(zero_boundary=True, using_scale=False)
        self.disp_bilinear = Bilinear(zero_boundary=True, using_scale=False)

        for i in range(len(enc_filters)):
            if i==0:
                self.encoders.append(convBlock(opt["drr_feature_num"]+1, enc_filters[i], stride=1, bias=True))
            else:
                self.encoders.append(convBlock(enc_filters[i-1], enc_filters[i], stride=2, bias=True))

        self.decoders.append(convBlock(enc_filters[-1], dec_filters[0], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[0] + enc_filters[3], dec_filters[1], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[1] + enc_filters[2], dec_filters[2], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[2] + enc_filters[1], dec_filters[3], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[3] + enc_filters[0], dec_filters[4], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[4], dec_filters[5],stride=1, bias=True))

        self.flow = nn.Conv3d(dec_filters[-1], output_channel, kernel_size=3, stride=1, padding=1, bias=True)
        
        torch.nn.init.normal_(self.flow.weight, mean=0., std=0.001)
        torch.nn.init.constant_(self.flow.bias, 0.)

    def forward(self, input):
        # Parse input
        moving = input['source']
        target = input['target']
        target_proj = input["target_proj"]
        if 'source_label' in input:
            moving_seg = input['source_label']
            target_seg = input['target_label']
            moving_cp = moving*moving_seg
            target_cp = target*target_seg
        else:
            moving_cp = moving
            target_cp = target

        # Lift 2D to 3D
        # Lift 2D to 3D
        batch_size, proj_num, proj_w, proj_h = target_proj.shape
        w, d, h = moving.shape[2:]
        target_poses = input['target_poses']
        with torch.no_grad():
            grids = backproj_grids_with_poses(target_poses.cpu().numpy(), moving.shape[2:], target_proj.shape[2:], device=moving.device).permute(0,1,3,4,5,2)

            target_volume = F.grid_sample(
                    target_proj.reshape(batch_size*proj_num, 1, proj_w, proj_h), 
                    grids.reshape(batch_size*proj_num, w*d, h, -1),
                    align_corners=True,
                    padding_mode="zeros").reshape(batch_size, proj_num, w, d, h).detach()


        x_enc_1 = self.encoders[0](torch.cat(
                        [moving_cp,
                         target_volume],
                        dim = 1
                        )  
                    )
        # del input
        x_enc_2 = self.encoders[1](x_enc_1)
        x_enc_3 = self.encoders[2](x_enc_2)
        x_enc_4 = self.encoders[3](x_enc_3)
        x_enc_5 = self.encoders[4](x_enc_4)

        x = self.decoders[0](x_enc_5)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((x, x_enc_4),dim=1)
        x = self.decoders[1](x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((x, x_enc_3), dim=1)
        x = self.decoders[2](x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((x, x_enc_2), dim=1)
        x = self.decoders[3](x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = torch.cat((x, x_enc_1), dim=1)
        x = self.decoders[4](x)
        x = self.decoders[5](x)

        disp_field = self.flow(x)

        # Compute inverse flow
        disp_inv = torch.zeros_like(disp_field, device=self.id_transform.device)
        for i in range(7):
            disp_inv = -1*self.disp_bilinear(disp_field, self.id_transform + disp_inv)

        deform_field = disp_field + self.id_transform

        moving_origin = input["original_source"]

        warped_source = self.bilinear(moving_origin, deform_field)
        warped_source_inv = self.bilinear(warped_source, self.id_transform + disp_inv)

        # Compute proj
        spacing = input['spacing']
        receptor_size = [*target_proj.shape[2:]]

        # compute drr of target image
        def compute_proj(img, grids, dx):
            p = grids.shape[1]
            proj_target_container = []
            for i in range(p):
                proj_target_container.append(
                    torch.mul(torch.sum(F.grid_sample(img, grids[:,i], align_corners = True, padding_mode="zeros"), dim=4), dx[i:i+1])
                )
            
            proj = torch.cat(proj_target_container, dim=1).float()
            proj = (proj - proj.min())/(proj.max() - proj.min())*2.-1.
            return proj


        forward_proj_grids, proj_dx = forward_grids_with_poses(target_poses[0].cpu().numpy(), 
                                                        spacing[0].cpu().numpy(), 
                                                        self.img_sz, 
                                                        moving.device,
                                                        receptor_size=receptor_size)

        # change dx from mm unit to cm unit
        proj_dx *= 0.1
        forward_proj_grids = forward_proj_grids.unsqueeze(0)
        warped_proj = compute_proj(warped_source, forward_proj_grids.expand(target.shape[0], -1, -1, -1, -1, -1), proj_dx)
        warped_inv_proj = compute_proj(warped_source_inv, forward_proj_grids.expand(target.shape[0], -1, -1, -1, -1, -1), proj_dx)
        source_proj = compute_proj(moving_origin, forward_proj_grids.expand(target.shape[0], -1, -1, -1, -1, -1), proj_dx)

        model_output = {"warped": warped_source,
                        "phi": deform_field,
                        "params": [disp_field, disp_inv],
                        "warped_proj": warped_proj,
                        "warped_proj_inv": warped_inv_proj,
                        "source_proj": source_proj,
                        "target_proj": target_proj}
        return model_output
    
    def init_affine_net(self, opt):
        affine_class = opt[('affine_class', "", 'if set, then use pretrained affine.')]
        if affine_class != "":
            affine_class = get_class(affine_class)
        affine = affine_class(self.img_sz, opt)

        # Load affine weight
        affine_weight = opt[('affine_init_path','',"the path of pretrained affine model")]
        if affine_weight != "":
            affine_weight = torch.load(affine_weight, map_location=torch.device('cpu'))
            affine.load_state_dict(affine_weight["state_dict"])
        
        # Affine should be freezed
        for p in affine.parameters():
            p.requires_grad = False
        return affine

    def get_extra_to_plot(self):
        return None, None
    
    def get_disp(self):
        return None, ""

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()


