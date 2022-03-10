import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import numbers

class Conv2d_block(nn.Module):
    def __init__(self, in_chanel, out_chanel, kernel_size=3, stride=1, 
                 padding=1, has_BN = False):
        super(Conv2d_block, self).__init__()
        if has_BN:
            self.seq = nn.Sequential(
                nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, 
                               stride=stride, padding=padding),
                nn.BatchNorm2d(out_chanel),
                nn.ReLU(inplace=True)
            )
        else:
            self.seq = nn.Sequential(
                nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, 
                               stride=stride, padding=padding),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.seq(x)

class Deconv2d_block(nn.Module):
    def __init__(self, in_chanel, out_chanel, kernel_size=1, stride=1, 
                 padding=1, has_BN = False):
        super(Deconv2d_block, self).__init__()
        if has_BN:
            self.seq = nn.Sequential(
                nn.ConvTranspose2d(in_chanel, out_chanel, kernel_size=kernel_size, 
                                   stride=stride, padding=padding),
                nn.BatchNorm2d(out_chanel),
                nn.ReLU(inplace=True)
            )
        else:
            self.seq = nn.Sequential(
                nn.ConvTranspose2d(in_chanel, out_chanel, kernel_size=kernel_size, 
                                   stride=stride, padding=padding),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.seq(x)

class Conv3d_block(nn.Module):
    def __init__(self, in_chanel, out_chanel, kernel_size=3, stride=1, 
                 padding=1, has_BN = False):
        super(Conv3d_block, self).__init__()
        if has_BN:
            self.seq = nn.Sequential(
                nn.Conv3d(in_chanel, out_chanel, kernel_size=kernel_size, 
                          stride=stride, padding=padding),
                nn.BatchNorm3d(out_chanel),
                nn.ReLU(inplace=True)
            )
        else:
            self.seq = nn.Sequential(
                nn.Conv3d(in_chanel, out_chanel, kernel_size=kernel_size, 
                          stride=stride, padding=padding),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.seq(x)
        
class Deconv3d_block(nn.Module):
    def __init__(self, in_chanel, out_chanel, kernel_size=1, stride=1, 
                 padding=1, has_BN = False):
        super(Deconv3d_block, self).__init__()
        if has_BN:
            self.seq = nn.Sequential(
                nn.ConvTranspose3d(in_chanel, out_chanel, kernel_size=kernel_size, 
                                   stride=stride, padding=padding),
                nn.BatchNorm3d(out_chanel),
                nn.ReLU(inplace=True)
            )
        else:
            self.seq = nn.Sequential(
                nn.ConvTranspose3d(in_chanel, out_chanel, kernel_size=kernel_size, 
                                   stride=stride, padding=padding),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.seq(x)

class resize_conv3D(nn.Module):
    def __init__(self, out_dim, mode, in_chanel, out_chanel, kernel_size=3, stride=1,
                 padding=1, has_BN = False):
        super(resize_conv3D, self).__init__()
        if has_BN:
            self.seq = nn.Sequential(
                nn.Upsample(size=out_dim, mode=mode, align_corners=True),
                nn.Conv3d(in_chanel, out_chanel, kernel_size=kernel_size, 
                          stride=stride, padding=padding),
                nn.BatchNorm3d(out_chanel),
                nn.ReLU(inplace=True)
            )
        else:
            self.seq = nn.Sequential(
                nn.Upsample(size=out_dim, mode=mode, align_corners=True),
                nn.Conv3d(in_chanel, out_chanel, kernel_size=kernel_size, 
                          stride=stride, padding=padding),
                nn.ReLU(inplace=True)
            )
            
    def forward(self, x):
        return self.seq(x)


class res_block(nn.Module):
    def __init__(self, in_chanel, out_chanel):
        super(res_block, self).__init__()
        self.seq_1 = Conv2d_block(in_chanel, out_chanel, kernel_size=4, stride=2,
                                  padding=1, has_BN=True)

        self.seq_2 = Conv2d_block(out_chanel, out_chanel, kernel_size=3, stride=1,
                                  padding=1, has_BN=True)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        conv1 = self.seq_1(x)
        conv2 = self.seq_2(conv1)
        return self.relu(conv1 + conv2)

        
class trans_layer(nn.Module):
    def __init__(self, in_chanel, in_chanel_3d):
        super(trans_layer, self).__init__()
        self.seq_1 = nn.Sequential(
            nn.Conv2d(in_chanel, in_chanel, kernel_size=1, stride=1,
                      padding=0),
            nn.ReLU(inplace=True)
        )

        self.seq_2 = nn.Sequential(
            nn.ConvTranspose3d(in_chanel_3d, in_chanel_3d, kernel_size=1,
                               padding=0),
            nn.ReLU(inplace=True)
        )
        

        self.in_chanel_3d = in_chanel_3d
    
    def forward(self, x):
        feat_2d = self.seq_1(x)
        feat_3d = feat_2d.view(-1, self.in_chanel_3d, 4, 4, 4)
        # feat_3d_interp = F.interpolate(feat_3d, size=[5, 5, 5])
        
        feat_3d_1 = self.seq_2(feat_3d)
        return feat_3d_1

class proj_layer(nn.Module):
    def __init__(self, volume_spacing, resolution_scale, scan_range, proj_num, in_shape, out_shape, device):
        super(proj_layer, self).__init__()
        self.spacing = volume_spacing
        self.resolution_scale = resolution_scale
        self.sample_rate = [int(1), int(1), int(1)]
        self.out_shape = out_shape

        angle_half = scan_range/2.
        emitter_count = proj_num
        self.poses_scale = np.ndarray((emitter_count,3),dtype=np.float)
        self.poses_scale[:,1] = 3.
        self.poses_scale[:,0] = np.tan(np.linspace(-angle_half,angle_half,num=emitter_count)/180.*np.pi)*3.
        self.poses_scale[:,2] = np.linspace(-0.2,0.2, num = emitter_count)

        emi_poses = self.poses_scale*in_shape[1]
        proj_resolution = [int(in_shape[0] * self.resolution_scale),
                           int(in_shape[2] * self.resolution_scale)]

        # Calc projection grids once
        grids, self.dx = self._project_grid_multi(emi_poses, proj_resolution, in_shape[0:], device)
        self.grids = torch.flip(grids, [4])
    
    def forward(self, x):
        x_expand = torch.unsqueeze(x, 1)
        (p, d, h, w) = self.grids.shape[0:4]
        b = x_expand.shape[0]
        grids = torch.reshape(self.grids, (1,1,1,-1,3)).repeat(b, 1, 1, 1, 1)
        x_proj = torch.mul(torch.sum(F.grid_sample(x_expand, grids, align_corners = True).reshape((b, p, d, h, w)), dim=4), self.dx)

        # Resample to out shape
        x_proj_resampled = F.interpolate(x_proj, self.out_shape)

        return x_proj_resampled

    def _project_grid_multi(self, emi_pos, proj_resolution, obj_shape, device):
        # Axes definition: 0-axial, 1-coronal, 2-sagittal
        # sample_rate: sample count per pixel
        d, w, h = obj_shape
        (res_d, res_h) = proj_resolution
        sr_d, sr_w, sr_h = self.sample_rate

        # P0 - one point in each coronal plane of CT. We use the points at Y axies.
        # I0 - start point of each rays.
        # N - Normal vector of coronal plane.
        P0 = torch.mm(
            torch.linspace(0, w-1, sr_w*w, device=device).unsqueeze(1), 
            torch.tensor([[0., 1., 0.]]).to(device))
        I0 = torch.from_numpy(emi_pos).to(device).float().unsqueeze(1).unsqueeze(1)
        N = torch.tensor([0., 1., 0.], device=device)
        
        # Calculate direction vectors for each rays
        lin_x = torch.linspace(-res_d/2, res_d/2-1, steps=res_d*sr_d)
        lin_y = torch.linspace(-res_h/2, res_h/2-1, steps=res_h*sr_h)
        grid_x, grid_y = torch.meshgrid(lin_x, lin_y)
        I = torch.zeros((lin_x.shape[0], lin_y.shape[0], 3), device=device)
        I[:,:,0] = grid_x
        I[:,:,2] = grid_y
        I = torch.add(I,-I0)
        dx = torch.mul(I, 1./I[:,:,:,1:2])
        I = I/torch.norm(I, dim=3, keepdim=True)
        dx = torch.norm(dx*self.spacing.to(device).unsqueeze(0).unsqueeze(0), dim=3)

        # Define a line as I(t)=I0+t*I
        # Define a plane as (P-P0)*N=0, P is a vector of points on the plane
        # Thus at the intersection of the line and the plane, we have d=(P0-I0)*N/(I*N)
        # Then we can get the position of the intersection by I(t) = t*I + I0
        # T = torch.matmul(1./(torch.matmul(I,N)).unsqueeze(2), torch.matmul(P0-I0, N).unsqueeze(0))
        # grid = torch.add(torch.matmul(T.unsqueeze(3), I.unsqueeze(2)), I0)

        T = torch.matmul(1./(torch.matmul(I,N)).unsqueeze(3).unsqueeze(4), torch.matmul(P0-I0, N).unsqueeze(1).unsqueeze(1))
        grid = torch.add(torch.matmul(I.unsqueeze(4), T).permute(0,1,2,4,3), I0.unsqueeze(1))

        # Since grid_sample function accept input in range (-1,1)
        grid[:,:,:,:,0] = grid[:,:,:,:,0]/obj_shape[0]*2.0
        grid[:,:,:,:,1] = (grid[:,:,:,:,1]-0.)/obj_shape[1]*2.0 + -1.
        grid[:,:,:,:,2] = grid[:,:,:,:,2]/obj_shape[2]*2.0
        return grid, dx

class NCCLoss(torch.nn.Module):
    def __init__(self):
        super(NCCLoss, self).__init__()

    def forward(self, x, y):
        n_batch = x.shape[0]
        dim = len(x.shape[2:])
        input_shape = [x.shape[0], x.shape[1], -1]+[1]*dim
        x = x.view(*input_shape)
        y = y.view(*input_shape)
        xmean = x.mean(dim=2, keepdim=True)
        ymean = y.mean(dim=2, keepdim=True)
        x_m_mean = x-xmean
        y_m_mean = y-ymean
        nccSqr = (((x_m_mean)*(y_m_mean)).mean(dim=2)**2)/\
                (((x_m_mean)**2).mean(dim=2)*((y_m_mean)**2).mean(dim=2)+1e-12)
        nccSqr =nccSqr.mean(dim=1).sum()
        return (1.-nccSqr/n_batch)


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        fil = torch.tensor([[1,2,1],[2,4,2],[1,2,1]])
        self.x_filter = nn.Parameter(torch.zeros((3,3,3)).view(1,1,3,3,3), 
                                    requires_grad=False)
        self.x_filter[0,0,0,:,:]=fil
        self.x_filter[0,0,2,:,:]=-fil

        self.y_filter = nn.Parameter(torch.zeros((3,3,3)).view(1,1,3,3,3),
                                    requires_grad=False)
        self.y_filter[0,0,:,0,:]=fil
        self.y_filter[0,0,:,2,:]=-fil

        self.z_filter = nn.Parameter(torch.zeros((3,3,3)).view(1,1,3,3,3),
                                    requires_grad=False)
        self.z_filter[0,0,:,:,0]=fil
        self.z_filter[0,0,:,:,2]=-fil
    
    
    def forward(self, x, y):
        x_g_x = F.conv3d(x, self.x_filter, padding=1)
        x_g_y = F.conv3d(x, self.y_filter, padding=1)
        x_g_z = F.conv3d(x, self.z_filter, padding=1)

        y_g_x = F.conv3d(y, self.x_filter, padding=1)
        y_g_y = F.conv3d(y, self.y_filter, padding=1)
        y_g_z = F.conv3d(y, self.z_filter, padding=1)

        return F.mse_loss(x_g_x, y_g_x) + F.mse_loss(x_g_y, y_g_y) +F.mse_loss(x_g_z, y_g_z)

class convResBlock(nn.Module):
    """
    A convolutional block including conv, BN, nonliear activiation, residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=True, batchnorm=False, residual=False, nonlinear=nn.LeakyReLU(0.2), groups=1):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param nonlinear:
        """

        super(convResBlock, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.bn_1 = nn.BatchNorm3d(out_channels) if batchnorm else None
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias, groups=groups)
        self.bn_2 = nn.BatchNorm3d(out_channels) if batchnorm else None
        self.nonlinear = nonlinear
        if residual:
            self.residual = nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=bias)
        else:
            self.residual = None


    def forward(self, x):
        x_1 = self.conv_1(x)
        if self.bn_1:
            x_1 = self.bn_1(x_1)
        if self.nonlinear:
            x_1 = self.nonlinear(x_1)
        x_1 = self.conv_2(x_1)
        if self.residual:
            x_1 = self.residual(x) + x_1
        if self.bn_2:
            x_1 = self.bn_2(x_1)
        if self.nonlinear:
            x_1 = self.nonlinear(x_1)
        return x_1

class convBlock(nn.Module):
    """
    A convolutional block including conv, BN, nonliear activiation, residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=True, batchnorm=False, residual=False, nonlinear=nn.LeakyReLU(0.2)):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param nonlinear:
        """

        super(convBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels) if batchnorm else None
        self.nonlinear = nonlinear
        if residual:
            self.residual = nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=bias)
        else:
            self.residual = None


    def forward(self, x):
        x_1 = self.conv(x)
        if self.bn:
            x_1 = self.bn(x_1)
        if self.nonlinear:
            x_1 = self.nonlinear(x_1)
        if self.residual:
            x_1 = self.residual(x) + x_1
        return x_1
    
class convBlock2D(nn.Module):
    """
    A convolutional block including conv, BN, nonliear activiation, residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=True, batchnorm=False, residual=False, nonlinear=nn.LeakyReLU(0.2)):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param nonlinear:
        """

        super(convBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.nonlinear = nonlinear
        if residual:
            self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=bias)
        else:
            self.residual = None


    def forward(self, x):
        x_1 = self.conv(x)
        if self.bn:
            x_1 = self.bn(x_1)
        if self.nonlinear:
            x_1 = self.nonlinear(x_1)
        if self.residual:
            x_1 = self.residual(x) + x_1
        return x_1

class FullyConnectBlock(nn.Module):
    """
    A fully connect block including fully connect layer, nonliear activiation
    """
    def __init__(self, in_channels, out_channels, bias=True, nonlinear=nn.LeakyReLU(0.2)):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param nonlinear:
        """

        super(FullyConnectBlock, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.nonlinear = nonlinear

    def forward(self, x):
        x_1 = self.fc(x)
        if self.nonlinear:
            x_1 = self.nonlinear(x_1)
        return x_1

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=int(self.weight.shape[2]/2))