import torch
import torch.nn.functional as F
import numpy as np
from numpy import genfromtxt

def calc_relative_atten_coef(img):
    new_img = img.astype(np.float32).copy()
    new_img[new_img<-1000] = -1000
    return (new_img+1000.)/1000.*0.2 # Assume the linear attenuation coefficient for water is 0.2

def calc_relative_atten_coef_cuda(img):
    img[img<-1000] = -1000
    return (img+1000.)/1000.*0.2

def project_grid_multi(emi_pos, resolution, sample_rate, obj_shape, spacing, device, dtype):
        # Axes definition: 0-axial, 1-coronal, 2-sagittal
        # sample_rate: sample count per pixel
        d, w, h = obj_shape
        (res_d, res_h) = resolution
        sr_d, sr_w, sr_h = sample_rate

        # P0 - one point in each coronal plane of CT. We use the points at Y axies.
        # I0 - start point of each rays.
        # N - Normal vector of coronal plane.
        P0 = torch.mm(
            torch.linspace(0, w-1, sr_w*w, device=device, dtype=dtype).unsqueeze(1), 
            torch.tensor([[0., 1., 0.]], device=device, dtype=dtype))
        I0 = torch.from_numpy(emi_pos).to(device).unsqueeze(1).unsqueeze(1).type(dtype)
        N = torch.tensor([0., 1., 0.], device=device, dtype=dtype)
        
        # Calculate direction vectors for each rays
        lin_x = torch.linspace(-res_d/2, res_d/2-1, steps=res_d*sr_d)
        lin_y = torch.linspace(-res_h/2, res_h/2-1, steps=res_h*sr_h)
        grid_x, grid_y = torch.meshgrid(lin_x, lin_y)
        I = torch.zeros((lin_x.shape[0], lin_y.shape[0], 3), device=device, dtype=dtype)
        I[:,:,0] = grid_x
        I[:,:,2] = grid_y
        I = torch.add(I,-I0)
        dx = torch.mul(I, 1./I[:,:,:,1:2])
        I = I/torch.norm(I, dim=3, keepdim=True)
        dx = torch.norm(dx*spacing.to(device).unsqueeze(0).unsqueeze(0), dim=3)

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
        grid[:,:,:,:,1] = (grid[:,:,:,:,1]-0.)/(obj_shape[1]-1.)*2.0 + -1.
        grid[:,:,:,:,2] = grid[:,:,:,:,2]/obj_shape[2]*2.0
        return grid, dx

def calculate_projection(img, poses, resolution, sample_rate, spacing, device):
    # Need to pay attention of the coordinate directions. 
    # The receptor is the XZ plane and the origin is at the center of the receptor.
    # The Y axis points at the emittor from the receptor.
    # The XYZ coordinate uses left hand coordinate system.

    # Since this is a cone beam radiation, one should pay attention to the image coordinate.
    # For example, for X-ray image the patient usually faces the receptor and the emitter comes from
    # the back of the patient. Thus, to generate X-ray image, the input image should in SAL.
    # However, for sDCT projection, the input image should in SPR.

    spacing = torch.tensor(spacing).to(device)
    I0 = torch.from_numpy(img).to(device)
    I0 = I0.unsqueeze(0).unsqueeze(0)
    # resolution = [int(I0.shape[2] * resolution_scale),
    #               int(I0.shape[4] * resolution_scale)]
    grids, dx = project_grid_multi(poses, resolution, sample_rate, I0.shape[2:], spacing, I0.device, I0.dtype)
    grids = torch.flip(grids, [4])
    (p, d, h, w) = grids.shape[0:4]
    b = I0.shape[0]
    grids = torch.reshape(grids, (1,1,1,-1,3))
    # dx = dx.unsqueeze(1).unsqueeze(1)
    I0_proj = torch.mul(torch.sum(F.grid_sample(I0, grids, align_corners = True).reshape((b, p, d, h, w)), dim=4), dx).float()

    # Since dx is in mm unit while linear attenuation coefficients are in cm unit.
    # Multiply the final result to reflect the unit transform.
    I0_proj *= 0.1

    # projections = torch.zeros((1, poses.shape[0], resolution[0]*sample_rate[0], resolution[1]*sample_rate[2])).to(device)
    # for i in range(poses.shape[0]):
    #     grid, dx = project_grid(I1, poses[i], (resolution[0], resolution[1]), sample_rate, I1.shape[2:], spacing)
    #     grid = torch.flip(grid,[3])
    #     dx = dx.unsqueeze(0).unsqueeze(0)
    #     projections[0, i] = torch.mul(torch.sum(F.grid_sample(I1, grid.unsqueeze(0), align_corners=False), dim=4), dx)[0, 0]
    #     # np.save("./log/grids_sim_matrix_"+str(i)+".npy", grid.cpu().numpy())
    #     del grid
    #     torch.cuda.empty_cache()

    proj = I0_proj[0].detach().cpu().numpy()
    del I0_proj, grids, I0, dx, spacing
    torch.cuda.empty_cache()
    return proj

def calculate_projection_multiB(img, poses_scale, resolution, sample_rate, spacing, device):
    
    poses = poses_scale*img.shape[1]
    spacing = torch.tensor(spacing).to(device)
    I0 = torch.from_numpy(img).to(device)
    I0 = I0.unsqueeze(0).unsqueeze(0)
    grids, dx = project_grid_multi(poses, resolution, sample_rate, I0.shape[2:], spacing, I0.device, I0.dtype)
    grids = torch.flip(grids, [4])
    (p, d, h, w) = grids.shape[0:4]
    b = I0.shape[0] 

    # split the grid_sample to batch so that the computation can be fit into memory
    points_per_batch = 10000

    # The batch number can be computed by int(total_points - 1)/points_per_batch + 1
    b_num = int(d*h/points_per_batch)+1
    b_num = 1

    grids = grids.reshape((1, p, d*h, w, 3))
    print(dx.shape)

    results = []
    for i in range(b_num):
        # If we are in the last batch, the points in this batch might not equal to points_per_batch
        points_in_batch = points_per_batch if i < b_num-1 else w - i * points_per_batch
        grids_in_batch = grids[:,:,i*points_per_batch::i*points_per_batch+points_in_batch]
        results.append(torch.mul(torch.sum(F.grid_sample(I0, grids_in_batch, align_corners = True), dim=4), dx[:,:,i*points_per_batch::i*points_per_batch+points_in_batch]).float())

        
    I0_proj = torch.cat(results, dim=3).reshape(b, 100, 100, h)

    proj = I0_proj[0].detach().cpu().numpy()
    del I0_proj, grids, I0, dx, spacing
    torch.cuda.empty_cache()
    return proj

def calculate_projection_wraper(img_3d, scan_range, proj_num, spacing, receptor_size=None):
    angle_half = scan_range/2.
    emitter_count = proj_num
    poses_scale = np.ndarray((emitter_count,3),dtype=np.float)
    poses_scale[:,1] = 3.5
    poses_scale[:,0] = np.tan(np.linspace(-angle_half,angle_half,num=emitter_count)/180.*np.pi)*3.
    poses_scale[:,2] = np.linspace(-0.2,0.2, num = emitter_count)

    if receptor_size is not None:
        resolution = list(receptor_size)
    else:
        resolution_scale = 1.5
        resolution = [int(img_3d.shape[0] * resolution_scale),
                      int(img_3d.shape[2] * resolution_scale)]
    sample_rate = [int(1), int(1), int(1)]

    device = torch.device("cuda")
    poses = poses_scale*img_3d.shape[1]

    img_proj = calculate_projection(img_3d, poses, resolution,
                                        sample_rate, spacing, device)
    return img_proj, poses

def calculate_projection_wraper_with_geo_csv_file(img_3d, img_spacing, geo_path, receptor_size=None):
    geo_txt = genfromtxt(geo_path, delimiter=',')[1:]
    poses = geo_txt/img_spacing  # Convert physical position to current world spacing

    if receptor_size is not None:
        resolution = list(receptor_size)
    else:
        resolution_scale = 1.5
        resolution = [int(img_3d.shape[0] * resolution_scale),
                      int(img_3d.shape[2] * resolution_scale)]
    sample_rate = [int(1), int(1), int(1)]

    device = torch.device("cuda")

    img_proj = calculate_projection(img_3d, poses, resolution,
                                        sample_rate, img_spacing, device)
    return img_proj, poses

def backproj_grids(scan_range, proj_num, img_shape, proj_shape, device=torch.device("cuda")):
    angle_half = scan_range/2.
    emitter_count = proj_num
    poses_scale = np.ndarray((emitter_count,3),dtype=np.float)
    poses_scale[:,1] = 3.
    poses_scale[:,0] = np.tan(np.linspace(-angle_half,angle_half,num=emitter_count)/180.*np.pi)*3.
    poses_scale[:,2] = np.linspace(-0.2,0.2, num = emitter_count)
 
    d, w, h = img_shape
    proj_w, proj_h = proj_shape
    x = torch.linspace(-d/2, d/2-1, d, device=device)
    y = torch.linspace(w-1, 0, w, device=device)
    z = torch.linspace(-h/2, h/2-1, h, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)

    poses = torch.from_numpy(poses_scale*w).to(device)
    scale = poses[:, 1:2]/(poses[:, 1:2] - y)  # dim: 4xW
    trans = poses[:, 0::2, None] * (-y/(poses[:, 1:2] - y)).reshape(proj_num, 1, w)  # dim: 4x2xW
    grids = torch.cat((grid_x[None, :], grid_z[None, :]), dim=0).unsqueeze(0) # dim: 1x2xDxWxH
    grids = torch.mul(scale.reshape(proj_num, 1, 1, w, 1), grids) + trans.reshape(proj_num, 2, 1, w, 1)
    grids[:, 0] = grids[:, 0]/proj_w*2.0
    grids[:, 1] = grids[:, 1]/proj_h*2.0
    
    return grids.flip(1)

def forward_grids(scan_range, proj_num, spacing, img_shape, device=torch.device("cpu"), receptor_size=None):
    angle_half = scan_range/2.
    emitter_count = proj_num
    poses_scale = np.ndarray((emitter_count,3),dtype=np.float)
    poses_scale[:,1] = 3.
    poses_scale[:,0] = np.tan(np.linspace(-angle_half,angle_half,num=emitter_count)/180.*np.pi)*3.
    poses_scale[:,2] = np.linspace(-0.2,0.2, num = emitter_count)

    if receptor_size is not None:
        resolution = list(receptor_size)
    else:
        resolution_scale = 1.5
        resolution = [int(img_shape[0] * resolution_scale),
                      int(img_shape[2] * resolution_scale)]
    sample_rate = [int(1), int(1), int(1)]

    poses = poses_scale*img_shape[1]
    spacing = torch.tensor(spacing).to(device)
    grids, dx = project_grid_multi(poses, resolution, sample_rate, img_shape, spacing, device, torch.float)
    grids = torch.flip(grids, [4])
    
    return grids, dx

def backproj_grids_with_poses(poses, img_shape, proj_shape, device=torch.device("cpu")):
    d, w, h = img_shape
    _, proj_num, _ = poses.shape
    proj_w, proj_h = proj_shape
    x = torch.linspace(-d/2, d/2-1, d, device=device)
    y = torch.linspace(w-1, 0, w, device=device)
    z = torch.linspace(-h/2, h/2-1, h, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)

    

    poses = torch.from_numpy(poses).to(device).unsqueeze(3).unsqueeze(3).unsqueeze(3)
    scale = poses[:, :, 1:2]/(poses[:, :, 1:2] - grid_y)  # dim: BxPx1xDxWxH
    grids = torch.cat((grid_x[None,:], grid_z[None,:]), dim=0).unsqueeze(0)
    grids = grids - poses[:, :, ::2] # dim: Bx1x2xDxWxH
    grids = torch.mul(grids, scale) + poses[:, :, ::2]

    # trans = poses[:, 0::2, None] * (-y/(poses[:, 1:2] - y)).reshape(proj_num, 1, w)  # dim: 4x2xW
    # grids = torch.cat((grid_x[None, :], grid_z[None, :]), dim=0).unsqueeze(0) # dim: 1x2xDxWxH
    # grids = torch.mul(scale.reshape(proj_num, 1, 1, w, 1), grids) + trans.reshape(proj_num, 2, 1, w, 1)
    grids[:, :, 0] = grids[:, :, 0]/proj_w*2.0
    grids[:, :, 1] = grids[:, :, 1]/proj_h*2.0
    
    return grids.flip(2)

def forward_grids_with_poses(poses, spacing, img_shape, device=torch.device("cpu"), receptor_size=None):
    sample_rate = [int(1), int(1), int(1)]

    spacing = torch.tensor(spacing).to(device)
    if receptor_size is not None:
        resolution = (receptor_size[0], receptor_size[1])
    else:
        resolution_scale = 1.5
        resolution = [int(img_shape[0] * resolution_scale),
                      int(img_shape[2] * resolution_scale)]
    grids, dx = project_grid_multi(poses, resolution, sample_rate, img_shape, spacing, device, torch.float)
    grids = torch.flip(grids, [4])
    
    return grids, dx