import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import liftreg.utils.module_parameters as pars
from liftreg.utils.metrics import get_multi_metric
from liftreg.utils.net_utils import Bilinear

from tools.preprocessing import COPD_spacing

parser = argparse.ArgumentParser(description='Show registration result')
parser.add_argument('--data_path', '-d', required=False, type=str,
                    default='',help='the name of the data related task (like subsampling)')
parser.add_argument('--setting', '-s', metavar='SETTING', default='',
                    help='setting')

def readPoint(f_path):
    """
    :param f_path: the path to the file containing the position of points.
    Points are deliminated by '\n' and X,Y,Z of each point are deliminated by '\t'.
    :return: numpy list of positions.
    """
    with open(f_path) as fp:
        content = fp.read().split('\n')

        # Read number of points from second
        count = len(content)-1

        # Read the points
        points = np.ndarray([count, 3], dtype=np.float32)
        for i in range(count):
            if content[i] == "":
                break
            temp = content[i].split('\t')
            points[i, 0] = float(temp[0])
            points[i, 1] = float(temp[1])
            points[i, 2] = float(temp[2])

        return points


def calc_warped_points(source_list_t, phi_t, dim, spacing, phi_spacing):
    """
    :param source_list_t: source image.
    :param phi_t: the inversed displacement. Domain in source coordinate.
    :param dim: voxel dimenstions.
    :param spacing: image spacing.
    :return: a N*3 tensor containg warped positions in the physical coordinate.
    """
    warped_list_t = F.grid_sample(phi_t, source_list_t, align_corners=True)

    warped_list_t = torch.flip(warped_list_t.permute(0, 2, 3, 4, 1), [4])[0, 0, 0]
    warped_list_t = torch.mul(torch.mul(warped_list_t, torch.from_numpy(dim-1.)), torch.from_numpy(phi_spacing))

    return warped_list_t

def eval_with_file(source_file, target_file, phi_file, dim, spacing, origin, phi_spacing, plot_result=False):
    """
    :param source_file: the path to the position of markers in source image.
    :param target_file: the path to the position of markers in target image.
    :param phi_file: the path to the displacement map (phi inverse). The basis is in source coordinate. 
    :param dim: voxel dimenstions.
    :param spacing: image spacing.
    :param plot_result: a bool value indicating whether to plot the result.
    """

    source_list = readPoint(source_file)
    target_list = readPoint(target_file)
    phi = np.expand_dims(np.load(phi_file), axis=0)
    # phi = np.expand_dims(np.moveaxis(sitk.GetArrayFromImage(sitk.ReadImage(phi_file)), -1, 0), axis=0)
    # phi = np.expand_dims(create_identity(dim), axis=0)

    res, res_seperate = eval_with_data(
        source_list, target_list, phi, dim, spacing, origin, phi_spacing, plot_result)
    return res, res_seperate

def eval_with_data(source_list, target_list, phi, dim, spacing, origin, phi_spacing, plot_result=False):
    """
    :param source_list: a numpy list of markers' position in source image.
    :param target_list: a numpy list of markers' position in target image.
    :param phi: displacement map in numpy format.
    :param dim: voxel dimenstions.
    :param spacing: image spacing.
    :param return: res, [dist_x, dist_y, dist_z] res is the distance between 
    the warped points and target points in MM. [dist_x, dist_y, dist_z] are 
    distances in MM along x,y,z axis perspectively.
    """
    origin_list = np.repeat([origin,], target_list.shape[0], axis=0)

    # Translate landmark from landmark coord to phi coordinate
    target_list_t = torch.from_numpy((target_list-1.)*spacing) - origin_list*phi_spacing
    source_list_t = torch.from_numpy((source_list-1.)*spacing) - origin_list*phi_spacing
    
    # The model accept SPR orientation of the 3D volume.
    # However, the landmarks are in SAR coordinates.
    # Need to change the coordinate system
    target_list_t[:,1] = (dim[1]-1)*phi_spacing[1] - target_list_t[:,1]
    source_list_t[:,1] = (dim[1]-1)*phi_spacing[1] - source_list_t[:,1]

    # Pay attention to the definition of align_corners in grid_sampling.
    # Translate landmarks to voxel index in image space [-1, 1]
    source_list_norm = source_list_t/phi_spacing/(dim-1.)*2.0-1.0
    source_list_norm = source_list_norm.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    phi_t = torch.from_numpy(phi).double()

    warped_list_t = calc_warped_points(source_list_norm, phi_t, dim, spacing, phi_spacing)
    # np.save( "./log/marker_warped.npy", warped_list_t.numpy())
    # warped_list_t = warped_list_t + torch.from_numpy((origin_list)*spacing)
    # np.save( "./log/marker_warped_target_coord.npy", warped_list_t.numpy())

    pdist = torch.nn.PairwiseDistance(p=2)
    dist = pdist(target_list_t, warped_list_t)
    idx = torch.argsort(dist).numpy()
    # np.save("./log/marker_most_inaccurate.npy", idx)
    dist_x = torch.mean(torch.abs(target_list_t[:,0] - warped_list_t[:,0])).item()
    dist_y = torch.mean(torch.abs(target_list_t[:,1] - warped_list_t[:,1])).item()
    dist_z = torch.mean(torch.abs(target_list_t[:,2] - warped_list_t[:,2])).item()
    res = torch.mean(dist).item()

    if plot_result:
        source_list_eucl = source_list*spacing - origin_list*phi_spacing
        fig, axes = plt.subplots(3,1)
        for i in range(3):
            axes[i].plot(target_list_t[:100,i].cpu().numpy(), "+", markersize=2, label="source")
            axes[i].plot(warped_list_t[:100,i].cpu().numpy(), '+', markersize=2, label="warped")
            axes[i].plot(source_list_eucl[:100,i], "+", markersize=2, label="target")
            axes[i].set_title("axis = %d"%i)
            
        plt.legend()
        # plt.show()
        # plt.savefig("../log/eval_dir_lab_reg.png", bbox_inches="tight", dpi=300)

    return res, [dist_x, dist_y, dist_z]

def compute_metrics(TRE):
    # compute GFR
    GFR = np.sum(TRE>10)/TRE.shape[0]
    Perc = np.percentile(TRE, (50, 75, 95))

    print(f"GFR:{GFR}")
    print(f"0.5, 0.75, 0.95 percentile:{Perc}")


def create_identity(shape):
    dim = len(shape)
    identity = np.ndarray([dim]+shape.tolist())
    if dim == 3:
        x = np.linspace(0, 1, shape[0])
        y = np.linspace(0, 1, shape[1])
        z = np.linspace(0, 1, shape[2])
        xv, yv, zv = np.meshgrid(x, y, z)
        
        identity[0,:,:,:] = yv
        identity[1,:,:,:] = xv
        identity[2,:,:,:] = zv

    return identity

def eval_copd_highres(dataset_path, exp_path):
    test_list = np.sort(np.load(dataset_path+"/test/data_id.npy"))
    landmark_folder = os.path.join(dataset_path, "landmarks")
    seg_folder = os.path.join(dataset_path, "preprocessed")
    disp_folder = os.path.join(exp_path, "tests")
    results = []
    TRE_list = []

    for case in test_list:
    # Load Params
        result = [case]
        copd_id = case.split('_')[0]
        phi_file = os.path.join(disp_folder, case+"_phi.npy")
        if not os.path.exists(phi_file):
            continue
        source_file = os.path.join(landmark_folder, f"{copd_id}_300_iBH_xyz_r1.txt")
        target_file = os.path.join(landmark_folder, f"{copd_id}_300_eBH_xyz_r1.txt")

        prop_file = dataset_path + "/preprocessed/"+case+"_prop.npy"
        if os.path.exists(prop_file):
            prop = np.load(prop_file, allow_pickle=True)
            origin = np.flip(prop.item().get('origin')).copy()
            phi_spacing = np.flip(prop.item().get('spacing')).copy()
        else:
            origin = np.array([0, 0, 0])
            phi_spacing = np.array([2.2, 2.2, 2.2])
        
        dim = np.array([160, 160, 160])
        spacing = COPD_spacing[copd_id]

        # Because we are reading phi instead of phi inverse. We switch target landmarks and source landmarks to 
        # keep the interface the same as the miccai versioni. Note, in miccai version, the input 
        # of the evaluation script is phi inverse.
        res, res_sep = eval_with_file(source_file=target_file, target_file=source_file,
                                      phi_file=phi_file, dim=dim, 
                                      spacing=spacing, origin=origin,
                                      phi_spacing=phi_spacing, plot_result=False)
        print("%s: TRE: %f, TRE(x,y,z): %f, %f, %f"%(case, res, res_sep[0], res_sep[1], res_sep[2]))
        TRE_list.append(res)
        result.append(res)
        result.append(res_sep[0])
        result.append(res_sep[1])
        result.append(res_sep[2])

        # Compute jacobian det
        # phi = np.expand_dims(np.load(phi_file), axis=0)
        # folding_mag, folding_count = compute_jacobi_map(
        #     phi, 
        #     1./(dim-1),
        #     crop_boundary=True,
        #     use_01=True
        # )

        source_seg = torch.from_numpy(np.flip(np.load(f"{seg_folder}/{copd_id}_source_seg.npy"), axis=(1)).copy()).unsqueeze(0).unsqueeze(0).float()
        target_seg = torch.from_numpy(np.flip(np.load(f"{seg_folder}/{copd_id}_target_seg.npy"), axis=(1)).copy()).unsqueeze(0).unsqueeze(0).float()
        
        phi = torch.from_numpy(np.load(phi_file)).float().unsqueeze(0)*2.-1.
        bilinear = Bilinear(zero_boundary=True, using_scale=False, mode="nearest")
        warped_seg = bilinear(source_seg, phi)
        metric = get_multi_metric(warped_seg, target_seg)
        result.append(metric['batch_avg_res']['dice'][0, 1])

        results.append(result)

    df = pd.DataFrame(data=results, columns=['id', 'dist', 'dist_x', 'dist_y', 'dist_z', 'dice'])
    df.to_csv(os.path.join(exp_path, 'evaluate_result.csv'))

    # Compute result
    compute_metrics(np.array(TRE_list))

    # print mean
    results_np = np.array([result[1] for result in results])
    print("The mean errors: {}".format(np.mean(results_np)))


def test_evaluation_script():
    print("------------Start Test-----------------")
    for i in range(1, 11):
        case_id = f"copd{i}"
        lung_reg_params = pars.ParameterDict()
        lung_reg_params.print_settings_off()
        lung_reg_params.load_JSON(f"../../Pre/lung_sdt_ct/settings/dirlab/lung_registration_setting_{case_id}.json")
        
        source_file = lung_reg_params["eval_marker_source_file"]
        target_file = lung_reg_params["eval_marker_target_file"]
        # source_file = "../../Pre/eval_data/" + lung_reg_params["eval_marker_source_file"]
        # target_file = "../../Pre/eval_data/" + lung_reg_params["eval_marker_target_file"]
        
        prop_file = f"../data/reg_lung_2d_3d_1000_dataset_4_proj_clean_bg/preprocessed/{case_id}_prop.npy"
        if os.path.exists(prop_file):
            prop = np.load(prop_file, allow_pickle=True)
            origin = np.flip(prop.item().get('origin')).copy()
            phi_spacing = np.flip(prop.item().get('spacing')).copy()
        else:
            origin = np.array([0, 0, 0])
            phi_spacing = np.array([2.2, 2.2, 2.2])
        
        dim = np.array([160, 160, 160])
        spacing = np.flipud(np.array(lung_reg_params["spacing"])).copy()

        # Create identity map
        phi = create_identity(dim)
        np.save("./temp/identity.npy", phi)
        phi_file = os.path.join("./temp/identity.npy")

        res, res_sep = eval_with_file(target_file, source_file, phi_file, dim, spacing, origin, phi_spacing, False)
        print("%s TRE: %f, TRE(x,y,z): %f, %f, %f"%(case_id, res, res_sep[0], res_sep[1], res_sep[2]))
    print("------------Finish Test-----------------")

if __name__ == "__main__":
    
    args = parser.parse_args()

    eval_copd_highres(args.data_path, '/'.join(args.setting.split('/')[:-1]))

