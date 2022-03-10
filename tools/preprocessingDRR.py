import argparse
import enum
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.general import make_dir
from utils.sdct_projection_utils import (
    calculate_projection_wraper, calculate_projection_wraper_with_geo_csv_file, calc_relative_atten_coef)
from tools.extract_ROI_from_syn_projection import extract_ROI_from_image_in_batch

parser = argparse.ArgumentParser(description="Generating DRR for dataset")
parser.add_argument('--data_path', required=True, type=str,
                    default=None, help='the path to the root of reference dataset folders')
parser.add_argument('--drr_folder_name', required=True, type=str,
                    default='',help='name of the projection folder')
parser.add_argument('--scan_range', required=False, type=float,
                    default=30 ,help='the scane range of the synthesized projections')
parser.add_argument('--scan_num', required=False, type=int,
                    default=4 ,help='the number of projections generated')
parser.add_argument('--geo_path', required=False, type=str,
                    default="" ,help='the path to the csv file storing geometry infomation')
parser.add_argument('--receptor_h', required=False, type=int,
                    default="0" ,help='the height of the receptor field')
parser.add_argument('--receptor_w', required=False, type=int,
                    default="0" ,help='the width of the receptor field')
parser.add_argument('--phase', required=False, type=str,
                    default="all" ,help='specify which subset of the given dataset should we run. \
                    Value sould be in ["train", "val", "debug","test", "all"].')
parser.add_argument('--preview', required=False, default=False, action='store_true', 
                    help='specify whether to plot the generated drr. The preview folder is under drr direction.')
parser.add_argument('-g',"--gpu_id",required=False,type=int,default=0,help='gpu_id to use')

class GEO_TYPE(enum.Enum):
    SCAN_RANGE = 1
    GEO_CSV = 2



def plot_drr(source, target, source_proj, target_proj, source_roi, target_roi, save_path):
    num_per_slice_plot = int(source.shape[1]/4)
    num_per_drr_plot = int(source_proj.shape[0]/4)
    if source_roi is not None:
        fig, axes = plt.subplots(6, 4)
    else:
        fig, axes = plt.subplots(4, 4)
    for i in range(4):
        axes[0, i].imshow(source[:, i*num_per_slice_plot, :])
        axes[1, i].imshow(target[:, i*num_per_slice_plot, :])
        axes[2, i].imshow(source_proj[i*num_per_drr_plot, :, :])
        axes[3, i].imshow(target_proj[i*num_per_drr_plot, :, :])
        if source_roi is not None:
            axes[4, i].imshow(source_roi[i*num_per_drr_plot, :, :])
            axes[5, i].imshow(target_roi[i*num_per_drr_plot, :, :])
    axes[0, 0].set_ylabel("Source")
    axes[1, 0].set_ylabel("Target")
    axes[2, 0].set_ylabel("Source proj")
    axes[3, 0].set_ylabel("Target proj")
    axes[4, 0].set_ylabel("Source roi")
    axes[5, 0].set_ylabel("Target roi")
    plt.savefig(save_path)
    plt.clf()
    plt.close()

if __name__ == "__main__":
    ##################################
    # Generate DRR for given dataset.
    # Three ways of providing geo info:
    # 1. Provide scan_range and proj_num
    # 2. Provide one single csv file storing geometry infomation
    # 3. (Lin TODO) Provide geometry files for each scan 
    ##################################
    args = parser.parse_args()

    # Set gpu
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
        torch.backends.cudnn.benchmark = True
    print(f"The current device: {torch.cuda.current_device()}")

    # What format does the geo info in?
    if args.geo_path is not "":
        geo_type = GEO_TYPE.GEO_CSV
    else:
        geo_type = GEO_TYPE.SCAN_RANGE
    
    # Setup the size of the receptor
    if not args.receptor_h == 0 and not args.receptor_w == 0:
        receptor_size = [args.receptor_h, args.receptor_w]
    else:
        receptor_size = None

    task_root = os.path.abspath(args.data_path)
    preprocessed_path = task_root + "/preprocessed/"
    assert os.path.exists(preprocessed_path), "No preprocessed folder found."

    # Creat folder structure
    drr_root_folder = os.path.join(task_root, f"drr/{args.drr_folder_name}")
    make_dir(drr_root_folder)
    
    drr_folder = os.path.join(drr_root_folder, "drr")
    make_dir(drr_folder)

    roi_folder = os.path.join(drr_root_folder, "roi")
    make_dir(roi_folder)

    # Do we need to plot the drr?
    if args.preview:
        log_path = drr_root_folder + "/preview"
        make_dir(log_path)

    # Load geo infomation
    scan_range = args.scan_range
    scan_num = args.scan_num
    geo_path = args.geo_path # Make sure the position is in SAL order.

    # Specify which subset of data should we process.
    phases = ["train","debug","val", "test"]
    if args.phase in phases:
        phases = [args.phase]
    else:
        assert args.phase=="all", "Wrong phase value."

    # Start to generate DRR
    for p in phases:
        print(f"Processing data in {p} ...")

        # Load data id list
        data_ids = np.load(os.path.join(task_root, f"{p}/data_id.npy"))

        for d in data_ids:
            # Load target and source
            target = np.load(os.path.join(preprocessed_path, f"{d}_target.npy"))
            source = np.load(os.path.join(preprocessed_path, f"{d}_source.npy"))

            # !!! temp code. change orientation from SAR to SPR
            target = np.flip(target, axis=1)
            source = np.flip(source, axis=1)

            # Generate DRR
            if geo_type == GEO_TYPE.SCAN_RANGE:
                source_proj, poses = calculate_projection_wraper(calc_relative_atten_coef(source), scan_range, scan_num, (2.2, 2.2, 2.2), receptor_size=receptor_size)
                target_proj, _ = calculate_projection_wraper(calc_relative_atten_coef(target), scan_range, scan_num, (2.2, 2.2, 2.2), receptor_size=receptor_size)
            else:
                source_proj, poses = calculate_projection_wraper_with_geo_csv_file(calc_relative_atten_coef(source), (2.2, 2.2, 2.2), geo_path, receptor_size=receptor_size)
                target_proj, _ = calculate_projection_wraper_with_geo_csv_file(calc_relative_atten_coef(target), (2.2, 2.2, 2.2), geo_path, receptor_size=receptor_size)

            # Extract roi
            source_roi, source_bbox = extract_ROI_from_image_in_batch(source_proj)
            target_roi, target_bbox = extract_ROI_from_image_in_batch(target_proj)

            # Save files
            np.save(os.path.join(drr_folder, f"{d}_target_proj.npy"), target_proj)
            np.save(os.path.join(drr_folder, f"{d}_source_proj.npy"), source_proj)
            np.save(os.path.join(roi_folder, f"{d}_target_proj_roi.npy"), target_roi)
            np.save(os.path.join(roi_folder, f"{d}_source_proj_roi.npy"), source_roi)
            np.save(os.path.join(roi_folder, f"{d}_target_proj_roi_bbox.npy"), target_bbox)
            np.save(os.path.join(roi_folder, f"{d}_source_proj_roi_bbox.npy"), source_bbox)

            # Plot drr
            if args.preview:
                plot_drr(source, target, source_proj, target_proj, source_proj*source_roi, target_proj*target_roi, os.path.join(log_path, f"{d}_preview.png"))

    np.save(os.path.join(drr_folder, "poses.npy"), poses)
