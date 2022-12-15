import argparse
import enum
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from liftreg.utils.medical_image_utils import load_IMG, resample, seg_bg_mask, seg_lung_mask

parser = argparse.ArgumentParser(description="Prepare data for training")
parser.add_argument('-o','--output_path', required=True, type=str,
                    default=None, help='the path to the root of dataset folders')
parser.add_argument('-d','--dataset_name', required=True, type=str,
                    default='',help='the name of the dataset')
parser.add_argument('--data_type', required=False, type=str,
                    default="all" ,help='value can be train/val/all')


class FILE_TYPE(enum.Enum):
    nii = 1
    copd = 2
    dct = 3
    copd_highres = 4


COPD_spacing = {"copd1":[0.625, 0.625, 2.5],
                "copd2":[0.645, 0.645, 2.5],
                "copd3":[0.652, 0.652, 2.5],
                "copd4":[0.590, 0.590, 2.5],
                "copd5":[0.647, 0.647, 2.5],
                "copd6":[0.633, 0.633, 2.5],
                "copd7":[0.625, 0.625, 2.5],
                "copd8":[0.586, 0.586, 2.5],
                "copd9":[0.664, 0.664, 2.5],
                "copd10":[0.742, 0.742, 2.5]}
COPD_shape = {"copd1":[121, 512, 512],
                "copd2":[102, 512, 512],
                "copd3":[126, 512, 512],
                "copd4":[126, 512, 512],
                "copd5":[131, 512, 512],
                "copd6":[119, 512, 512],
                "copd7":[112, 512, 512],
                "copd8":[115, 512, 512],
                "copd9":[116, 512, 512],
                "copd10":[135, 512, 512]}
FDCT_spacing = {"dct1":[0.97, 0.97, 2.5],
                   "dct2":[1.16, 1.16, 2.5],
                   "dct3":[1.15, 1.15, 2.5],
                   "dct4":[1.13, 1.13, 2.5],
                   "dct5":[1.10, 1.10, 2.5],
                   "dct6":[0.97, 0.97, 2.5],
                   "dct7":[0.97, 0.97, 2.5],
                   "dct8":[0.97, 0.97, 2.5],
                   "dct9":[0.97, 0.97, 2.5],
                   "dct10":[0.97, 0.97, 2.5]}
FDCT_shape = {"dct1":[94, 256, 256],
                   "dct2":[112, 256, 256],
                   "dct3":[104, 256, 256],
                   "dct4":[99, 256, 256],
                   "dct5":[106, 256, 256],
                   "dct6":[128, 512, 512],
                   "dct7":[136, 512, 512],
                   "dct8":[128, 512, 512],
                   "dct9":[128, 512, 512],
                   "dct10":[120, 512, 512]
}   

Data_Info = {
    "val":[
        {
            "data_list_path":"/playpen-raid1/lin.tian/data/raw/copd",
            "file_type": FILE_TYPE.copd,
            "idx_file_name": "data_id"
        },
        #### We were using high resolution of the DIRLAB COPDGene images from COPDGene Study for evaluation in the paper.
        #### If you do not have the access to the high resolution version, you can use the low
        #### resolution version obtained from DIRLAB.
        #### If low resolution is used, comment out the following part.
        # {
        #     "data_list_path": "/playpen-raid1/lin.tian/data/raw/DIRLABCasesHighRes",
        #     "file_type": FILE_TYPE.copd_highres,
        #     "idx_file_name": "data_id"
        # }
    ]
    }

def resize_img(source, target, source_seg, target_seg, ori_spacing, new_spacing):
    scale_factor = np.array(new_spacing)/np.array(ori_spacing)

    affine = sitk.AffineTransform(3)
    affine.Scale(scale_factor)
    reference_target = target
    reference_source = source
    new_target = sitk.Resample(target, reference_target, affine, sitk.sitkLinear)
    new_source = sitk.Resample(source, reference_source, affine, sitk.sitkLinear)
    new_target_seg = sitk.Resample(target_seg, target_seg, affine, sitk.sitkLinear, 0)
    new_source_seg = sitk.Resample(source_seg, source_seg, affine, sitk.sitkLinear, 0)
    
    return new_source, new_target, new_source_seg, new_target_seg, affine.GetMatrix()


def normalize_intensity(img, linear_clip=False):
    # TODO: Lin-this line is for CT. Modify it to make this method more general.
    img[img<-1024] = -1024
    return img
    # if linear_clip:
    #     img = img - img.min()
    #     normalized_img =img / np.percentile(img, 95) * 0.95
    # else:
    #     min_intensity = img.min()
    #     max_intensity = img.max()
    #     normalized_img = (img-img.min())/(max_intensity - min_intensity)
    # normalized_img = normalized_img*2 - 1
    # return normalized_img


def calc_relative_atten_coef(img):
    new_img = img.astype(np.float32).copy()
    new_img[new_img<-1024] = -1024
    return (new_img+1024.)/1024.


def process_single_file(path_pair, sz, spacing, seg_bg=False, type=FILE_TYPE.nii):
    if type == FILE_TYPE.copd:
        p = path_pair[0]
        case_id = path_pair[4] # p[p.rfind("/")+1:p.rfind("_")]
        ori_spacing = np.flipud(COPD_spacing[case_id])
        ori_sz = COPD_shape[case_id]

        ori_source = load_IMG(path_pair[0], ori_sz, ori_spacing, ori_spacing)-1024
        source, _, _ = resample(ori_source, ori_spacing, spacing)
        source[source<-1024] = -1024
        ori_target = load_IMG(path_pair[1], ori_sz, ori_spacing, ori_spacing)-1024
        target, new_spacing, _ = resample(ori_target, ori_spacing, spacing)
        target[target<-1024] = -1024

        if seg_bg:
            bg_hu = np.min(source)
            source_bg_seg, source_bbox = seg_bg_mask(source)
            source[source_bg_seg==0] = bg_hu

            bg_hu = np.min(target)
            target_bg_seg, source_bbox = seg_bg_mask(target)
            target[target_bg_seg==0] = bg_hu
            total_voxel = np.prod(target.shape)
            print("##########Area percentage of ROI:{:.2f}, {:.2f}".format(float(np.sum(source_bg_seg))/total_voxel,
               float(np.sum(target_bg_seg))/total_voxel))

        source_seg, _ = seg_lung_mask(source)
        target_seg, _ = seg_lung_mask(target)

        # Pad 0 if shape is smaller than desired size.
        new_origin = np.array((0, 0, 0))
        sz_diff = sz - source.shape
        sz_diff[sz_diff < 0] = 0
        pad_width = [[int(sz_diff[0]/2), sz_diff[0]-int(sz_diff[0]/2)], 
                    [int(sz_diff[1]/2), sz_diff[1]-int(sz_diff[1]/2)],
                    [int(sz_diff[2]/2), sz_diff[2]-int(sz_diff[2]/2)]]
        source = np.pad(source, pad_width, constant_values=-1024)
        target = np.pad(target, pad_width, constant_values=-1024)
        source_seg = np.pad(source_seg, pad_width, constant_values=0)
        target_seg = np.pad(target_seg, pad_width, constant_values=0)
        new_origin[sz_diff>0] = -np.array(pad_width)[sz_diff>0,0]

        # Crop if shape is greater than desired size.
        sz_diff = source.shape - sz
        bbox = [[int(sz_diff[0]/2), int(sz_diff[0]/2) + sz[0]],
                [int(sz_diff[1]/2), int(sz_diff[1]/2) + sz[1]],
                [int(sz_diff[2]/2), int(sz_diff[2]/2) + sz[2]]]
        source = source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        target = target[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        source_seg = source_seg[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        target_seg = target_seg[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        new_origin[sz_diff>0] = np.array(bbox)[sz_diff>0,0]

        source = normalize_intensity(source)
        target = normalize_intensity(target)
        return source, target, source_seg, target_seg, new_origin, new_spacing
    elif type == FILE_TYPE.dct:
        p = path_pair[0]
        case_id = path_pair[4] #p[p.rfind("/")+1:p.rfind("_")]
        ori_spacing = np.flipud(FDCT_spacing[case_id])
        ori_sz = FDCT_shape[case_id]
        
        source = load_IMG(path_pair[0], ori_sz, ori_spacing, ori_spacing)-1024
        source, _, _ = resample(source, ori_spacing, spacing)
        source[source<-1024] = -1024

        target = load_IMG(path_pair[1], ori_sz, ori_spacing, ori_spacing)-1024
        target, new_spacing, _ = resample(target, ori_spacing, spacing)
        target[target<-1024] = -1024
        
        bbox = [0, 0, 0] + list(ori_sz)
        if seg_bg:
            (D, W, H) = ori_sz
            bg_hu = np.min(source)
            source_bg_seg, source_bbox = seg_bg_mask(source)
            source[source_bg_seg==0] = bg_hu

            bg_hu = np.min(target)
            target_bg_seg, target_bbox = seg_bg_mask(target)
            target[target_bg_seg==0] = bg_hu

            bbox = [min(s,t) for s,t in zip(source_bbox[:3], target_bbox[:3])] +\
                [max(s,t) for s,t in zip(source_bbox[3:], target_bbox[3:])]
            total_voxel = np.prod(target.shape)
            print("##########Area percentage of ROI:{:.2f}, {:.2f}".format(float(np.sum(source_bg_seg))/total_voxel,
               float(np.sum(target_bg_seg))/total_voxel))
        
        source_seg,_ = seg_lung_mask(source)
        target_seg,_ = seg_lung_mask(target)

        # Pad 0 if shape is smaller than desired size.
        new_origin = np.array((0, 0, 0))
        sz_diff = sz - source.shape
        sz_diff[sz_diff < 0] = 0
        pad_width = [[int(sz_diff[0]/2), sz_diff[0]-int(sz_diff[0]/2)], 
                    [int(sz_diff[1]/2), sz_diff[1]-int(sz_diff[1]/2)],
                    [int(sz_diff[2]/2), sz_diff[2]-int(sz_diff[2]/2)]]
        source = np.pad(source, pad_width, constant_values=-1024)
        target = np.pad(target, pad_width, constant_values=-1024)
        source_seg = np.pad(source_seg, pad_width, constant_values=0)
        target_seg = np.pad(target_seg, pad_width, constant_values=0)
        new_origin[sz_diff>0] = -np.array(pad_width)[sz_diff>0,0]

        # Crop if shape is greater than desired size.
        sz_diff = source.shape - sz
        bbox = [[int(sz_diff[0]/2), int(sz_diff[0]/2) + sz[0]],
                [int(sz_diff[1]/2), int(sz_diff[1]/2) + sz[1]],
                [int(sz_diff[2]/2), int(sz_diff[2]/2) + sz[2]]]
        source = source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        target = target[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        source_seg = source_seg[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        target_seg = target_seg[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        new_origin[sz_diff>0] = np.array(bbox)[sz_diff>0,0]

        source = normalize_intensity(source)
        target = normalize_intensity(target)

        return source, target, source_seg, target_seg, new_origin, new_spacing

def read_txt_into_list(file_path):
    """
    read the list from the file, each elem in a line compose a list, each line compose to a list,
    the elem "None" would be filtered and not considered
    :param file_path: the file path to read
    :return: list of list
    """
    import re
    lists= []
    with open(file_path,'r') as f:
        content = f.read().splitlines()
        if len(content)>0:
            lists= [[x if x!='None'else None for x in re.compile('\s*[,|\s+]\s*').split(line)] for line in content]
            lists = [list(filter(lambda x: x is not None, items)) for items in lists]
        lists = [item[0] if len(item) == 1 else item for item in lists]
    return lists

def read_copd_data_list(data_folder_path):
    case_list = os.listdir(data_folder_path)
    return_list = []
    for case in case_list:
        case_dir = os.path.join(data_folder_path, case+'/'+case)
        case_data = ["","","","",case]
        case_data[0] = os.path.join(case_dir, case+'_iBHCT.img')
        case_data[1] = os.path.join(case_dir, case+'_eBHCT.img')
        return_list.append(case_data)
    return return_list

def read_dct_data_list(data_folder_path):
    case_list = os.listdir(data_folder_path)
    return_list = []
    for case in case_list:
        case_id = case.lower()[0:case.find('Pack')]
        case_dir = os.path.join(data_folder_path, case+'/Images')

        # rename the file
        # for f in os.listdir(case_dir):
        #     series_id = f[f.find('_T')+1: f.find('_T')+4]
        #     os.rename(os.path.join(case_dir, f), os.path.join(case_dir, case_id+'_'+series_id+'.img'))  

        case_data = ["","","","", "dct"+case_id[4:]]
        case_data[0] = os.path.join(case_dir, case_id + '_T00.img')
        case_data[1] = os.path.join(case_dir, case_id + '_T50.img')
        return_list.append(case_data)
    return return_list

def plot_preprocessed(source, target, save_path, source_seg=None, target_seg=None):
    if source_seg is not None and target_seg is not None:
        fig, axes = plt.subplots(4, 4)
    else:
        fig, axes = plt.subplots(2, 4)
    for i in range(4):
        axes[0, i].imshow(source[:, 30*i, :])
        axes[1, i].imshow(target[:, 30*i, :])
        if source_seg is not None and target_seg is not None:
            axes[2, i].imshow(source_seg[:, 30*i, :])
            axes[3, i].imshow(target_seg[:, 30*i, :])
    axes[0, 0].set_ylabel("Source")
    axes[1, 0].set_ylabel("Target")
    if source_seg is not None and target_seg is not None:
        axes[2, 0].set_ylabel("Source Seg")
        axes[3, 0].set_ylabel("Target Seg")
    plt.savefig(save_path)
    plt.clf()
    plt.close()

def preprocess(data_folder_path, preprocessed_path, log_path, file_type=FILE_TYPE.nii, case_num=200):
    if not os.path.exists(data_folder_path):
        print("Did not find data list file at %s"%data_folder_path)
        return
    
    # file_list = read_txt_into_list(os.path.join(data_folder_path,
    #                                             'pair_path_list.txt'))
    if file_type == FILE_TYPE.copd:
        file_list = read_copd_data_list(data_folder_path)
    elif file_type == FILE_TYPE.dct:
        file_list = read_dct_data_list(data_folder_path)

    if len(file_list) > case_num:
        file_list = file_list[:case_num]
    
    case_id_list = []
    data_count = len(file_list)
    for i in range(data_count):
        case_id = file_list[i][4]
        case_id_list.append(case_id)
        print("Preprocessing %i/%i %s"%(i, data_count, case_id))
        
        source, target, source_seg, target_seg, new_origin, new_spacing = process_single_file(file_list[i],
                                                                np.array((160, 160, 160)),
                                                                np.array((2.2, 2.2, 2.2)),
                                                                seg_bg=True,
                                                                type=file_type)
        np.save(os.path.join(preprocessed_path, "%s_target.npy"%case_id), target)
        np.save(os.path.join(preprocessed_path, "%s_source.npy"%case_id), source)
        np.save(os.path.join(preprocessed_path, "%s_source_seg.npy"%case_id), source_seg)
        np.save(os.path.join(preprocessed_path, "%s_target_seg.npy"%case_id), target_seg)
        # Plot image
        plot_preprocessed(source, target,
                        os.path.join(log_path, "%s_preprocessed.png"%case_id),
                        source_seg, target_seg)
        if new_origin is not None:
            prop = {}
            prop["origin"] = new_origin
            prop["spacing"] = new_spacing
            np.save(os.path.join(preprocessed_path, "%s_prop.npy"%case_id), prop)

    return case_id_list


def save_id_list(task_root, file_name, case_id_list, mode="train"):
    if mode == "train":
        train_data_id_path = task_root + "/train"
        if not os.path.exists(train_data_id_path):
            os.makedirs(train_data_id_path)
        
        val_data_id_path = task_root + "/val"
        if not os.path.exists(val_data_id_path):
            os.makedirs(val_data_id_path)

        debug_data_id_path = task_root + "/debug"
        if not os.path.exists(debug_data_id_path):
            os.makedirs(debug_data_id_path)
        case_count = len(case_id_list)
        np.random.shuffle(case_id_list)
        train_list = case_id_list[:int(case_count*4/5)]
        val_list = case_id_list[int(case_count*4/5):case_count]

        np.save(os.path.join(train_data_id_path, file_name), train_list)
        np.save(os.path.join(debug_data_id_path, file_name), train_list)
        np.save(os.path.join(val_data_id_path, file_name), val_list)
    elif mode == "test":
        test_data_id_path = task_root + "/test"
        if not os.path.exists(test_data_id_path):
            os.makedirs(test_data_id_path)
        np.save(os.path.join(test_data_id_path, file_name), case_id_list) 

if __name__ == "__main__":
    # After processing, the ct image should in SAR orientation.
    args = parser.parse_args()

    task_root = os.path.join(os.path.abspath(args.output_path), args.dataset_name)
    preprocessed_path = task_root + "/preprocessed/"
    if not os.path.exists(preprocessed_path):
        os.makedirs(preprocessed_path)

    log_path = preprocessed_path + "/preview"
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    data_type = args.data_type
    if data_type == "train" or data_type == "all":
        data_list_path = Data_Info["train"]["data_list_path"]
        file_type = Data_Info["train"]["file_type"]
        id_file_name = Data_Info["train"]["idx_file_name"]
        case_id_list = preprocess(data_list_path, preprocessed_path, log_path, file_type, case_num=1000)

        save_id_list(task_root, id_file_name, case_id_list, mode="train")
    
    if data_type == "val" or data_type == "all":
        for d in Data_Info["val"]:
            data_list_path = d["data_list_path"]
            file_type = d["file_type"]
            id_file_name = d["idx_file_name"]
            case_id_list = preprocess(data_list_path, preprocessed_path, log_path, file_type, case_num=1000)

            save_id_list(task_root, id_file_name, case_id_list, mode="test")
