from __future__ import division, print_function

import os
from multiprocessing import *
from pathlib import Path

import blosc
import numpy as np
import progressbar as pb
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

blosc.set_nthreads(1)

class Registration2D3DDataset(Dataset):
    """registration dataset."""

    def __init__(self, data_path, phase=None, transform=None, option=None):
        """
        the dataloader for registration task, to avoid frequent disk communication, all pairs are compressed into memory
        :param data_path:  string, path to the data
            the data should be preprocessed and saved into txt
        :param phase:  string, 'train'/'val'/ 'test'/ 'debug' ,    debug here means a subset of train data, to check if model is overfitting
        :param transform: function,  apply transform on data
        : seg_option: pars,  settings for segmentation task,  None for segmentation task
        : reg_option:  pars, settings for registration task, None for registration task

        """
        self.data_id_path = data_path + "/" + phase + "/data_id.npy"
        self.data_path = data_path + "/preprocessed"
        self.drr_path = data_path + f"/drr/{option['drr_folder_name']}/drr"
        self.roi_path = data_path + f"/drr/{option['drr_folder_name']}/roi"
        self.roi_path = self.roi_path if os.path.exists(self.roi_path) else ""
        self.phase = phase
        self.transform = transform
        ind = ['train', 'val', 'test', 'debug'].index(phase)
        max_num_for_loading=option['max_num_for_loading',(-1,-1,-1,-1),"the max number of pairs to be loaded, set -1 if there is no constraint,[max_train, max_val, max_test, max_debug]"]
        self.max_num_for_loading = max_num_for_loading[ind]
        """ the max number of pairs to be loaded into the memory,[max_train, max_val, max_test, max_debug]"""
        self.has_label = option['use_segmentation_map', False, 'indicates whether to load segmentation map from dataset.']
        self.spacing = option['spacing_to_refer', (1,1,1)]
        self.load_projection_interval = option['load_projection_interval', 2]

        self.get_file_list()
        self.reg_option = option
        load_training_data_into_memory = option[('load_training_data_into_memory',False,"when train network, load all training sample into memory can relieve disk burden")]
        self.load_into_memory = load_training_data_into_memory if phase == 'train' else False
        self.pair_list = []
        self.original_source_list = []
        self.proj_list = []
        self.target_proj_roi = []
        self.spacing_list = []
        self.target_poses_list = []

        self.init_img_pool()

    def get_file_list(self):
        """
        get the all files belonging to data_type from the data_path,
        :return: full file path list, file name list
        """
        if not os.path.exists(self.data_id_path):
            self.name_list = []
            return
        self.name_list = np.load(self.data_id_path)

        if self.max_num_for_loading > 0:
            read_num = min(self.max_num_for_loading, len(self.name_list))
            self.name_list = self.name_list[:read_num]

    def _read_case(self, case_id_list, img_label_dic):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(case_id_list)).start()
        count = 0
        for case_id in case_id_list:
            img_label_np = {}

            # Load source image
            # Pay attention here. Flip is used to transform SAR orientation to SPR orientation.
            # This is only applied to synthetic dataset because the real data has already been transformed in preprocessing script.
            # TODO: Should make such change in preprocess script.
            source_img = np.flip(np.load(os.path.join(self.data_path, case_id+"_source.npy")).astype(np.float32), axis=(1))
            img_label_np['original_source'] = blosc.pack_array(self.calc_relative_atten_coef(source_img))
            source_img = self._normalize_intensity(source_img, linear_clip=False)
            if self.has_label:
                source_seg = np.flip(np.load(os.path.join(self.data_path, case_id+"_source_seg.npy")).astype(np.float32), axis=(1))
                img_label_np['source_seg'] = blosc.pack_array(source_seg)
            else:
                img_label_np['source_seg'] = None
            img_label_np['source'] = blosc.pack_array(source_img)

            # Load source projections
            source_proj = np.load(os.path.join(self.drr_path, case_id+"_source_proj.npy")).astype(np.float32)
            source_proj = self._normalize_intensity(source_proj, linear_clip=True, clip_range=(0,6))[::self.load_projection_interval]
            img_label_np['source_proj'] = blosc.pack_array(source_proj.astype(np.float32))

            # Load target image
            target_img = np.flip(np.load(os.path.join(self.data_path, case_id+"_target.npy")).astype(np.float32), axis=(1))
            target_img = self._normalize_intensity(target_img, linear_clip=False)#, clip_range=[-1024, -600])
            if self.has_label:
                target_seg = np.flip(np.load(os.path.join(self.data_path, case_id+"_target_seg.npy")).astype(np.float32), axis=(1))
                img_label_np['target_seg'] = blosc.pack_array(target_seg)
            else:
                img_label_np['target_seg'] = None
            img_label_np['target'] = blosc.pack_array(target_img)

            # Load target projections
            target_proj = np.load(os.path.join(self.drr_path, case_id+"_target_proj.npy")).astype(np.float32)
            target_proj = self._normalize_intensity(target_proj, linear_clip=True, clip_range=(0,6))[::self.load_projection_interval]
            img_label_np['target_proj'] = blosc.pack_array(target_proj.astype(np.float32))

            # Load roi
            if self.roi_path is not "":
                roi = np.load(os.path.join(self.roi_path, f"{case_id}_target_proj_roi.npy"))[::self.load_projection_interval]
                img_label_np["target_proj_roi"] = blosc.pack_array(roi.astype(np.int))

            # Load geo info
            img_label_np['target_poses'] = np.load(os.path.join(self.drr_path, "poses.npy")).astype(np.float32)[::self.load_projection_interval]
            
            img_label_np["spacing"] = np.array(self.spacing)
            
            img_label_dic[case_id] = img_label_np
            count += 1
            pbar.update(count)
        pbar.finish()

    def init_img_pool(self):
        """img pool shoudl include following thing:
        img_label_path_dic:{img_name:{'img':img_fp,'label':label_fp,...}
        img_label_dic: {img_name:{'img':img_np,'label':label_np},......}
        pair_name_list:[[pair1_s,pair1_t],[pair2_s,pair2_t],....]
        pair_list [[s_np,t_np,sl_np,tl_np],....]
        only the pair_list need to be used by get_item method
        """
        manager = Manager()
        img_label_dic = manager.dict()
        num_of_workers = 12
        num_of_workers = num_of_workers if len(self.name_list)>num_of_workers else len(self.name_list)
        split_dict = self.__split_dict(self.name_list, num_of_workers)
        procs = []
        for i in range(num_of_workers):
            p = Process(target=self._read_case, args=(split_dict[i], img_label_dic,))
            p.start()
            print("pid:{} start:".format(p.pid))

            procs.append(p)

        for p in procs:
            p.join()
        
        for case_name in self.name_list:
            case = img_label_dic[case_name]
            if self.has_label:
                self.pair_list.append([case['source'], case['target'], case['source_seg'], case['target_seg']])
            else:
                self.pair_list.append([case['source'], case['target']])
            self.proj_list.append([case['source_proj'], case['target_proj']])
            self.original_source_list.append(case['original_source'])
            if "target_proj_roi" in case:
                self.target_proj_roi.append(case["target_proj_roi"])
            
            self.target_poses_list.append(case["target_poses"])
            self.spacing_list.append(case['spacing'])
        
        print("the loading phase {} finished, total {} img and labels have been loaded".format(self.phase, len(img_label_dic)))

    def _normalize_intensity(self, img, linear_clip=False, clip_range=None):
        """
        a numpy image, normalize into intensity [-1,1]
        (img-img.min())/(img.max() - img.min())
        :param img: image
        :param linear_clip:  Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
        :return:
        """

        if linear_clip:
            if clip_range is not None:
                img[img<clip_range[0]] = clip_range[0]
                img[img>clip_range[1]] = clip_range[1]
                normalized_img = (img-clip_range[0]) / (clip_range[1] - clip_range[0]) 
            else:
                img = img - img.min()
                normalized_img =img / np.percentile(img, 95) * 0.95
        else:
            # If we normalize in HU range of softtissue
            min_intensity = img.min()
            max_intensity = img.max()
            normalized_img = (img-img.min())/(max_intensity - min_intensity)
        normalized_img = normalized_img*2 - 1
        return normalized_img

    def __split_dict(self, dict_to_split, split_num):
        index_list = list(range(len(dict_to_split)))
        index_split = np.array_split(np.array(index_list), split_num)
        split_dict = []
        for i in range(split_num):
            dj = dict_to_split[index_split[i][0]:index_split[i][-1]+1]
            split_dict.append(dj)
        return split_dict

    def __len__(self):
        return len(self.name_list)
        # return len(self.name_list)*500 if len(self.name_list)<200 and self.phase=='train' else len(self.name_list)  #############################3

    def __getitem__(self, idx):
        """
        # todo  update the load data part to mermaid fileio
        :param idx: id of the items
        :return: the processed data, return as type of dic

        """
        # print(idx)
        idx = idx % len(self.name_list)

        filename = self.name_list[idx]
        pair_list = [blosc.unpack_array(item) for item in self.pair_list[idx]]
        target_proj = blosc.unpack_array(self.proj_list[idx][1])
        source_proj = blosc.unpack_array(self.proj_list[idx][0])
        original_source = blosc.unpack_array(self.original_source_list[idx])

        sample = {'source': np.expand_dims(pair_list[0], axis=0),
                  'target': np.expand_dims(pair_list[1], axis=0),
                  'target_proj': np.asarray(target_proj).astype(np.float32),
                  'source_proj': np.asarray(source_proj).astype(np.float32),
                  'original_source': np.expand_dims(original_source, axis=0)}

        if self.has_label:
            sample["source_label"] = np.expand_dims(pair_list[2], axis=0)
            sample["target_label"] = np.expand_dims(pair_list[3], axis=0)

        if len(self.target_proj_roi) > 0:
            sample["target_proj_roi"] = np.asarray(blosc.unpack_array(self.target_proj_roi[idx])).astype(np.int)

        if self.transform:
            sample['source'] = self.transform(sample['source'])
            sample['target'] = self.transform(sample['target'])
            if self.has_label:
                sample['source_label'] = self.transform(sample['source_label'])
                sample['target_label'] = self.transform(sample['target_label'])
            sample['target_proj'] = self.transform(sample['target_proj'])
            sample['source_proj'] = self.transform(sample['source_proj'])
            sample['original_source'] = self.transform(sample['original_source'])
            if "target_proj_roi" in sample:
                sample['target_proj_roi'] = self.transform(sample['target_proj_roi'])
        
        sample['target_poses'] = self.target_poses_list[idx]
        sample['spacing'] = self.spacing_list[idx].copy()
        return sample, filename
    
    def calc_relative_atten_coef(self, img):
        new_img = img.astype(np.float32).copy()
        new_img[new_img<-1000] = -1000
        return (new_img+1000.)/1000.*0.2


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        n_tensor = torch.from_numpy(sample)
        return n_tensor
