
import os

import torch

import utils.module_parameters as pars
from tools.evaluate_dir_lab import eval_copd_highres
from utils.general import get_class
from utils.utils import set_seed_for_demo

if __name__ == '__main__':
    """
        An evaluation interface for learning methods.
        The method support list :  mermaid-related methods
        Assume there is three level folder, output_root_path/ data_task_folder/ task_folder 
        Arguments: 
            --setting_folder_path/ -ts: path of the folder where settings are saved,should include cur_task_setting.json, mermaid_affine_settings.json(optional) and mermaid_nonp_settings(optional)
            --gpu_id/ -g: gpu_id to use
    """
    import argparse

    parser = argparse.ArgumentParser(description="An easy interface for training registration models")
    parser.add_argument('-s','--setting_path', required=True, type=str,
                        default=None,help='path of the folder where settings are saved,should include cur_task_setting.json')         
    parser.add_argument('-g',"--gpu_id",required=False,type=int,default=0,help='gpu_id to use')
    
    args = parser.parse_args()
    print(args)
    set_seed_for_demo()
    # Set gpu
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
        torch.backends.cudnn.benchmark = True

    setting = pars.ParameterDict()
    setting.load_JSON(args.setting_path)

    # Run network in test mode
    setting["train"]["mode"] = "test"
    setting["train"]["test_from"] = os.path.join(setting["train"]["output_path"], "checkpoints", "model_best.pth.tar")
    setting["train"]["save_fig"] = True
    setting["train"]["save_3d_img"] = True
    setting["train"]["gpu_ids"] = args.gpu_id
    test_network = get_class(setting["train"]["network_class"])()
    test_network.initialize(setting)
    test_network.run()

    # Eval with landmarks
    eval_copd_highres(setting["dataset"]["data_path"], setting["train"]["output_path"])
    