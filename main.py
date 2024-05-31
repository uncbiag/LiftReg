
import os
import sys
from datetime import datetime
from stat import S_IREAD

import torch

import liftreg.utils.module_parameters as pars
from liftreg.utils.general import get_class, get_git_revisions_hash, make_dir
from liftreg.utils.utils import set_seed_for_demo


def prepare(args):
    output_path = args.output_path
    exp_name = args.exp_name
    data_path = args.data_path
    setting_path = args.setting_path
    continue_from = args.continue_from
    is_continue = True if continue_from is not None else False
    dataset_name = data_path.split('/')[-1]

    # Create experiment folder
    timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    exp_folder_path = os.path.join(output_path, dataset_name, exp_name, timestamp)
    make_dir(exp_folder_path)

    # Create checkpoint path, record path and log path
    checkpoint_path = os.path.join(exp_folder_path, "checkpoints")
    make_dir(checkpoint_path)
    record_path = os.path.join(exp_folder_path, "records")
    make_dir(record_path)
    log_path = os.path.join(exp_folder_path, "logs")
    make_dir(log_path)
    test_path = os.path.join(exp_folder_path, "tests")
    make_dir(test_path)


    setting_folder_path = args.setting_path
    setting_path = os.path.join(setting_folder_path, 'cur_task_setting.json')
    assert os.path.isfile(setting_path), "Setting file is not found."
    setting = pars.ParameterDict()
    setting.load_JSON(setting_path)

    # Update setting file with command input
    setting["dataset"]["data_path"] = data_path
    setting["train"]["output_path"] = exp_folder_path
    if is_continue:
        setting["train"]["continue_train"] = True
        setting["train"]["continue_from"] = continue_from
    setting["train"]["gpu_ids"] = args.gpu_id

    # Write the commit hash for current codebase
    label = get_git_revisions_hash()
    setting["exp"]["git_commit"] = label

    # Write the command argument list to the setting file
    setting["exp"]["command_line"] = ' '.join(sys.argv)

    task_output_path = os.path.join(exp_folder_path, 'cur_task_setting.json')
    setting.write_ext_JSON(task_output_path)

    # Make the setting file read-only
    os.chmod(task_output_path, S_IREAD)

    if "mermaid" in setting['train']['model_class']:
        mermaid_backup_json_path = os.path.join(exp_folder_path, 'mermaid_nonp_settings.json')
        mermaid_setting_json = setting['train']['model']['mermaid_net_json_pth']
        if len(mermaid_setting_json) == 0:
            mermaid_setting_json = os.path.join(setting_folder_path, 'mermaid_nonp_settings.json')
        mermaid_setting = pars.ParameterDict()
        mermaid_setting.load_JSON(mermaid_setting_json)
        mermaid_setting.write_ext_JSON(mermaid_backup_json_path)
    return setting

if __name__ == '__main__':
    """
        A training interface for learning methods.
        The method support list :  mermaid-related methods
        Assume there is three level folder, output_root_path/ data_task_folder/ task_folder 
        Arguments: 
            --output_path/ -o: the path of output folder
            --data_path/ -d: the path to the dataset folder
            --task_name / -tn: task name i.e. run_training_vsvf_task, run_training_rdmm_task
            --setting_folder_path/ -ts: path of the folder where settings are saved,should include cur_task_setting.json, mermaid_affine_settings.json(optional) and mermaid_nonp_settings(optional)
            --train_affine_first: train affine network first, then train non-parametric network
            --gpu_id/ -g: gpu_id to use
    """
    import argparse

    parser = argparse.ArgumentParser(description="An easy interface for training registration models")
    parser.add_argument('-o','--output_path', required=True, type=str,
                        default=None,help='the path of output folder')
    parser.add_argument('-d','--data_path', required=True, type=str,
                        default='',help='the path to the data folder')
    parser.add_argument('-e','--exp_name', required=True, type=str,
                        default=None,help='the name of the experiment')
    parser.add_argument('-s','--setting_path', required=True, type=str,
                        default=None,help='path of the folder where settings are saved,should include cur_task_setting.json')
    parser.add_argument('--continue_from',required=False, type=str,
                        help='Which checkpoint we should continue train from')             
    parser.add_argument('-g',"--gpu_id",required=False,type=int,default=0,help='gpu_id to use')
    
    args = parser.parse_args()
    print(args)

    # Set gpu
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
        torch.backends.cudnn.benchmark = True
    print(f"The current device: {torch.cuda.current_device()}")

    set_seed_for_demo()
    setting = prepare(args)

    network = get_class(setting["train"]["network_class"])()
    network.initialize(setting)
    network.run()
    