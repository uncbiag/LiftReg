import os
from datetime import datetime
from time import time

import numpy as np
import torch
from ..layers.losses import NCCLoss
from mermaid.utils import compute_warped_image_multiNC
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..utils.general import get_class, make_dir
from ..utils.metrics import get_multi_metric
from ..utils.net_utils import resume_train, save_model
from ..utils.utils import compute_jacobi_map, save_deformations, save_fig_3D
from ..utils.visualize_registration_results import show_current_images

from .NetworkBase import NetworkBase

from tqdm import tqdm

class RegistrationNet(NetworkBase):
    """
    the base class for image registration
    """
    def name(self):
        return 'RegistrationNetworkBase'

    def initialize(self, setting):
        """
        :param setting: ParameterDict, task settings
        :return: None
        """
        train_setting = setting['train']
        dataset_setting = setting['dataset']
        self.mode = train_setting[('mode', "train", '\'train\' or \'test\'')]

        # Init dataset and dataloader
        data_path = dataset_setting["data_path"]
        batch_size = train_setting["dataloader"]["batch_size"]
        shuffle = train_setting["dataloader"]["shuffle"]
        workers = train_setting["dataloader"]["workers"]
        
        dataset_class = get_class(dataset_setting["dataset_class"])
        if self.mode == "train":
            self.dataset = {'train': dataset_class(data_path, phase="train", 
                                                option=dataset_setting),
                            'val': dataset_class(data_path, phase='val',
                                                option=dataset_setting),
                            'debug': dataset_class(data_path, phase='debug',
                                                option=dataset_setting)}
            self.dataloaders = {'train': DataLoader(self.dataset["train"],
                                                batch_size=batch_size,
                                                shuffle=shuffle[0],
                                                num_workers=workers[0]),
                            'val': DataLoader(self.dataset["val"],
                                              batch_size=batch_size,
                                              shuffle=shuffle[1],
                                              num_workers=workers[1]),
                            'debug': DataLoader(self.dataset["debug"],
                                               batch_size=batch_size,
                                               shuffle=shuffle[2],
                                               num_workers=workers[2])}
        elif self.mode == "test":
            self.dataset = {'test': dataset_class(data_path, phase="test",
                                                  option=dataset_setting)}
            self.dataloaders = {"test": DataLoader(self.dataset["test"],
                                                   batch_size=batch_size,
                                                   shuffle=shuffle[3],
                                                   num_workers=workers[3])}

        # data param
        self.input_img_sz = setting['dataset'][('img_after_resize', None, "image size after resampling")]
     
        self.spacing =1. / (np.array(self.input_img_sz) - 1)

        # training param
        self.gpu_ids = train_setting[('gpu_ids', 0, 'the gpu id used for network methods')]
        exp_folder_path = train_setting["output_path"]
        self.check_point_path = os.path.join(exp_folder_path, "checkpoints")
        self.record_path = os.path.join(exp_folder_path, "records")
        self.log_path = os.path.join(exp_folder_path, "logs")
        self.test_path = os.path.join(exp_folder_path, "tests")
        
        self.save_model_frequency = train_setting[('save_model_frequency', 5, 'How often we need to save the model.')]
        
        self.epochs = train_setting[('epoch', 100, 'num of training epoch')]
        self.save_3d_img = train_setting[('save_3d_img', False, 'Should we save the 3D image')]
        self.val_frequency = train_setting[('val_frequency', 10, 'How many epoch per one validation')]
        self.save_fig = train_setting[('save_fig', True, 'Should we save figures')]
        self.save_fig_frequency = train_setting[('save_fig_frequency', 2, 'How many epoch to save figures')]
        self.debug_on = train_setting[("debug_on", False, 'Should we show debug result in training. If debug is on, it shares the same frequency as validation.')]
            
        # Init model.
        self.model = get_class(train_setting['model_class'])(self.input_img_sz, setting["train"]["model"])
        

        # Init loss.
        self.loss = get_class(train_setting['loss_class'])(setting["train"]["loss"])

        # Init optimizer and lr_scheduler.
        self.opt_setting = train_setting['optim']
        self.optimizer = None
        self.lr_scheduler = None
        self.exp_lr_scheduler = None
        self._init_optim(self.opt_setting)

        # Resume training if specified.
        if self.mode == 'train':
            self.continue_train = train_setting[('continue_train', False,
            "for network training method, continue training the model loaded from model_path")]
            continue_from = train_setting['continue_from']
            continue_train_lr = train_setting[('continue_train_lr', -1,
                                'Used when continue_train=True. The network \
                                will restore the lr from model_load_path if \
                                it is set to -1.')]
            if self.continue_train:
                self.start_epoch, self.global_step = resume_train(continue_from, self.model, self.optimizer, self.lr_scheduler)
                if continue_train_lr > 0:
                    self._update_learning_rate(continue_train_lr)
                    train_setting['optim']['lr'] = train_setting['optim']['lr'] if not self.continue_train else continue_train_lr
                    print("the learning rate has been changed into {} when resuming the training".format(continue_train_lr))
            else:
                self.start_epoch = 0
                self.global_step = {"train":0, "val":0, "debug":0, "test":0}
        elif self.mode == 'test':
            test_from = train_setting['test_from']
            self.start_epoch, self.global_step = resume_train(test_from, self.model, self.optimizer, self.lr_scheduler)

        # Init variables.
        self.iter_count = 0
        self.dim = len(self.input_img_sz)
        self.moving = None
        self.target = None
        self.output = None
        self.warped_label_map = None
        self.l_moving = None
        self.l_target = None
        self.cur_epoch = self.start_epoch
        self.use_01 = False
        """ the map is normalized to [-1,1] in registration net, todo normalized into [0,1], to be consisitent with mermaid """

        if self.gpu_ids != -1:
            self.model = self.model.cuda()

        if self.mode == "train":
            self.writer = SummaryWriter(self.log_path + "/" +datetime.now().strftime("%Y%m%d-%H%M%S"), flush_secs=30)

        # Should be removed
        self.fname_list = None

    def set_input(self, input):
        """
        :param data:
        :param is_train:
        :return:
        """
        _, self.fname_list = input
        prepared_input = {}
        self.moving = input[0]['source']
        self.target = input[0]['target']
        if 'source_label' in input[0]:
            self.l_moving = input[0]['source_label']
            self.l_target = input[0]['target_label']
        else:
            self.l_moving = None
            self.l_target = None

        if self.gpu_ids is not None and self.gpu_ids >= 0:
            for k,v in input[0].items():
                if isinstance(v, torch.Tensor) and len(v.shape) > 3:
                    prepared_input[k] = v.cuda()
                else:
                    prepared_input[k] = v
        else:
            for k,v in input[0].items():
                prepared_input[k] = v
            
        prepared_input['epoch'] = self.cur_epoch

        return prepared_input

    def do_some_clean(self):
        for k,v in self.output.items():
            del v
        self.output = None

    def _after_val(self, output):
        if self.l_moving is not None and self.l_target is not None:
            with torch.no_grad():
                self.warped_label_map = compute_warped_image_multiNC(self.l_moving.cuda(),
                                                                    output['phi'],
                                                                    self.spacing,
                                                                    spline_order=0,
                                                                    zero_boundary=True,
                                                                    use_01_input=self.use_01).cpu().numpy()
    
    def _compute_metrics(self, output):
        metrics_dic={}
        if self.l_target is not None:
            metrics_info = get_multi_metric(self.warped_label_map, self.l_target, verbose=False)

            # Since in this application, we only have label of 0 and 1
            # metrics_dic['score'] = np.mean(metrics_info['batch_avg_res']['dice'][0, 1:])
            for k,v in metrics_info['batch_avg_res'].items():
                metrics_dic[k] = v[0, 1]
        
        ncc = NCCLoss()
        with torch.no_grad():
            metrics_dic['score'] = 1. - ncc(output['warped'], output['target']).cpu().item()/output['warped'].shape[0]

        metrics_dic["folding_sum"], metrics_dic["folding_count"] = compute_jacobi_map(
            output["phi"].cpu().numpy(), 
            self.spacing,
            crop_boundary=True,
            use_01=self.use_01
        )
        return metrics_dic


    def _update_scheduler(self, epoch_val_loss):
        if self.lr_scheduler is not None and self.cur_epoch > 0:
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(epoch_val_loss)
            else:
                self.lr_scheduler.step()

    def _init_optim(self, setting, warmming_up=False):
        """
        set optimizers and scheduler

        :param setting: settings on optimizer
        :param network: model with learnable parameters
        :param warmming_up: if set as warmming up
        :return: optimizer, custom scheduler, plateau scheduler
        """
        optimize_name = setting['optim_type']
        lr = setting['lr']
        beta = setting['adam']['beta']
        lr_sched_setting = setting[('lr_scheduler', {},
                            "settings for learning scheduler")]
        self.lr_sched_type = lr_sched_setting['type']

        if optimize_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5, betas=beta)
            # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, eps=1e-5, betas=beta)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.optimizer.zero_grad()

        if self.lr_sched_type == 'custom':
            step_size = lr_sched_setting['custom'][('step_size', 50,
                            "update the learning rate every # epoch")]
            gamma = lr_sched_setting['custom'][('gamma', 0.5,
                            "the factor for updateing the learning rate")]
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                step_size=step_size, gamma=gamma)
        elif self.lr_sched_type == 'plateau':
            patience = lr_sched_setting['plateau']['patience']
            factor = lr_sched_setting['plateau']['factor']
            threshold = lr_sched_setting['plateau']['threshold']
            min_lr = lr_sched_setting['plateau']['min_lr']
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer,
                                                    mode='max',
                                                    patience=patience,
                                                    factor=factor,
                                                    verbose=True,
                                                    threshold=threshold,
                                                    min_lr=min_lr,
                                                    cooldown=lr_sched_setting['plateau']['cooldown'])

        if not warmming_up:
            print(" no warming up the learning rate is {}".format(lr))
        else:
            lr = setting['lr']/10
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr_scheduler.base_lrs = [lr]
            print(" warming up on the learning rate is {}".format(lr))

    def _update_learning_rate(self, new_lr=-1):
        """
        set new learning rate

        :param new_lr: new learning rate
        :return:
        """
        if new_lr < 0:
            lr = self.opt_setting['lr']
        else:
            lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr_scheduler.base_lrs=[lr]
        self.lr_scheduler.last_epoch = 1

    def _save_fig(self, output, phase):
        """
        save 2d center slice from x,y, z axis, for moving, target, warped, l_moving (optional), l_target(optional), (l_warped)

        :param phase: train|val|test|debug
        :return:
        """
        visual_param = {}
        visual_param['visualize'] = False
        visual_param['save_fig'] = True
        if phase == "test":
            visual_param['save_fig_path'] = self.test_path
            visual_param['save_fig_path_byname'] = os.path.join(self.test_path, 'byname')
            visual_param['save_fig_path_byiter'] = os.path.join(self.test_path, 'byiter')
        else:
            visual_param['save_fig_path'] = self.record_path
            visual_param['save_fig_path_byname'] = os.path.join(self.record_path, 'byname')
            visual_param['save_fig_path_byiter'] = os.path.join(self.record_path, 'byiter')
        visual_param['save_fig_num'] = 4
        visual_param['pair_name'] = self.fname_list
        visual_param['iter'] = phase + "_iter_{:0>6d}".format(self.cur_epoch)
        vizImage, vizTitle = self.model.get_disp()
        extraImage, extraName = self.model.get_extra_to_plot()
        show_label = True

        warped = output['warped']
        phi = output['phi']

        if show_label:
            show_current_images(self.iter_count, iS=self.moving, iT=self.target, iW=warped,
                                iSL=self.l_moving, iTL=self.l_target, iWL=self.warped_label_map,
                                vizImages=vizImage, vizName=vizTitle, phiWarped=phi,
                                visual_param=visual_param, extraImages=extraImage, extraName=extraName)
        else:
            show_current_images(self.iter_count, iS=self.moving, iT=self.target, iW=warped,
                                iSL=None, iTL=None, iWL=None,
                                vizImages=None, vizName=vizTitle, phiWarped=phi,
                                visual_param=visual_param, extraImages=None, extraName=extraName) 

        # If we have projection plot projection as well
        if 'warped_proj' in output:
            visual_param['pair_name'] = [f'{n}_proj' for n in self.fname_list]
            show_current_images(self.iter_count, 
                                iS=output['warped_proj'], 
                                iT=output['target_proj'], 
                                iW=output['warped_proj'],
                                visual_param=visual_param) 

    

    def _save_fig_3D(self, output, phase=None):
        warped = output["warped"]
        if 'target' in output:
            target = output['target']
        else:
            target = None
        if self.mode == "train":
            path = os.path.join(self.record_path, '3D')
            make_dir(path)
            save_fig_3D(warped[:,0],
                        path, 
                        self.fname_list, 
                        "_"+ phase + "_iter_" + str(self.cur_epoch) + "_warped", 
                        spacing=self.spacing, 
                        save_as_np=True)
            if target is not None:
                save_fig_3D(target[:,0],
                            path, 
                            self.fname_list, 
                            "_"+ phase + "_iter_" + str(self.cur_epoch) + "_target", 
                            spacing=self.spacing, 
                            save_as_np=True)

        else:
            path = os.path.join(self.test_path, '3D')
            make_dir(path)
            save_fig_3D(warped[:,0],
                        path, 
                        self.fname_list, 
                        "_"+ phase + "_warped", 
                        spacing=self.spacing, 
                        save_as_np=True)
            if target is not None:
                save_fig_3D(target[:,0],
                            path, 
                            self.fname_list, 
                            "_"+ phase + "_target", 
                            spacing=self.spacing, 
                            save_as_np=True)
        


    def step(self, input):
        self.optimizer.zero_grad()

        # Call model forward
        self.iter_count += 1
        if hasattr(self.model, 'set_cur_epoch'):
            self.model.set_cur_epoch(self.cur_epoch)
        output = self.model(input)
        
        output["epoch"] = self.cur_epoch
        losses = self.loss(output)
        
        losses["total_loss"].backward()
        self.optimizer.step()
        
        losses["total_loss"] = losses["total_loss"].detach().item()

        return losses
    
    def val_step(self, input, phase, save_fig=False, debug_on=False):
        with torch.no_grad():
            output = self.model(input)
            
            output["epoch"] = self.cur_epoch

            if phase == "val" and debug_on:
                losses = self.loss(output)
                losses["total_loss"] = losses["total_loss"].detach().item()
            else:
                losses = None

            for k,v in output.items():
                if isinstance(v, torch.Tensor):
                    output[k] = v.detach()
                else:
                    output[k] = v
        
        # Compute warped image for later usage (e.g., computing metrics, plot figures)
        self._after_val(output)

        # Compute metrics for validation set
        metrics_dict = self._compute_metrics(output)

        if save_fig:
            self._save_fig(output, phase)
            if self.save_3d_img:
                self._save_fig_3D(output, phase)
                if phase == "test":
                    save_deformations(output['phi'].cpu().numpy(), self.fname_list, self.test_path)

        return losses, metrics_dict
    
    def _train_model(self):
        since = time()
        best_score = -1
        best_metrics = {}
        best_epoch = -1
        val_score = 0

        for epoch in tqdm(range(self.start_epoch, self.epochs+1)):
            self.cur_epoch = epoch

            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)
 
            # Val
            if epoch % self.val_frequency == 0:
                self.set_val()
                running_val_loss = {}
                running_val_metrics = {}
                running_val_batch = 0

                if self.save_fig and epoch % self.save_fig_frequency == 0:
                    save_fig = True
                else:
                    save_fig = False
                
                for data in self.dataloaders['val']:
                    batch_size = len(data[1])
                    losses, metrics_dict = self.val_step(self.set_input(data), 'val', save_fig, debug_on=self.debug_on)

                    running_val_batch += 1
                    # Post metrics
                    for k, v in metrics_dict.items():
                        if k in running_val_metrics.keys():
                            running_val_metrics[k] += v
                        else:
                            running_val_metrics[k] = v
                        
                    # Poset loss
                    if self.debug_on:
                        for k,v in losses.items():
                            if k in running_val_loss.keys():
                                running_val_loss[k] += v
                            else:
                                running_val_loss[k] = v
                        
                    self.global_step['val'] += 1
                
                for k, v in running_val_metrics.items():
                    self.writer.add_scalar(f"Val_metrics/{k}", v/running_val_batch, epoch)
                if self.debug_on:
                    for k, v in running_val_loss.items():
                        self.writer.add_scalar(f"Val_loss/{k}", v/running_val_batch, epoch)

                # Save best model
                val_score = running_val_metrics["score"]/running_val_batch
                if val_score > best_score:
                    best_score = val_score
                    best_epoch = epoch

                    save_model(epoch, self, self.global_step,
                            self.check_point_path, '',
                            True)

                    ite_num = len(self.dataloaders['val'])
                    for k,v in running_val_metrics.items():
                        running_val_metrics[k] = v/ite_num
                    best_metrics = running_val_metrics

            # Train
            self.set_train()
            for data in self.dataloaders['train']:
                self.global_step['train'] += 1
                losses = self.step(self.set_input(data))

                for k,v in losses.items():
                    self.writer.add_scalar(f"Train/{k}", v, self.global_step['train'])            

            # Debug
            if self.debug_on and self.save_fig and epoch % self.save_fig_frequency==0:
                self.set_val()
                running_debug_metrics = {}
                running_debug_batch = 0
                save_fig = True

                for data in self.dataloaders['debug']:
                    batch_size = len(data[1])
                    _, metrics_dict = self.val_step(self.set_input(data), 'debug', save_fig, debug_on=self.debug_on)
                    save_fig = False # Only save fig for the first iteration

                    running_debug_batch += 1
                    # Cumulate metrics
                    for k, v in metrics_dict.items():
                        if k in running_debug_metrics.keys():
                            running_debug_metrics[k] += v
                        else:
                            running_debug_metrics[k] = v
                        
                    self.global_step['debug'] += 1
                
                for k, v in running_debug_metrics.items():
                    self.writer.add_scalar(f"Debug/{k}", v/running_debug_batch, epoch)
            
            # Update scheduler
            self._update_scheduler(val_score)

            # After finishing one epoch, check whether we should save the model.
            if epoch % self.save_model_frequency == 0:
                if self.debug_on:
                    save_model(epoch, self, self.global_step,
                                self.check_point_path, f'epoch_{epoch}',
                                False)
                else:
                    save_model(epoch, self, self.global_step,
                                self.check_point_path, 'latest',
                                False)

        time_elapsed = time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val score : {:4f} is at epoch {}'.format(best_score, best_epoch))
        print(f'Best val metrics:{best_metrics}')
        
        for k,v in best_metrics.items():
            self.writer.add_scalar(f"Best_metrics/{k}", v, 0)
        
        self.writer.close()

    def _test_model(self):
        since = time()

        # Test
        self.set_val()
        running_debug_metrics = {}
        for data in self.dataloaders['test']:
            with torch.no_grad():
                _, metrics_dict = self.val_step(self.set_input(data), 'test', True)

                for k, v in metrics_dict.items():
                    if k in running_debug_metrics.keys():
                        running_debug_metrics[k] += v
                    else:
                        running_debug_metrics[k] = v

        time_elapsed = time() - since
        for k, v in running_debug_metrics.items():
            print(f"{k}: {v/len(self.dataloaders['test'])}")
        print('Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
    def eval(self):
        since = time()

        # Test
        self.set_val()
        for data in self.dataloaders['test']:
            with torch.no_grad():
                _, metrics_dict = self.val_step(self.set_input(data), 'test', True)

        time_elapsed = time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))



