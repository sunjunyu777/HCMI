from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model import STDnet
from dataset.dataset import Crowd
from glob import glob
import cv2
import random
from tqdm import tqdm


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'venice' in args.data_dir or 'dronecrowd' in args.data_dir:
            self.datasets = {x: [Crowd(args.data_dir+'/'+x+'/'+file, args.is_gray, x, args.frame_number,
                                       args.crop_height, args.crop_width, args.roi_path)
                                 for file in sorted(os.listdir(os.path.join(args.data_dir, x)), key=int)]
                             for x in ['train', 'val']}
            self.dataloaders = {x: [DataLoader(self.datasets[x][file],
                                               batch_size=(args.batch_size
                                               if x == 'train' else 1),
                                               shuffle=(True if x == 'train' else False),
                                               num_workers=args.num_workers * self.device_count,
                                               pin_memory=(True if x == 'train' else False))
                                    for file in range(len(os.listdir(os.path.join(args.data_dir, x))))]
                                for x in ['train', 'val']}
        else:
            self.datasets = {x: Crowd(os.path.join(args.data_dir, x), args.is_gray, x, args.frame_number, args.crop_height,
                                      args.crop_width, args.roi_path) for x in ['train', 'val']}
            self.dataloaders = {x: DataLoader(self.datasets[x],
                                              batch_size=(args.batch_size
                                              if x == 'train' else 1),
                                              shuffle=(True if x == 'train' else False),
                                              num_workers=args.num_workers*self.device_count,
                                              pin_memory=(True if x == 'train' else False))
                                for x in ['train', 'val']}
        self.model = STDnet(
            in_chans=3,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            load_pretrained=True
        )
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0
        self.criterion = torch.nn.MSELoss(reduction='sum').to(self.device)
        self.save_all = args.save_all
        self.num = -1

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % 50 == 0:
                self.num += 1
                self.best_mae = np.inf
                self.best_mse = np.inf
            elif epoch % args.val_epoch == 0:
                self.val_epoch()

    def train_eopch(self):
        args = self.args
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  

        if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'venice' in args.data_dir or 'dronecrowd' in args.data_dir:
            file_list = list(range(len(os.listdir(os.path.join(args.data_dir, 'train')))))
            random.shuffle(file_list)
            total_steps = sum([len(self.dataloaders['train'][file]) for file in file_list])
            pbar = tqdm(total=total_steps, desc=f'Epoch {self.epoch} Train')
            
            for file in file_list:
                for step, (imgs, targets, keypoints, mask) in enumerate(self.dataloaders['train'][file]):
                    b0, f0, c0, h0, w0 = imgs.shape
                    assert b0 == 1
                    imgs = imgs.to(self.device)
                    center_target = targets.to(self.device)[:, 1:2]  # [B, 1, 1, H/8, W/8]
                    
                    if isinstance(mask, torch.Tensor) and mask.dim() > 2:
                        center_mask = mask[:, 1].to(self.device)  # [B, 1, H/8, W/8]
                        if center_mask.dim() > 3:
                            center_mask = center_mask.squeeze(1)  # [B, H/8, W/8]
                    else:
                        center_mask = mask.to(self.device)  # [B, H/8, W/8]

                    with torch.set_grad_enabled(True):
                        output = self.model(imgs)
                        output = output * center_mask

                        if center_target.dim() == 5:  
                            center_target = center_target.squeeze(2)  
                        loss = self.criterion(output, center_target)
                        
                        pred_count = torch.sum(output)
                        if keypoints.dim() > 1 and keypoints.shape[1] > 1:
                            center_keypoint = keypoints[0, 1].clone().detach().float().to(self.device)
                        else:
                            center_keypoint = keypoints[0].clone().detach().float().to(self.device)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        pre_count = pred_count.detach().cpu().numpy()
                        center_keypoint_np = center_keypoint.cpu().numpy()
                        res = pre_count - center_keypoint_np
                        epoch_loss.update(loss.item(), 1)
                        epoch_mse.update(res * res, 1)
                        epoch_mae.update(abs(res), 1)
                        
                        pbar.update(1)
                        pbar.set_posSTDix({'Loss': f'{loss.item():.4f}', 'MAE': f'{abs(res):.2f}'})
            
            pbar.close()

            logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                                 time.time() - epoch_start))
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)  # control the number of saved models
        else:
            dataloader = self.dataloaders['train']
            pbar = tqdm(dataloader, desc=f'Epoch {self.epoch} Train')
            for step, (imgs, targets, keypoints, mask) in enumerate(pbar):
                b0, f0, c0, h0, w0 = imgs.shape
                assert b0 == 1
                imgs = imgs.to(self.device)
                center_target = targets.to(self.device)[:, 1:2]  # [B, 1, 1, H/8, W/8]
                center_mask = mask.to(self.device)  # [B, H/8, W/8]

                with torch.set_grad_enabled(True):
                    output = self.model(imgs)
                    output = output * center_mask

                    if center_target.dim() == 5:  
                        center_target = center_target.squeeze(2)  
                    loss = self.criterion(output, center_target)
                    
                    pred_count = torch.sum(output)
                    if keypoints.dim() > 1 and keypoints.shape[1] > 1:
                        center_keypoint = keypoints[0, 1].clone().detach().float().to(self.device)
                    else:
                        center_keypoint = keypoints[0].clone().detach().float().to(self.device)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pre_count = pred_count.detach().cpu().numpy()
                    center_keypoint_np = center_keypoint.cpu().numpy()
                    res = pre_count - center_keypoint_np
                    epoch_loss.update(loss.item(), 1)
                    epoch_mse.update(res * res, 1)
                    epoch_mae.update(abs(res), 1)
                    
                    pbar.set_posSTDix({'Loss': f'{loss.item():.4f}', 'MAE': f'{abs(res):.2f}'})
            
            pbar.close()

            logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                                 time.time()-epoch_start))
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  
        if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'dronecrowd' in args.data_dir:
            file_list = sorted(os.listdir(os.path.join(args.data_dir, 'val')), key=int)
            total_steps = sum([len(self.dataloaders['val'][file]) for file in range(len(file_list))])
            pbar = tqdm(total=total_steps, desc=f'Epoch {self.epoch} Val')
            
            for file in range(len(file_list)):
                epoch_res = []
                
                if 'ucsd' in args.data_dir or 'fdst' in args.data_dir:
                    for imgs, keypoints, masks in self.dataloaders['val'][file]:
                        b, f, c, h, w = imgs.shape
                        assert b == 1, 'the batch size should equal to 1 in validation mode'
                        imgs = imgs.to(self.device)
                        
                        if masks.dim() > 3:  
                            center_mask = masks[:, 1].to(self.device)  # [B, 1, H, W]
                            if center_mask.dim() > 3:
                                center_mask = center_mask.squeeze(1)  # [B, H, W]
                        else:
                            center_mask = masks.to(self.device)  # [B, H, W]
                        
                        if keypoints.dim() > 1 and keypoints.shape[1] > 1:
                            center_keypoint = keypoints[0, 1].clone().detach().float().to(self.device)
                        else:
                            center_keypoint = keypoints[0].clone().detach().float().to(self.device)

                        with torch.set_grad_enabled(False):
                            output = self.model(imgs)
                            output = output * center_mask
                            
                            pred_count = torch.sum(output).detach().cpu().numpy()
                            res = center_keypoint.cpu().numpy() - pred_count
                            epoch_res.append(res)
                            
                            pbar.update(1)
                            pbar.set_posSTDix({'MAE': f'{abs(res):.2f}'})
                elif 'dronecrowd' in args.data_dir:
                    for imgs, keypoints, masks in self.dataloaders['val'][file]:
                        b, f, c, h, w = imgs.shape
                        assert b == 1, 'the batch size should equal to 1 in validation mode'
                        imgs = imgs.to(self.device)
                        
                        if masks.dim() > 3:  
                            center_mask = masks[:, 1].to(self.device)  # [B, 1, H, W]
                            if center_mask.dim() > 3:
                                center_mask = center_mask.squeeze(1)  # [B, H, W]
                        else:
                            center_mask = masks.to(self.device)  # [B, H, W]
                        
                        if keypoints.dim() > 1 and keypoints.shape[1] > 1:
                            center_keypoint = keypoints[0, 1].clone().detach().float().to(self.device)
                        else:
                            center_keypoint = keypoints[0].clone().detach().float().to(self.device)

                        with torch.set_grad_enabled(False):
                            output = self.model(imgs)
                            output = output * center_mask
                                
                            pred_count = torch.sum(output).detach().cpu().numpy()
                            res = center_keypoint.cpu().numpy() - pred_count
                            epoch_res.append(res)
                            
                            pbar.update(1)
                            pbar.set_posSTDix({'MAE': f'{abs(res):.2f}'})
                else:
                    for imgs, keypoints, masks in self.dataloaders['val'][file]:
                        b, f, c, h, w = imgs.shape
                        assert b == 1, 'the batch size should equal to 1 in validation mode'
                        imgs = imgs.to(self.device)
                        
                        if masks.dim() > 3:  
                            center_mask = masks[:, 1].to(self.device)  # [B, 1, H, W]
                            if center_mask.dim() > 3:
                                center_mask = center_mask.squeeze(1)  # [B, H, W]
                        else:
                            center_mask = masks.to(self.device)  # [B, H, W]
                        
                        if keypoints.dim() > 1 and keypoints.shape[1] > 1:
                            center_keypoint = keypoints[0, 1].clone().detach().float().to(self.device)
                        else:
                            center_keypoint = keypoints[0].clone().detach().float().to(self.device)

                        with torch.set_grad_enabled(False):
                            output = self.model(imgs)
                            output = output * center_mask
                                
                            pred_count = torch.sum(output).detach().cpu().numpy()
                            res = center_keypoint.cpu().numpy() - pred_count
                            epoch_res.append(res)
                            
                            pbar.update(1)
                            pbar.set_posSTDix({'MAE': f'{abs(res):.2f}'})
                
                epoch_res = np.array(epoch_res)
                if 'fdst' in args.data_dir or 'ucsd' in args.data_dir:
                    val_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'val'+'/'+file_list[file], '*.jpg')),
                                          key=lambda x: int(x.split('/')[-1].split('.')[0]))
                else:
                    val_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'val'+'/'+file_list[file], '*.jpg')),
                                          key=lambda x: int(x.split('_')[-1].split('.')[0]))
                
                if len(epoch_res) > len(val_img_list) - args.frame_number + 1:
                    epoch_res = epoch_res[:len(val_img_list) - args.frame_number + 1]
                
                for e in epoch_res:
                    sum_res.append(e)
            sum_res = np.array(sum_res)
            mse = np.sqrt(np.mean(np.square(sum_res)))
            mae = np.mean(np.abs(sum_res))
            logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

            model_state_dic = self.model.state_dict()
            if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
                self.best_mse = mse
                self.best_mae = mae
                logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
                if self.save_all:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                    self.best_count += 1
                else:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.num)))

        elif 'venice' in args.data_dir:
            sum_res = []
            file_list = sorted(os.listdir(os.path.join(args.data_dir, 'val')), key=int)
            total_steps = sum([len(self.dataloaders['val'][file]) for file in range(len(file_list))])
            pbar = tqdm(total=total_steps, desc=f'Epoch {self.epoch} Val')
            
            for file in range(len(file_list)):
                epoch_res = []
                for imgs, keypoints, masks in self.dataloaders['val'][file]:
                    b, f, c, h, w = imgs.shape
                    assert b == 1, 'the batch size should equal to 1 in validation mode'
                    imgs = imgs.to(self.device)
                    
                    if masks.dim() > 3:  
                        center_mask = masks[:, 1].to(self.device)  # [B, 1, H, W]
                        if center_mask.dim() > 3:
                            center_mask = center_mask.squeeze(1)  # [B, H, W]
                    else:
                        center_mask = masks.to(self.device)  # [B, H, W]
                    
                    if keypoints.dim() > 1 and keypoints.shape[1] > 1:
                        center_keypoint = keypoints[0, 1].clone().detach().float().to(self.device)
                    else:
                        center_keypoint = keypoints[0].clone().detach().float().to(self.device)

                    with torch.set_grad_enabled(False):
                        output = self.model(imgs)
                        output = output * center_mask
                        
                        pred_count = torch.sum(output).detach().cpu().numpy()
                        res = center_keypoint.cpu().numpy() - pred_count
                        epoch_res.append(res)
                        
                        pbar.update(1)
                        pbar.set_posSTDix({'MAE': f'{abs(res):.2f}'})
                
                epoch_res = np.array(epoch_res)
                val_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'val'+'/'+file_list[file], '*.jpg')),
                                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
                
                if len(epoch_res) > len(val_img_list) - args.frame_number + 1:
                    epoch_res = epoch_res[:len(val_img_list) - args.frame_number + 1]
                
                for e in epoch_res:
                    sum_res.append(e)
            sum_res = np.array(sum_res)
            mse = np.sqrt(np.mean(np.square(sum_res)))
            mae = np.mean(np.abs(sum_res))
            logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

            model_state_dic = self.model.state_dict()
            if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
                self.best_mse = mse
                self.best_mae = mae
                logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
                if self.save_all:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                    self.best_count += 1
                else:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.num)))

        else:
            epoch_res = []
            pbar = tqdm(self.dataloaders['val'], desc=f'Epoch {self.epoch} Val')
            for imgs, keypoints, masks in pbar:
                b, f, c, h, w = imgs.shape
                assert b == 1, 'the batch size should equal to 1 in validation mode'
                imgs = imgs.to(self.device)
                
                if masks.dim() > 3:  
                    center_mask = masks[:, 1].to(self.device)  # [B, 1, H, W]
                    if center_mask.dim() > 3:
                        center_mask = center_mask.squeeze(1)  # [B, H, W]
                else:
                    center_mask = masks.to(self.device)  # [B, H, W]
                
                if keypoints.dim() > 1 and keypoints.shape[1] > 1:
                    center_keypoint = keypoints[0, 1].clone().detach().float().to(self.device)
                else:
                    center_keypoint = keypoints[0].clone().detach().float().to(self.device)

                with torch.set_grad_enabled(False):
                    output = self.model(imgs)
                    output = output * center_mask
                    
                    pred_count = torch.sum(output).detach().cpu().numpy()
                    res = center_keypoint.cpu().numpy() - pred_count
                    epoch_res.append(res)
                    
            
                    pbar.set_posSTDix({'MAE': f'{abs(res):.2f}'})
            
            pbar.close()

            epoch_res = np.array(epoch_res)
            val_img_list = sorted(glob(os.path.join(os.path.join(args.data_dir, 'val'), '*.jpg')),
                                  key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            if len(epoch_res) > len(val_img_list) - args.frame_number + 1:
                epoch_res = epoch_res[:len(val_img_list) - args.frame_number + 1]
            
            for e in epoch_res:
                sum_res.append(e)
            mse = np.sqrt(np.mean(np.square(sum_res)))
            mae = np.mean(np.abs(sum_res))
            logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time()-epoch_start))

            model_state_dic = self.model.state_dict()
            if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
                self.best_mse = mse
                self.best_mae = mae
                logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
                if self.save_all:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                    self.best_count += 1
                else:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.num)))
