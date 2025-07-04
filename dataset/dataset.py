from PIL import Image
import torch.utils.data as data
import os
from glob import glob
from torchvision import transforms
import numpy as np
import h5py
import torch
from scipy.io import loadmat
import cv2
import random



class Crowd(data.Dataset):
    def __init__(self, root_path, is_gray=False, method='train', frame_number=3,
                 crop_height=512, crop_width=512, roi_path=None):
        self.root_path = root_path
        self.frame_number = frame_number
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.roi_path = roi_path
        if 'fdst' in self.root_path or 'ucsd' in self.root_path:
            self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')),
                                  key=lambda x: int(x.split('/')[-1].split('.')[0]))
        else:
            self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')),
                                  key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        if self.method == 'train':
            return len(self.im_list) - self.frame_number + 1
        elif self.method == 'val':
            return len(self.im_list) - self.frame_number + 1

    def __getitem__(self, item):
        img_list = []
        target_list = []
        keypoint_list = []
        mask_list = []
        if self.method == 'train' and 'venice' not in self.root_path:
            width, height = Image.open(self.im_list[0]).convert('RGB').size
            new_width = self.crop_width
            new_height = self.crop_height
            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)
            right = left + new_width
            bottom = top + new_height
            rate = random.random()
            
            for q in range(item, item+self.frame_number):
                img_path = self.im_list[q]
                img = Image.open(img_path).convert('RGB')
                img = img.crop((left, top, right, bottom))
                if rate > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img = self.trans(img)
                img_list.append(img)

                target_path = img_path.replace('jpg', 'h5')
                target_file = h5py.File(target_path, mode='r')
                target_ori = np.asarray(target_file['density'])
                target_ori = target_ori[top:bottom, left:right]
                if rate > 0.5:
                    target_ori = np.fliplr(target_ori)
                target = cv2.resize(target_ori, (target_ori.shape[1] // 32 * 4, target_ori.shape[0] // 32 * 4),
                                            interpolation=cv2.INTER_CUBIC) * (
                                         (target_ori.shape[0] * target_ori.shape[1]) / (
                                         (target_ori.shape[1] // 32 * 4) * (target_ori.shape[0] // 32 * 4)))
                
                if self.roi_path:
                    mask = np.load(self.roi_path)
                    mask = mask[top:bottom, left:right]  
                    if rate > 0.5:
                        mask = np.fliplr(mask)
                    mask = cv2.resize(mask, (target.shape[1], target.shape[0]))
                    target = target * mask
                
                keypoint = np.sum(target)
                keypoint_list.append(keypoint)
                target_list.append(torch.from_numpy(target.copy()).float().unsqueeze(0))
            
            if self.roi_path:
                mask_final = np.load(self.roi_path)
                mask_final = mask_final[top:bottom, left:right]
                if rate > 0.5:
                    mask_final = np.fliplr(mask_final)
                mask_final = cv2.resize(mask_final, (target.shape[1], target.shape[0]))
            else:
                mask_final = np.ones((target.shape[0], target.shape[1]), dtype=int)
            
            return torch.stack(img_list, dim=0), torch.stack(target_list, dim=0), torch.tensor(keypoint_list), torch.tensor(mask_final)

        elif self.method == 'train' and 'venice' in self.root_path:
            width, height = Image.open(self.im_list[0]).convert('RGB').size
            new_width = self.crop_width
            new_height = self.crop_height
            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)
            right = left + new_width
            bottom = top + new_height
            rate = random.random()
            for q in range(item, item+self.frame_number):
                img_path = self.im_list[q]
                img = Image.open(img_path).convert('RGB')
                img = img.crop((left, top, right, bottom))
                if rate > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img = self.trans(img)
                img_list.append(img)

                roi_path = img_path.replace('jpg', 'mat')
                mask = loadmat(roi_path)['roi']
                mask = mask[top:bottom, left:right]
                if rate > 0.5:
                    mask = np.fliplr(mask)

                target_path = img_path.replace('jpg', 'h5')
                target_file = h5py.File(target_path, mode='r')
                target_ori = np.asarray(target_file['density'])
                target_ori = target_ori[top:bottom, left:right]
                if rate > 0.5:
                    target_ori = np.fliplr(target_ori)
                target = cv2.resize(target_ori, (target_ori.shape[1] // 32 * 4, target_ori.shape[0] // 32 * 4),
                                            interpolation=cv2.INTER_CUBIC) * (
                                         (target_ori.shape[0] * target_ori.shape[1]) / (
                                         (target_ori.shape[1] // 32 * 4) * (target_ori.shape[0] // 32 * 4)))
                
                mask = cv2.resize(mask, (target.shape[1], target.shape[0]))
                mask_list.append(torch.from_numpy(mask.copy()).float().unsqueeze(0))
                
                target = target * mask
                keypoint = np.sum(target)
                keypoint_list.append(keypoint)
                target_list.append(torch.from_numpy(target.copy()).float().unsqueeze(0))
            return torch.stack(img_list, dim=0), torch.stack(target_list, dim=0), torch.tensor(keypoint_list), torch.stack(mask_list, dim=0)

        elif self.method == 'val' and 'venice' not in self.root_path:
            for q in range(item, item+self.frame_number):
                img_path = self.im_list[q]
                img = Image.open(img_path).convert('RGB')
                img = self.trans(img)
                img_list.append(img)

                h5_path = img_path.replace('jpg', 'h5')
                h5_file = h5py.File(h5_path, mode='r')
                h5_map = np.asarray(h5_file['density'])
                if self.roi_path:
                    mask = np.load(self.roi_path)
                    mask_resize = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
                    mask_list.append(torch.from_numpy(mask_resize.copy()).float().unsqueeze(0))
                    h5_map = h5_map * mask
                else:
                    h, w = h5_map.shape
                    mask_resize = np.ones((h // 32 * 4, w // 32 * 4), dtype=np.float32)
                    mask_list.append(torch.from_numpy(mask_resize.copy()).float().unsqueeze(0))
                
                keypoint = np.sum(h5_map)
                keypoint_list.append(keypoint)

            return torch.stack(img_list, dim=0), torch.tensor(keypoint_list), torch.stack(mask_list, dim=0)

        elif self.method == 'val' and 'venice' in self.root_path:
            for q in range(item, item+self.frame_number):
                img_path = self.im_list[q]
                img = Image.open(img_path).convert('RGB')
                img = self.trans(img)
                img_list.append(img)

                h5_path = img_path.replace('jpg', 'h5')
                h5_file = h5py.File(h5_path, mode='r')
                h5_map = np.asarray(h5_file['density'])

                roi_path = img_path.replace('jpg', 'mat')
                mask = loadmat(roi_path)['roi']
                mask_resize = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
                mask_list.append(torch.from_numpy(mask_resize.copy()).float().unsqueeze(0))
                h5_map = h5_map * mask
                keypoint = np.sum(h5_map)

                keypoint_list.append(keypoint)

            return torch.stack(img_list, dim=0), torch.tensor(keypoint_list), torch.stack(mask_list, dim=0)
