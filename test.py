import torch
import os
import numpy as np
from dataset.dataset import Crowd
from model.model import HCDI
import argparse
from glob import glob
import cv2
from torch.utils.data import DataLoader
import h5py
from tqdm import tqdm

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='',
                        help='training data directory')
    parser.add_argument('--save-dir', default='',
                        help='model directory')
    parser.add_argument('--roi-path', default='',
                        help='roi path')
    parser.add_argument('--frame-number', type=int, default=3,
                        help='the number of input frames')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu··
    model = HCDI(
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        load_pretrained=False
    )
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(args.save_dir, device))

    if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'dronecrowd' in args.data_dir:
        sum_res = []
        datasets = [Crowd(args.data_dir+'/'+'test'+'/'+file, is_gray=args.is_gray, method='val',
                          frame_number=args.frame_number, roi_path=args.roi_path)
                     for file in tqdm(sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int), desc="创建数据集")]
        dataloader = [DataLoader(datasets[file], 1, shuffle=False, num_workers=8, pin_memory=False)
                      for file in range(len(os.listdir(os.path.join(args.data_dir, 'test'))))]
        file_list = sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int)
        for file in tqdm(range(len(file_list)), desc="处理测试文件"):
            epoch_res = []
            if 'ucsd' in args.data_dir:
                for imgs, keypoints, masks in tqdm(dataloader[file], desc=f"处理文件 {file_list[file]}", leave=False):
                    b, f, c, h, w = imgs.shape
                    assert b == 1, 'the batch size should equal to 1 in validation mode
                    imgs = imgs.to(device)
                    
                    center_mask = masks[:, 1].to(device)
                    if center_mask.dim() > 3:
                        center_mask = center_mask.squeeze(1)
                    
                    with torch.set_grad_enabled(False):
                        output = model(imgs)
                        output = output * center_mask
                        
                        center_count = keypoints[0, 1].item()
                        pred_count = torch.sum(output).item()
                        res = center_count - pred_count
                        epoch_res.append(res)
            else:
                for imgs, keypoints, masks in tqdm(dataloader[file], desc=f"处理文件 {file_list[file]}", leave=False):
                    b, f, c, h, w = imgs.shape
                    assert b == 1, 'the batch size should equal to 1 in validation mode'
                    imgs = imgs.to(device)
                    
                    with torch.set_grad_enabled(False):
                        output = model(imgs)
                        if args.roi_path:
                            mask = np.load(args.roi_path)
                            mask = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
                            mask = torch.tensor(mask).to(device)
                            output = output * mask
                        
                        center_mask = masks[:, 1].to(device)
                        if center_mask.dim() > 3:
                            center_mask = center_mask.squeeze(1)
                        output = output * center_mask
                        
                        center_count = keypoints[0, 1].item()
                        pred_count = torch.sum(output).item()
                        res = center_count - pred_count
                        epoch_res.append(res)
            
            epoch_res = np.array(epoch_res)
            if 'fdst' in args.data_dir or 'ucsd' in args.data_dir:
                test_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'test'+'/'+file_list[file], '*.jpg')),
                                      key=lambda x: int(x.split('/')[-1].split('.')[0]))
            else:
                test_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'test'+'/'+file_list[file], '*.jpg')),
                                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            if len(epoch_res) > len(test_img_list) - args.frame_number + 1:
                epoch_res = epoch_res[:len(test_img_list) - args.frame_number + 1]
            
            print(f"正在处理测试文件 {file_list[file]}:")
            print(f"图像总数: {len(test_img_list)}, 预测结果数量: {len(epoch_res)}")
            
            valid_img_count = min(len(test_img_list), len(epoch_res))
            for j in tqdm(range(valid_img_count), desc="计算测试结果"):
                k = test_img_list[j]
                h5_path = k.replace('jpg', 'h5')
                h5_file = h5py.File(h5_path, mode='r')
                h5_map = np.asarray(h5_file['density'])
                count = np.sum(h5_map)
                
                img_name = os.path.basename(k)
                print(f"{img_name}: 预测误差={epoch_res[j]:.2f}, 真实计数={count:.2f}, 预测计数={count-epoch_res[j]:.2f}")
                
            for e in epoch_res:
                sum_res.append(e)
        
        sum_res = np.array(sum_res)
        rmse = np.sqrt(np.mean(np.square(sum_res)))
        mae = np.mean(np.abs(sum_res))
        log_str = f'Final Test: mae {mae:.4f}, rmse {rmse:.4f}, 总样本数: {len(sum_res)}'
        print(log_str)

    elif 'venice' in args.data_dir:
        sum_res = []
        datasets = [Crowd(args.data_dir+'/'+'test'+'/'+file, is_gray=args.is_gray, method='val',
                          frame_number=args.frame_number, roi_path=args.roi_path)
                     for file in tqdm(sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int), desc="创建数据集")]
        dataloader = [DataLoader(datasets[file], 1, shuffle=False, num_workers=8, pin_memory=False)
                       for file in range(len(os.listdir(os.path.join(args.data_dir, 'test'))))]
        file_list = sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int)
        for file in tqdm(range(len(file_list)), desc="处理测试文件"):
            epoch_res = []  
            for imgs, keypoints, masks in tqdm(dataloader[file], desc=f"处理子目录 {file_list[file]}", leave=False):
                b, f, c, h, w = imgs.shape
                assert b == 1, 'the batch size should equal to 1 in validation mode'
                imgs = imgs.to(device)
                
                center_mask = masks[:, 1].to(device)
                if center_mask.dim() > 3:
                    center_mask = center_mask.squeeze(1)
                
                with torch.set_grad_enabled(False):
                    output = model(imgs)
                    output = output * center_mask
                    
                    center_count = keypoints[0, 1].item()
                    pred_count = torch.sum(output).item()
                    res = center_count - pred_count
                    epoch_res.append(res)
            
            epoch_res = np.array(epoch_res)
            test_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'test'+'/'+file_list[file], '*.jpg')),
                                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            print(f"正在处理Venice子目录 {file_list[file]}:")
            print(f"图像总数: {len(test_img_list)}, 预测结果数量: {len(epoch_res)}")
            
        
            if len(epoch_res) > len(test_img_list) - args.frame_number + 1:
                epoch_res = epoch_res[:len(test_img_list) - args.frame_number + 1]
            
            valid_img_count = min(len(test_img_list), len(epoch_res))
            for j in tqdm(range(valid_img_count), desc="计算测试结果"):
                k = test_img_list[j]
                h5_path = k.replace('jpg', 'h5')
                h5_file = h5py.File(h5_path, mode='r')
                h5_map = np.asarray(h5_file['density'])
                
                mat_path = k.replace('jpg', 'mat')
                mask = None
                try:
                    from scipy.io import loadmat
                    mask = loadmat(mat_path)['roi']
                except Exception as e:
                    print(f"读取ROI掩码失败: {e}")
                    if args.roi_path:
                        mask = np.load(args.roi_path)
                
                if mask is not None:
                    h5_map = h5_map * mask
                
                count = np.sum(h5_map)
                
                img_name = os.path.basename(k)
                print(f"{img_name}: 预测误差={epoch_res[j]:.2f}, 真实计数={count:.2f}, 预测计数={count-epoch_res[j]:.2f}")
                
            for e in epoch_res:
                sum_res.append(e)
        
        sum_res = np.array(sum_res)
        rmse = np.sqrt(np.mean(np.square(sum_res)))
        mae = np.mean(np.abs(sum_res))
        log_str = f'Final Test on Venice: mae {mae:.4f}, rmse {rmse:.4f}, 总样本数: {len(sum_res)}'
        print(log_str)

    else:
        datasets = Crowd(os.path.join(args.data_dir, 'test'), is_gray=args.is_gray, method='val',
                         frame_number=args.frame_number, roi_path=args.roi_path)
        dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False, num_workers=8, pin_memory=False)
        epoch_res = []
        
        for imgs, keypoints, masks in tqdm(dataloader, desc="处理测试数据"):
            b, f, c, h, w = imgs.shape
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            imgs = imgs.to(device)
            
            if masks.dim() > 3:  # 如果是[B, T, 1, H, W]格式
                center_mask = masks[:, 1].to(device)  # [B, 1, H, W]
                if center_mask.dim() > 3:
                    center_mask = center_mask.squeeze(1)  # [B, H, W]
            else:
                center_mask = masks.to(device)  # [B, H, W]
            
            with torch.set_grad_enabled(False):
                output = model(imgs)
                output = output * center_mask
                
                center_count = keypoints[0, 1].item()
                pred_count = torch.sum(output).item()
                res = center_count - pred_count
                epoch_res.append(res)
        
        epoch_res = np.array(epoch_res)
        
        test_img_list = sorted(glob(os.path.join(os.path.join(args.data_dir, 'test'), '*.jpg')),
                               key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"测试图像总数: {len(test_img_list)}, 预测结果数量: {len(epoch_res)}")
        
        if len(epoch_res) > len(test_img_list) - args.frame_number + 1:
            epoch_res = epoch_res[:len(test_img_list) - args.frame_number + 1]
        
        valid_img_count = min(len(test_img_list), len(epoch_res))
        for j in tqdm(range(valid_img_count), desc="计算测试结果"):
            k = test_img_list[j]
            h5_path = k.replace('jpg', 'h5')
            h5_file = h5py.File(h5_path, mode='r')
            h5_map = np.asarray(h5_file['density'])
            if args.roi_path:
                mask = np.load(args.roi_path)
                h5_map = h5_map * mask
            count = np.sum(h5_map)
            
            img_name = os.path.basename(k)
            print(f"{img_name}: 预测误差={epoch_res[j]:.2f}, 真实计数={count:.2f}, 预测计数={count-epoch_res[j]:.2f}")
        
        valid_errors = epoch_res[:valid_img_count]
        rmse = np.sqrt(np.mean(np.square(valid_errors)))
        mae = np.mean(np.abs(valid_errors))
        log_str = f'Final Test: mae {mae:.4f}, rmse {rmse:.4f}, 有效样本数: {valid_img_count}'
        print(log_str)
