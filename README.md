# HCDI

official code for Hierarchical Cross-Domain Interaction Network for Video Crowd Counting.

## Install dependencies

torch >= 1.1, torchvision, opencv, numpy, scipy, etc.

## Take training and testing of Bus dataset for example:
### 1. Training
```bash
python train.py --data-dir (dataset path)  --roi-path (roi path)  --crop-height (num)  --crop-width (num)  --max-epoch (num)
```
### 2. Testing
```bash 
python test.py --data-dir (dataset path)  --roi-path (roi path)  --save-dir (weight's path)
```
### 3. Folder Structure
```
Bus/
├── train/
│   ├── ground_truth/
│   │   ├── xxx_0.h5
│   │   ├── xxx_10.h5
│   │   └── ...
│   └── images/
│       ├── xxx_0.jpg
│       ├── xxx_10.jpg
│       └── ...
├── val/
│   ├── ground_truth/
│   └── images/
├── test/
│   ├── ground_truth/
│   └── images/
└── bus_roi.npy
```
