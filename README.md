# HCDI

official code for Hierarchical Cross-Domain Interaction Network for Video Crowd Counting.

### Dataset

* Bus: [BaiduNetDisk](https://pan.baidu.com/s/18YosH0MWtXZQZ5xf3Y9y_A?pwd=nknu).
* Classroom: [BaiduNetDisk](https://pan.baidu.com/s/1fasDO6quWNLVuG_yVMmCCQ?pwd=eehx).

## Install dependencies

torch >= 1.1, pytorch_wavelets, torchvision, opencv, numpy, scipy, etc.

## Take training and testing of Bus dataset for example:
### 1. download Bus and generate the ground truth maps.

### 2. Set the folder structure should look like this:
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
### 3. Training
```bash
python train.py --data-dir (dataset path)  --roi-path (roi path) 
```
### 4. Testing
```bash 
python test.py --data-dir (dataset path)  --roi-path (roi path)  --save-dir (weight's path)
```

