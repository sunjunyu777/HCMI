# STDNet

official code for Spatial and Temporal-frequency Domain Fusion Network For Video Crowd Counting.

## Install dependencies

torch >= 1.0, torchvision, opencv, numpy, scipy, etc.

## Take training and testing of Bus dataset for example:
1.Training
python train.py --data-dir (dataset path)  --roi-path (roi path)  --crop-height (num)  --crop-width (num)  --max-epoch 200

2.Testing  
python test.py --data-dir (dataset path)  --roi-path (roi path)  --save-dir checkpoint.pth

3.Set the folder structure should look like this:
Bus
├──train
    ├──ground_truth    
        ├──xxx_0.h5
        ├──xxx_10.h5
        ├──....
    ├──images
        ├──xxx_0.jpg
        ├──xxx_10.jpg
├──val
    ├──ground_truth
    ├──images
├──test
    ├──ground_truth
    ├──images
├──bus_roi.npy

