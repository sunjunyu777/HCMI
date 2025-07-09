# STDNet

official code for Spatial and Temporal-frequency Domain Fusion Network For Video Crowd Counting.

## Install dependencies

torch >= 1.0, torchvision, opencv, numpy, scipy, etc.

## Take training and testing of Bus dataset for example:
# Training
python train.py --dataset bus --epochs 100

# Testing  
python test.py --dataset bus --model checkpoint.pth
```

## Results

| Method | Bus MAE | Bus RMSE | Canteen MAE | Canteen RMSE |
|--------|---------|----------|-------------|--------------|
| DACM   | 1.35    | 1.78     | 2.45        | 3.12         |

## Citation

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```
