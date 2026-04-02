SCA-Net

Recommended environment
- Python 3.10.15
- PyTorch 2.10.0
- CUDA 12.8

Additional packages
- torchvision version matched to your PyTorch build
- timm
- numpy
- pillow
- scipy

Notes
- Install the CUDA-enabled PyTorch build that matches your local CUDA toolkit or driver.
- The default training setup in train.py uses batch size 8 and gradient accumulation 4.
- The dataset settings are consistent with MEGANet (https://github.com/UARK-AICV/MEGANet).
- The checkpoint will be released after the paper is accepted.


Basic commands
- Train: python train.py
- Predict: python predict.py --checkpoint ./checkpoints/sca_net/epoch_100.pth
- Evaluate: python evaluate.py --pred-root ./results/sca_net
