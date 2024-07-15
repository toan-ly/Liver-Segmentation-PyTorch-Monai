# Liver Segmentation Using PyTorch and Monai

This project focuses on segmenting the liver from medical images using a 3D UNet implemented with PyTorch and MONAI. The model is trained and evaluated on the dataset from the Meddical Segmentation Decathlon.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Training](#training)
- [Demo](#demo)
- [License](#license)

## Introduction
Medical image segmentation is a critical task in healthcare, aiding in diagnostics and treatment planning. This project leverages the power of deep learning to accurately segment the liver from 3D medical images.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU for acceleration (optional but recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

Requirements File
```
monai
torch
matplotlib
numpy
glob
```

## Training
```python
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
```

## Demo
![image](https://github.com/user-attachments/assets/09bb9c22-c683-4024-ba3f-080962bb90e3)
![image](https://github.com/user-attachments/assets/d28d19df-974b-4e3f-8cfb-3bb360cfa496)

