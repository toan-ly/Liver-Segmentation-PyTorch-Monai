import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from monai.transforms import (
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.utils import first

def plot_training_results(model_dir):
    train_loss = np.load(os.path.join(model_dir, 'loss_train.npy'))
    train_metric = np.load(os.path.join(model_dir, 'metric_train.npy'))
    test_loss = np.load(os.path.join(model_dir, 'loss_test.npy'))
    test_metric = np.load(os.path.join(model_dir, 'metric_test.npy'))

    plt.figure("Training and Validation Results", (12, 6))

    plt.subplot(2, 2, 1)
    plt.title("Train Dice Loss")
    plt.xlabel("Epoch")
    plt.plot(range(1, len(train_loss) + 1), train_loss)

    plt.subplot(2, 2, 2)
    plt.title("Train Metric (Dice)")
    plt.xlabel("Epoch")
    plt.plot(range(1, len(train_metric) + 1), train_metric)

    plt.subplot(2, 2, 3)
    plt.title("Test Dice Loss")
    plt.xlabel("Epoch")
    plt.plot(range(1, len(test_loss) + 1), test_loss)

    plt.subplot(2, 2, 4)
    plt.title("Test Metric (Dice)")
    plt.xlabel("Epoch")
    plt.plot(range(1, len(test_metric) + 1), test_metric)

    plt.tight_layout()
    plt.show()
    
def get_data_paths(data_dir):
    path_train_volumes = sorted(glob(os.path.join(data_dir, "TrainVolumes", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(data_dir, "TrainSegmentation", "*.nii.gz")))

    path_test_volumes = sorted(glob(os.path.join(data_dir, "TestVolumes", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(data_dir, "TestSegmentation", "*.nii.gz")))

    train_files = [{"vol": vol, "seg": seg} for vol, seg in zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": vol, "seg": seg} for vol, seg in zip(path_test_volumes, path_test_segmentation)]

    return train_files, test_files[:9]

def create_test_transforms():
    keys = ['vol', 'seg']
    return Compose(
        [
            LoadImaged(keys=keys),
            AddChanneld(keys=keys),
            Spacingd(keys=keys, pixdim=(1.5, 1.5, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=keys, axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=keys, source_key='vol'),
            Resized(keys=keys, spatial_size=[128, 128, 64]),
            ToTensord(keys=keys),
        ]
    )

def load_model(device, model_dir):
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth")))
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    roi_size = (128, 128, 64)
    sw_batch_size = 4

    with torch.no_grad():
        test_patient = first(test_loader)
        t_volume = test_patient['vol']
        #t_segmentation = test_patient['seg']
        test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
        sigmoid_activation = Activations(sigmoid=True)
        test_outputs = sigmoid_activation(test_outputs)
        test_outputs = test_outputs > 0.53

        for i in range(32):
            plt.figure("Check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"Image {i}")
            plt.imshow(test_patient["vol"][0, 0, :, :, i], cmap="gray")

            plt.subplot(1, 3, 2)
            plt.title(f"Label {i}")
            plt.imshow(test_patient["seg"][0, 0, :, :, i] != 0)

            plt.subplot(1, 3, 3)
            plt.title(f"Output {i}")
            plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i])

            plt.show()
            
def main():
    data_dir = 'datasets/Data_Train_Test'
    model_dir = 'model'

    plot_training_results(model_dir)

    train_files, test_files = get_data_paths(data_dir)
    test_transforms = create_test_transforms()
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

    device = torch.device("cuda:0")
    model = load_model(device, model_dir)

    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
