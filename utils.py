import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.utils import first
from monai.losses import DiceLoss

def dice_metric(pred, gt):
    """
    Calculate the Dice coefficient metric.

    Parameters:
        pred (torch.Tensor): Predicted segmentation.
        gt (torch.Tensor): Ground truth segmentation.

    Returns:
        float: Dice coefficient.
    """
    dice_loss = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    return 1 - dice_loss(pred, gt).item()

def calculate_weights(val1, val2):
    """
    Calculate class weights for cross-entropy loss.

    Parameters:
        val1 (int): Number of background pixels.
        val2 (int): Number of foreground pixels.

    Returns:
        torch.Tensor: Class weights.
    """
    count = np.array([val1, val2])
    weights = 1 / (count / count.sum())
    weights /= weights.sum()
    return torch.tensor(weights, dtype=torch.float32)

def train(model, data_in, loss_fn, optimizer, max_epochs, model_dir, test_inteval=1, device=torch.device('cuda:0')):
    """
    Train the model.

    Parameters:
        model (torch.nn.Module): The neural network model.
        data_in (tuple): Tuple of train and test data loaders.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        max_epochs (int): Maximum number of epochs to train.
        model_dir (str): Directory to save the model and metrics.
        test_interval (int): Interval of epochs to evaluate the model on the test set.
        device (torch.device): Device to run the model on.
    """
    best_metric = -1
    best_metric_epoch = -1
    train_loader, test_loader = data_in
    save_loss_train, save_loss_test = []
    save_metric_train, save_metric_test = []

    for epoch in range(max_epochs):
        print(f'{'-' * 10}\nepoch {epoch+1}/{max_epochs}')
        model.train()
        train_epoch_loss = 0
        epoch_metric_train = 0

        for train_step, batch_data in enumerate(train_loader, 1):
            vol, label = batch_data['vol'].to(device), (batch_data['seg'] != 0).to(device)

            optimizer.zero_grad()
            outputs = model(vol)

            train_loss = loss_fn(outputs, label)
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric

            print(f"{train_step}/{len(train_loader)}, Train_loss: {train_loss.item():.4f}, Train_dice: {train_metric:.4f}")

        print('-' * 20)
        train_epoch_loss /= train_step
        epoch_metric_train /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}, Epoch_metric: {epoch_metric_train:.4f}')

        save_loss_train.append(train_epoch_loss)
        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        if (epoch + 1) % test_inteval == 0:
            model.eval()
            test_epoch_loss = 0
            epoch_metric_test = 0

            with torch.no_grad():
                for test_step, test_data in enumerate(test_loader, 1):
                    test_volume, test_label = test_data["vol"].to(device), (test_data["seg"] != 0).to(device)
                    test_outputs = model(test_volume)
                    
                    test_loss = loss_fn(test_outputs, test_label)
                    test_epoch_loss += test_loss.item()
                    test_metric = dice_metric(test_outputs, test_label)
                    epoch_metric_test += test_metric

                test_epoch_loss /= test_step
                epoch_metric_test /= test_step
                print(f'test_loss_epoch: {test_epoch_loss:.4f}, test_dice_epoch: {epoch_metric_test:.4f}')

                save_loss_test.append(test_epoch_loss)
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)
                
                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, 'best_metric_model.pth'))
                    print(f"Model saved at epoch {epoch + 1} with mean dice: {best_metric:.4f}")
        
    print(f"Training completed. Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

def display_patient(data, slice_idx=1, train=True, test=False):
    """
    Display a slice of the patient data.

    Parameters:
        data (DataLoader): DataLoader containing the patient data.
        slice_idx (int): Index of the slice to display.
        train (bool): Flag to display a training sample.
        test (bool): Flag to display a testing sample.
    """
    pat_train, pat_test = data
    pat_train = first(pat_train)
    pat_test = first(pat_test)

    if train:
        plt.figure('Train Visualization', figsize=(12, 6))
        plt.subplot(121)
        plt.title(f'vol {slice_idx}')
        plt.imshow(pat_train['vol'][0, 0, :, :, slice_idx], cmap='gray')

        plt.subplot(122)
        plt.title(f'seg {slice_idx}')
        plt.imshow(pat_train['seg'][0, 0, :, :, slice_idx])

    if test:
        plt.figure('Test Visualization', figsize=(12, 6))
        plt.subplot(121)
        plt.title(f'vol {slice_idx}')
        plt.imshow(pat_test['vol'][0, 0, :, :, slice_idx], cmap='gray')

        plt.subplot(122)
        plt.title(f'seg {slice_idx}')
        plt.imshow(pat_test['seg'][0, 0, :, :, slice_idx])
        plt.show()
        
def calculate_pixels(data):
    """
    Calculate the number of background and foreground pixels in the dataset.

    Parameters:
        data (DataLoader): DataLoader containing the dataset.

    Returns:
        np.ndarray: Array containing the number of background and foreground pixels.
    """
    val = np.zeros((1, 2))

    for batch in tqdm(data):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)
        count = np.append(count, 0) if len(count) == 1 else count
        val += count

    print('The last values:', val)
    return val