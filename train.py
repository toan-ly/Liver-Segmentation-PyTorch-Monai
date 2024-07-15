import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss

from preprocess import prepare
from utils import train

def main():
    data_dir = 'datasets/Data_Train_Test'
    model_dir = 'model'
    data_in = prepare(data_dir, cache=True)
    
    device = torch.device('cuda:0')
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    # loss_fn = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250, 2510860).to(device))
    loss_fn = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)
    
    train(
        model=model,
        data_in=data_in,
        loss_fn=loss_fn,
        optimizer=optimizer,
        max_epochs=600,
        model_dir=model_dir,
    )

if __name__ == '__main__':
    main()