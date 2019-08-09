import argparse
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
import json
from models import UNet11
from dataset import WaterDataset
import utilsTrain
import cv2


from transformsdata import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        VerticalFlip)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--limit', type=int, default=10000, help='number of images in epoch')
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)

    arg('--model', type=str, default='UNet11', choices=['UNet11'])

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1
    if args.model == 'UNet11':
        model = UNet11(num_classes=num_classes)
    """elif args.model == 'other':
        model = other(num_classes=num_classes, pretrained=True)
    elif args.model == 'other':
        model = other(num_classes=num_classes, pretrained=True)
    """
    else:
        model = UNet11(num_classes=num_classes, input_channels=4)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    loss = calc_loss(pred, target, metrics, bce_weight=0.5):

    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None, limit=None):
        return DataLoader(
            dataset=WaterDataset(file_names, transform=transform, limit=limit),
            shuffle=shuffle,            
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available() #### in process arguments
        )

    #train_file_names, val_file_names = get_split(args.fold)
    data_path = Path('data')
    train_path= data_path/'train'/'images'
    val_path= data_path/'val'/'images'

    train_file_names = np.array(sorted(list(train_path.glob('*.npy'))))
    print(len(train_file_names))
    val_file_names = np.array(sorted(list(val_path.glob('*.npy'))))

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    train_transform = DualCompose([
        CenterCrop(512),
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
    
        ImageOnly(Normalize())
    ])

    val_transform = DualCompose([
        CenterCrop(512),
        ImageOnly(Normalize())
    ])

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform, limit=args.limit)
    valid_loader = make_loader(val_file_names, transform=val_transform)

    
    dataloaders = {
    'train': train_loader, 'val': valid_loader
    }

    dataloaders_sizes = {
    x: len(dataloaders[x]) for x in dataloaders.keys()
    }

    
    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    utilsTrain.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        fold=args.fold
    )


if __name__ == '__main__':
    main()
