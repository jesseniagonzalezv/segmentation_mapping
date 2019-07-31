import argparse
from pathlib import Path
from validation import validation_binary
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
import json
from models import UNet11
from loss import LossBinary
from dataset import WaterDataset
import utils

from prepare_train_val import get_split

from transforms import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        RandomHueSaturationValue,
                        VerticalFlip)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=0.3, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    #arg('--fold', type=int, help='fold', default=0)
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
    """elif args.model == 'UNet11':
        model = UNet11(num_classes=num_classes, pretrained=True)
    elif args.model == 'UNet16':
        model = UNet16(num_classes=num_classes, pretrained=True)
    elif args.model == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes, pretrained=True)
    elif args.model == 'AlbuNet':
        model = AlbuNet34(num_classes=num_classes, pretrained=True) """
    else:
        model = UNet11(num_classes=num_classes, input_channels=4)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    loss = LossBinary(jaccard_weight=args.jaccard_weight)

    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None, limit=None):
        return DataLoader(
            dataset=WaterDataset(file_names, transform=transform, limit=limit),
            shuffle=shuffle,
            
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(args.fold)

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

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=validation_binary,
        fold=args.fold
    )


if __name__ == '__main__':
    main()
