import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dataset import DetectionDataset
from unet import UNet
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transform import Compose, Resize, Crop, Pad, Flip
# the proper way to do this is relative import, one more nested package and main.py outside the package
# will sort this out
sys.path.insert(0, os.path.abspath((os.path.dirname(__file__)) + '/../'))
from utils import get_logger, dice_coeff, dice_loss


def eval_net(net, dataset, device):
    net.eval()
    bce_tot, dice_tot = 0., 0.
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            imgs, true_masks = batch
            masks_pred = net(imgs.to(device)).squeeze(1)  # (b, 1, h, w) -> (b, h, w)
            masks_pred = (F.sigmoid(masks_pred) > 0.5).float()

            bce_val, dice_val = criterion(masks_pred.cpu().view(-1), true_masks.view(-1))
            bce_tot += bce_val.item()
            dice_tot += dice_val.item()

    return bce_tot / len(dataset), dice_tot / len(dataset)


def train(net, optimizer, criterion, scheduler, train_dataloader, val_dataloader, logger, writer, args=None, device=None):
    num_batches = len(train_dataloader)

    best_model_info = {'epoch': -1, 'val_bce': np.inf, 'val_dice': 0., 'train_dice': 0., 'train_loss': 0.}

    for epoch in range(args.epochs):
        logger.info('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        net.train()
        if scheduler is not None:
            scheduler.step(epoch)

        epoch_loss = 0.
        tqdm_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        mean_bce, mean_dice = [], []
        for i, batch in tqdm_iter:
            imgs, true_masks = batch
            masks_pred = net(imgs.to(device))
            masks_probs = F.sigmoid(masks_pred)

            bce_val, dice_val = criterion(masks_probs.cpu().view(-1), true_masks.view(-1))
            loss = args.weight_bce * bce_val + (1. - args.weight_bce) * dice_val
            
            mean_bce.append(bce_val.item())
            mean_dice.append(dice_val.item())
            epoch_loss += loss.item()
            tqdm_iter.set_description('mean loss: {:.4f}'.format(epoch_loss / (i + 1)))
            writer.add_scalar('detection/train/batch/loss', loss.item(), i + epoch * num_batches)
            writer.add_scalar('detection/train/batch/bce', mean_bce[-1], i + epoch * num_batches)
            writer.add_scalar('detection/train/batch/dice', mean_dice[-1], i + epoch * num_batches)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()

        logger.info('Epoch finished! Loss: {:.5f} ({:.5f} | {:.5f})'.format(
            epoch_loss / num_batches,
            np.mean(mean_bce),
            np.mean(mean_dice),
        ))
        writer.add_scalar('detection/epoch/loss/train', epoch_loss / num_batches, epoch)
        writer.add_scalar('detection/epoch/bce/train', np.mean(mean_bce), epoch)
        writer.add_scalar('detection/epoch/dice/train', np.mean(mean_dice), epoch)

        val_bce, val_dice = eval_net(net, val_dataloader, device=device)
        if val_dice > best_model_info['val_dice']:
            best_model_info['val_bce'] = val_bce
            best_model_info['val_dice'] = val_dice
            best_model_info['train_loss'] = epoch_loss / num_batches
            best_model_info['epoch'] = epoch
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'CP-best.pth'))
            logger.info('Validation BCE Coeff: {:.5f} (best)'.format(val_bce))
            logger.info('Validation Dice Coeff: {:.5f} (best)'.format(val_dice))
        else:
            logger.info('Validation BCE Coeff: {:.5f} (best {:.5f})'.format(val_bce, best_model_info['val_bce']))
            logger.info('Validation Dice Coeff: {:.5f} (best {:.5f})'.format(val_dice, best_model_info['val_dice']))

        writer.add_scalar('detection/epoch/loss/val', args.weight_bce * val_bce + (1. - args.weight_bce) * val_dice, epoch)
        writer.add_scalar('detection/epoch/bce/val', val_bce, epoch)
        writer.add_scalar('detection/epoch/dice/val', val_dice, epoch)

        torch.save(net.state_dict(), os.path.join(args.output_dir, 'last.pth'))


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=str, default=None ,help='path to the data')
    parser.add_argument('-e', '--epochs', dest='epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=40, type=int, help='batch size')
    parser.add_argument('-s', '--image_size', dest='image_size', default=256, type=int, help='input image size')
    parser.add_argument('-lr', '--learning_rate', dest='lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', dest='weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('-lrs', '--learning_rate_step', dest='lr_step', default=10, type=int, help='learning rate step')
    parser.add_argument('-lrg', '--learning_rate_gamma', dest='lr_gamma', default=0.5, type=float,
                        help='learning rate gamma')
    parser.add_argument('-m', '--model', dest='model', default='unet', choices=('unet',))
    parser.add_argument('-w', '--weight_bce', default=0.5, type=float, help='weight BCE loss')
    parser.add_argument('-l', '--load', dest='load', default=False, help='load file model')
    parser.add_argument('-v', '--val_split', dest='val_split', default=0.8, help='train/val split')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='/tmp/logs/', help='dir to save log and models')
    parser.add_argument('-en', '--exp_name', dest='exp_name', default='baseline', help='name of cur experiment')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.exp_name)

    #
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, 'train.log'))

    root_logs_dir = '/tmp/log_dir/detection/'
    os.makedirs(root_logs_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(root_logs_dir, args.exp_name))

    logger.info('Start training with params:')
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)
    #
    net = UNet() # TODO: to use move novel arch or/and more lightweight blocks (mobilenet) to enlarge the batch_size
    # TODO: img_size=256 is rather mediocre, try to optimize network for at least 512
    logger.info('Model type: {}'.format(net.__class__.__name__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.load:
        net.load_state_dict(torch.load(args.load))
    net.to(device)
    # net = nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # TODO: loss experimentation, fight class imbalance, there're many ways you can tackle this challenge
    criterion = lambda x, y: (nn.BCELoss()(x, y), dice_loss(x, y))
    # TODO: you can always try on plateau scheduler as a default option
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma) \
        if args.lr_step > 0 else None

    # dataset
    # TODO: to work on transformations a lot, look at albumentations package for inspiration
    train_transforms = Compose([
        Crop(min_size=1 - 1 / 3., min_ratio=1.0, max_ratio=1.0, p=0.5),
        Flip(p=0.05),
        Pad(max_size=0.6, p=0.25),
        Resize(size=(args.image_size, args.image_size), keep_aspect=True)
    ])
    # TODO: don't forget to work class imbalance and data cleansing
    val_transforms = Resize(size=(args.image_size, args.image_size))
    
    train_dataset = DetectionDataset(args.data_path, os.path.join(args.data_path, 'train_mask.json'),
                                     transforms=train_transforms)
    val_dataset = DetectionDataset(args.data_path, None, transforms=val_transforms)

    # split dataset into train/val, don't try to do this at home ;)
    train_size = int(len(train_dataset) * args.val_split)
    val_dataset.image_names = train_dataset.image_names[train_size:]
    val_dataset.mask_names = train_dataset.mask_names[train_size:]
    train_dataset.image_names = train_dataset.image_names[:train_size]
    train_dataset.mask_names = train_dataset.mask_names[:train_size]
    
    # TODO: always work with the data: cleaning, sampling
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8,
                                  shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4,
                                shuffle=False, drop_last=False)
    logger.info('Length of train/val=%d/%d', len(train_dataset), len(val_dataset))
    logger.info('Number of batches of train/val=%d/%d', len(train_dataloader), len(val_dataloader))
    
    try:
        train(
            net, optimizer, criterion, scheduler,
            train_dataloader, val_dataloader,
            writer=writer, logger=logger, args=args, device=device,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(args.output_dir, 'INTERRUPTED.pth'))
        logger.info('Saved interrupt')
        sys.exit(0)


if __name__ == '__main__':
    main()
