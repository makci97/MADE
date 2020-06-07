import gc
import os
import sys
import tqdm
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from argparse import ArgumentParser
from torchvision.transforms import transforms

from contest2.segmentation.unet import UNet
from contest2.segmentation.dataset import SegmentationDataset
from contest2.segmentation.mask_rcnn import get_detector_model
from contest2.segmentation.transform import Compose, Resize, Crop, Pad, Flip

from contest2.utils import get_logger, dice_coeff, dice_loss, collate_fn


def eval_net(net, dataset, device, threshold_score=0.9, threshold_mask=0.05):
    net.eval()
    dice_tot = 0.
    n_masks = 0
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            images, targets = batch
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = net(images)
            for j, prediction in enumerate(predictions):
                true_masks = targets[j]['masks']
                for k, score in enumerate(prediction['scores']):
                    score = float(score.cpu())
                    if score > threshold_score:
                        pred_mask = (prediction['masks'][k][0, :, :].cpu().numpy() > threshold_mask).astype(np.uint8)
                        dice_tot += np.max([dice_coeff(pred_mask, true_mask) for true_mask in true_masks[:, None, None]])
                        n_masks += 1

    return dice_tot / n_masks if n_masks > 0 else 0.


def train(net, optimizer, scheduler, train_dataloader, val_dataloader, logger, writer, args=None, device=None):
    num_batches = len(train_dataloader)

    best_model_info = {'epoch': -1, 'val_bce': np.inf, 'val_dice': 0., 'train_dice': 0., 'train_loss': 0.}

    for epoch in range(args.epochs):
        logger.info('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        net.train()
        writer.add_scalar('segmentation/lr/epoch', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        epoch_loss = 0.
        tqdm_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, batch in tqdm_iter:
            images, targets = batch
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = net(images, targets)
            losses = sum(loss_dict.values())

            epoch_loss += losses.item()
            tqdm_iter.set_description('mean loss: {:.4f}'.format(epoch_loss / (i + 1)))
            writer.add_scalar('segmentation/train/batch/loss', losses.item(), i + epoch * num_batches)

            writer.add_scalar('segmentation/lr/batch', optimizer.state_dict()['param_groups'][0]['lr'], i + epoch * num_batches)

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step(epoch_loss / (i + 1))

            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        logger.info('Epoch finished! Loss: {:.5f}'.format(epoch_loss / num_batches))
        writer.add_scalar('segmentation/epoch/loss/train', epoch_loss / num_batches, epoch)

        val_dice = eval_net(net, val_dataloader, device=device, threshold_score=args.threshold_score, threshold_mask=args.threshold_mask)
        if val_dice > best_model_info['val_dice']:
            best_model_info['val_dice'] = val_dice
            best_model_info['train_loss'] = epoch_loss / num_batches
            best_model_info['epoch'] = epoch
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'CP-best.pth'))
            logger.info('Validation Dice Coeff: {:.5f} (best)'.format(val_dice))
        else:
            logger.info('Validation Dice Coeff: {:.5f} (best {:.5f})'.format(val_dice, best_model_info['val_dice']))

        writer.add_scalar('segmentation/epoch/dice/val', val_dice, epoch)

        torch.save(net.state_dict(), os.path.join(args.output_dir, 'last.pth'))


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=str, default='../data', help='path to the data')
    parser.add_argument('-e', '--epochs', dest='epochs', default=4, type=int, help='number of epochs')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=15, type=int, help='batch size')
    parser.add_argument('-s', '--image_size', dest='image_size', default=256, type=int, help='input image size')
    parser.add_argument('-lr', '--learning_rate', dest='lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', dest='weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('-lrs', '--learning_rate_step', dest='lr_step', default=250, type=int, help='learning rate step')
    parser.add_argument('-lrg', '--learning_rate_gamma', dest='lr_gamma', default=0.5, type=float,
                        help='learning rate gamma')
    parser.add_argument('-ts', '--threshold_score', dest='threshold_score', default=0.9, type=float, help='threshold score')
    parser.add_argument('-tm', '--threshold_mask', dest='threshold_mask', default=0.05, type=float, help='threshold mask')
    parser.add_argument('-l', '--load', dest='load', default=False, help='load file model')
    parser.add_argument('-v', '--val_split', dest='val_split', default=0.95, help='train/val split')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='/tmp/logs/', help='dir to save log and models')
    parser.add_argument('-en', '--exp_name', dest='exp_name', default='baseline', help='name of cur experiment')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.exp_name)

    #
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, 'train.log'))

    root_logs_dir = '/tmp/log_dir/segmentation/'
    os.makedirs(root_logs_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(root_logs_dir, args.exp_name))

    logger.info('Start training with params:')
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)
    #
    # net = UNet() # TODO: to use move novel arch or/and more lightweight blocks (mobilenet) to enlarge the batch_size
    net = get_detector_model()
    # TODO: img_size=256 is rather mediocre, try to optimize network for at least 512
    logger.info('Model type: {}'.format(net.__class__.__name__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.load:
        net.load_state_dict(torch.load(args.load))
    net.to(device)
    # net = nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # TODO: loss experimentation, fight class imbalance, there're many ways you can tackle this challenge
    # criterion = lambda x, y: (nn.BCELoss()(x, y), dice_loss(x, y))
    # TODO: you can always try on plateau scheduler as a default option
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.lr_step, factor=args.lr_gamma, verbose=True,
    ) if args.lr_step > 0 else None

    # dataset
    # TODO: to work on transformations a lot, look at albumentations package for inspiration
    my_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    # train_transforms = Compose([
    #     Crop(min_size=1 - 1 / 3., min_ratio=1.0, max_ratio=1.0, p=0.5),
    #     Flip(p=0.05),
    #     Pad(max_size=0.6, p=0.25),
    #     Resize(size=(args.image_size, args.image_size), keep_aspect=True)
    # ])
    # TODO: don't forget to work class imbalance and data cleansing

    train_dataset = SegmentationDataset(
        args.data_path, os.path.join(args.data_path, 'train.json'),
        transforms=my_transforms, val_split=args.val_split, is_train=True,
    )
    val_dataset = SegmentationDataset(
        args.data_path, os.path.join(args.data_path, 'train.json'),
        transforms=my_transforms, val_split=args.val_split, is_train=False,
    )


    # train_dataset.marks = train_dataset.marks[:20]
    # val_dataset.marks = val_dataset.marks[:4]

    # TODO: always work with the data: cleaning, sampling
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=8,
        shuffle=True, drop_last=True, collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=4,
        shuffle=False, drop_last=False, collate_fn=collate_fn,
    )
    logger.info('Length of train/val=%d/%d', len(train_dataset), len(val_dataset))
    logger.info('Number of batches of train/val=%d/%d', len(train_dataloader), len(val_dataloader))
    
    try:
        train(
            net, optimizer, scheduler,
            train_dataloader, val_dataloader,
            writer=writer, logger=logger, args=args, device=device,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(args.output_dir, 'INTERRUPTED.pth'))
        logger.info('Saved interrupt')
        sys.exit(0)
    except:
        torch.save(net.state_dict(), os.path.join(args.output_dir, 'EXCEPTION.pth'))
        logger.info('Saved exception')
        sys.exit(1)


if __name__ == '__main__':
    main()
