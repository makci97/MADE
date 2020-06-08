import os
import sys
import tqdm
import numpy as np
import editdistance

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import ctc_loss, log_softmax

from argparse import ArgumentParser
from torchvision.transforms import Compose, transforms

from contest2.utils import get_logger
from contest2.recognition.common import abc
from contest2.recognition.model import RecognitionModel
from contest2.recognition.dataset import RecognitionDataset
from contest2.recognition.transform import Compose, Resize, Pad, Rotate


def eval(net, data_loader, device):
    count, tp, avg_ed = 0, 0, 0
    iterator = tqdm.tqdm(data_loader)
    
    with torch.no_grad():
        for batch in iterator:
            images = batch['images'].to(device)
            out = net(images, decode=True)
            gt = (batch['seqs'].numpy() - 1).tolist()
            lens = batch['seq_lens'].numpy().tolist()
            
            pos, key = 0, ''
            for i in range(len(out)):
                gts = ''.join(abc[c] for c in gt[pos:pos + lens[i]])
                pos += lens[i]
                if gts == out[i]:
                    tp += 1
                else:
                    avg_ed += editdistance.eval(out[i], gts)
                count += 1
    
    acc = tp / count
    avg_ed = avg_ed / count
    
    return acc, avg_ed


def train(net, optimizer, criterion, scheduler, train_dataloader, val_dataloader, writer, args, logger, device):
    # TODO: try different techniques for fighting overfitting of the trained network

    num_batches = len(train_dataloader)

    best_acc_val = -1
    for epoch in range(args.epochs):
        logger.info('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        net.train()
        writer.add_scalar('recognition/lr/epoch', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            
        loss_mean = []
        train_iter = tqdm.tqdm(train_dataloader, total=len(train_dataloader))
        for i, batch in enumerate(train_iter):
            images = batch['images'].to(device)
            seqs = batch['seqs']
            seq_lens = batch['seq_lens']
            
            seqs_pred = net(images).cpu()
            log_probs = log_softmax(seqs_pred, dim=2)
            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

            # TODO: ctc_loss is not an only choice here
            loss = criterion(log_probs, seqs, seq_lens_pred, seq_lens) #/ args.batch_size

            loss_mean.append(loss.item())
            train_iter.set_description('mean loss: {:.4f}'.format(np.mean(loss_mean[-args.lr_step:])))
            writer.add_scalar('recognition/train/batch/loss', loss_mean[-1], i + epoch * num_batches)

            writer.add_scalar('recognition/lr/batch', optimizer.state_dict()['param_groups'][0]['lr'], i + epoch * num_batches)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step(np.mean(loss_mean[-args.lr_step:]))
            
        logger.info('Epoch finished! Loss: {:.5f}'.format(np.mean(loss_mean)))
        writer.add_scalar('recognition/epoch/loss/train', np.mean(loss_mean), epoch)

        net.eval()
        acc_val, acc_ed_val = eval(net, val_dataloader, device=device)

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'cp-best.pth'))
            logger.info('Valid acc: {:.5f}, acc_ed: {:.5f} (best)'.format(acc_val, acc_ed_val))
        else:
            logger.info('Valid acc: {:.5f}, acc_ed: {:.5f} (best {:.5f})'.format(acc_val, acc_ed_val, best_acc_val))

        writer.add_scalar('recognition/epoch/acc/val', acc_val, epoch)
        writer.add_scalar('recognition/epoch/acc_ed/val', acc_ed_val, epoch)

        torch.save(net.state_dict(), os.path.join(args.output_dir, f'epoch_{epoch}.pth'))
        torch.save(net.state_dict(), os.path.join(args.output_dir, 'cp-last.pth'))
    logger.info('Best valid acc: {:.5f}'.format(best_acc_val))


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=str, default='../data', help='path to the data')
    parser.add_argument('--epochs', '-e', dest='epochs', type=int, help='number of train epochs', default=4)
    parser.add_argument('--batch_size', '-b', dest='batch_size', type=int, help='batch size', default=128) # 1o024
    parser.add_argument('--weight_decay', '-wd', dest='weight_decay', type=float, help='weight_decay', default=1e-5)
    parser.add_argument('--lr', '-lr', dest='lr', type=float, help='lr', default=3e-4)
    parser.add_argument('--lr_step', '-lrs', dest='lr_step', type=int, help='lr step', default=250)
    parser.add_argument('--lr_gamma', '-lrg', dest='lr_gamma', type=float, help='lr gamma factor', default=0.5)
    parser.add_argument('--input_wh', '-wh', dest='input_wh', type=str, help='model input size', default='320x64')
    parser.add_argument('--rnn_dropout', '-rdo', dest='rnn_dropout', type=float, help='rnn dropout p', default=0.1)
    parser.add_argument('--rnn_num_directions', '-rnd', dest='rnn_num_directions', type=int, help='bi', default=1)
    parser.add_argument('--augs', '-a', dest='augs', type=float, help='degree of geometric augs', default=0)
    parser.add_argument('--load', '-l', dest='load', type=str, help='pretrained weights', default=None)
    parser.add_argument('-v', '--val_split', dest='val_split', default=0.95, help='train/val split')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='logs_rec/',
                        help='dir to save log and models')
    parser.add_argument('-en', '--exp_name', dest='exp_name', default='baseline', help='name of cur experiment')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, 'train.log'))

    root_logs_dir = 'log_dir/recognition/'
    os.makedirs(root_logs_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(root_logs_dir, args.exp_name))

    logger.info('Start training with params:')
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    w, h = list(map(int, args.input_wh.split('x')))
    net = RecognitionModel(
        model_name='resnet18', input_size=(h, w), output_len=24,
        dropout=args.rnn_dropout, num_directions=args.rnn_num_directions,
    )
    if args.load is not None:
        net.load_state_dict(torch.load(args.load))
    net = net.to(device)
    criterion = ctc_loss
    logger.info('Model type: {}'.format(net.__class__.__name__))
    
    # TODO: try other optimizers and schedulers
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.lr_step, factor=args.lr_gamma, verbose=True,
    ) if args.lr_step is not None else None
    
    # dataset
    # TODO: again, augmentations is the key for many tasks
    my_ocr_transforms = transforms.Compose([
        Resize(size=(w, h)),
        transforms.ToTensor()
    ])
    # TODO: don't forget to work on data cleansing
    train_dataset = RecognitionDataset(
        args.data_path, os.path.join(args.data_path, 'train_rec.json'),
        abc=abc, transforms=my_ocr_transforms, val_split=args.val_split, is_train=True,
    )
    val_dataset = RecognitionDataset(
        args.data_path, os.path.join(args.data_path, 'train_rec.json'),
        abc=abc, transforms=my_ocr_transforms, val_split=args.val_split, is_train=False,
    )


    # train_dataset.marks = train_dataset.marks[:20]
    # val_dataset.marks = val_dataset.marks[:4]

    
    # TODO: maybe implement batch_sampler for tackling imbalance, which is obviously huge in many respects
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, collate_fn=train_dataset.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, collate_fn=val_dataset.collate_fn,
    )
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
    except:
        torch.save(net.state_dict(), os.path.join(args.output_dir, 'EXCEPTION.pth'))
        logger.info('Saved exception')
        raise
    

if __name__ == '__main__':
    sys.exit(main())
