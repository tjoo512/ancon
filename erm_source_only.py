import random
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import time
import utils
# from tllib.modules.classifier import Classifier
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance
from tllib.utils.data import ForeverDataIterator

import torch.nn.functional as F

import torch.nn.utils.weight_norm as weightNorm

from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SourceFreeClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck=None,
                 bottleneck_dim=None, head=None, finetune=True, pool_layer=None):
        super(SourceFreeClassifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # After Backbone, we have
        #    - FC (not freeze) - BC (not freeze) - FC (freeze) - WN (freeze)

        if pool_layer is not None:
            print('='*100)
            print('='*100)
            print('='*100)
            print('='*100)
            print(' ARE YOU USING VIT')
            self.pool_layer = nn.Identity()
            h_dim = backbone.out_features
            print('='*100)
            print('='*100)
            print('='*100)
            print('='*100)

            self.head_fc = nn.Identity() # nn.Linear(h_dim, b_dim)
            self.head_bn = nn.Identity() # nn.BatchNorm1d(b_dim, affine=True)
            self.fc_final = weightNorm(nn.Linear(h_dim, num_classes))
        else:
            print("POOL LAYER IS NONE")
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
            h_dim = backbone.out_features
            print("="*100)
            print("="*100)
            print("="*100)
            print('h dim: ', h_dim)
            print("="*100)
            print("="*100)
            print("="*100)
            # h_dim = 2048
            b_dim = 256
            self.head_fc = nn.Linear(h_dim, b_dim)
            self.head_bn = nn.BatchNorm1d(b_dim, affine=True)
            self.fc_final = weightNorm(nn.Linear(b_dim, num_classes))
        # self.head_fc = nn.Linear(backbone.out_features, 256)

        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor):
        """"""
        f = self.pool_layer(self.backbone(x))
        # print("AFTER POOL", f.shape)

        f = self.head_fc(f)
        # print("AFTER FC", f.shape)
        f = self.head_bn(f)
        predictions = self.fc_final(f)
        #predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr},
            {"params": self.head_fc.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head_bn.parameters(), "lr": 1.0 * base_lr},
            {"params": self.fc_final.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


def main(args):
    logger = CompleteLogger(args.log + '-' + args.arch, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)

    ##Split
    dataset_size = len(train_source_dataset)
    num_val = int(np.floor(0.1 * dataset_size))
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_source_dataset, [dataset_size-num_val, num_val])
    # val_loader_sp = DataLoader(val_dataset_sp, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


    ##
    train_source_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)

    #pool_layer = nn.Identity() if args.no_pool else None
    if 'vit' in args.arch:
        pool_layer = nn.Identity()
    else:
        pool_layer = None
    
    classifier = SourceFreeClassifier(backbone, num_classes, pool_layer=pool_layer, finetune=False).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=0.9, weight_decay=1e-3, nesterov=True)
    # lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    max_iter = float(args.epochs * int(len(train_source_loader)))
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + 10.0 * float(x) / max_iter) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    if 'vit' in args.arch:
        classifier = torch.nn.DataParallel(classifier)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_source_loader, classifier, optimizer, lr_scheduler, epoch, args, device)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('epoch_{}'.format(epoch)))
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    # print("best_acc1 = {:3.1f}".format(best_acc1))

    # # evaluate on test set
    # classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    # acc1 = utils.validate(test_loader, classifier, args, device)
    # print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_loader, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        len(train_source_loader),
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    # for i in range(args.iters_per_epoch):
        # x_s, labels_s = next(train_source_iter)[:2]
    for i, xx in enumerate(train_source_loader):
        x_s, labels_s = xx[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source Only for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='src_only',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)



