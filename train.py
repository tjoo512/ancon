import random
import time
import warnings
import argparse
import shutil
import os.path as osp
import os

import torch.nn.utils.weight_norm as weightNorm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tllib.self_training.pseudo_label import ConfidenceBasedSelfTrainingLoss
from tllib.vision.transforms import MultipleApply
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

import numpy as np
from scipy import optimize
from scipy.optimize import minimize
from sklearn import linear_model
import math
from scipy import stats

from utils_ancon import gce_gen, convert_dataset, gather_features, AnConLoss
from utils import ELRloss #, SourceFreeClassifier
import utils

device = torch.device("cuda")


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

    def forward(self, x: torch.Tensor, return_feature=False):
        """"""
        f = self.pool_layer(self.backbone(x))
        # print("AFTER POOL", f.shape)

        f = self.head_fc(f)
        # print("AFTER FC", f.shape)
        f = self.head_bn(f)
        predictions = self.fc_final(f)
        #predictions = self.head(f)
        if self.training or return_feature:
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


def initialize_nrc(train_target_loader,
          model, args: argparse.Namespace,
          num_classes, fea_bank, score_bank):

    print("="*100)
    print("\tINITIALIZE BANKS")
    model.eval()

    with torch.no_grad():
        for i, xxp in enumerate(train_target_loader):
            idx_u, xx = xxp
            x_t, labels_t = xx[:2]
            # idx_u = idx_u.to(device)

            x_t = x_t.to(device)
            labels_t = labels_t.to(device)

            # compute output
            logits, features = model(x_t, return_feature=True)
            outputs = F.softmax(logits, -1)

            feature_norm = F.normalize(features)
            fea_bank[idx_u] = feature_norm.detach().clone().cpu()
            score_bank[idx_u] = outputs.detach().clone()  #.cpu()
    return fea_bank, score_bank


def get_nrc_loss(features_test, logits_test, idx_u, fea_bank, score_bank, num_classes, nrc_K=5, nrc_KK=5, from_ancon=None):
    softmax_out = F.softmax(logits_test, -1)
    tar_idx = idx_u.cpu()

    with torch.no_grad():
        # Update banks
        output_f_norm = F.normalize(features_test)
        output_f_ = output_f_norm.cpu().detach().clone()
        fea_bank[tar_idx] = output_f_.detach().clone().cpu()
        score_bank[tar_idx] = softmax_out.detach().clone()
        ## 
        distance = output_f_@fea_bank.T
        _, idx_near = torch.topk(distance, dim=-1, largest=True, k=nrc_K+1)
        idx_near = idx_near[:, 1:]  #batch x K
        score_near = score_bank[idx_near]    #batch x K x C

        fea_near = fea_bank[idx_near]  #batch x K x num_dim
        fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0],-1,-1) # batch x n x dim
        distance_ = torch.bmm(fea_near, fea_bank_re.permute(0,2,1))  # batch x K x n
        _,idx_near_near=torch.topk(distance_,dim=-1,largest=True,k=nrc_KK+1)  # M near neighbors for each of above K ones
        idx_near_near = idx_near_near[:,:,1:] # batch x K x M
        tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
        match = (
            idx_near_near == tar_idx_).sum(-1).float()  # batch x K
        weight = torch.where(
            match > 0., match,
            torch.ones_like(match).fill_(0.1))  # batch x K
        weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                    nrc_KK)  # batch x K x M
        weight_kk = weight_kk.fill_(0.1)

        # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
        #weight_kk[idx_near_near == tar_idx_]=0

        score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
        # print('weight_kk shape', weight_kk.shape)
        weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM

        score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                               num_classes)  # batch x KM x C

        score_self = score_bank[tar_idx]

        
    softmax_out = torch.clamp(softmax_out, 1e-4, 1.0 - 1e-4)

    if from_ancon is None:
        score_near_kk = torch.clamp(score_near_kk, 1e-4, 1.0 - 1e-4)
        score_near = torch.clamp(score_near, 1e-4, 1.0 - 1e-4)
    else:
        score_near_kk = torch.clamp(0.3*from_ancon.unsqueeze(1) + 0.7*score_near_kk, 1e-4, 1.0 - 1e-4)
        score_near = torch.clamp(0.3*from_ancon.unsqueeze(1) + 0.7*score_near, 1e-4, 1.0 - 1e-4)

    output_re = softmax_out.unsqueeze(1).expand(-1, nrc_K * nrc_KK,
                                                -1)  # batch x C x 1
    const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
             weight_kk.cuda()).sum(1)) # kl_div here equals to dot product since we do not use log for score_near_kk
    loss = torch.mean(const)

    # nn
    softmax_out_un = softmax_out.unsqueeze(1).expand(-1, nrc_K,
                                                         -1)  # batch x K x C

    # print("softmaxout: ", softmax_out_un)
    # print("score_near: ", score_near)
    loss += torch.mean((
            F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *
            weight.cuda()).sum(1))

    # self, if not explicitly removing the self feature in expanded neighbor then no need for this
    #loss += -torch.mean((softmax_out * score_self).sum(-1))

    msoftmax = softmax_out.mean(dim=0)
    # print("msoftmax: ", msoftmax)
    gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
    # print("gent: ", gentropy_loss)
    loss += gentropy_loss

    return loss, fea_bank, score_bank


def main(args: argparse.Namespace, exp_name: str):
    logger = CompleteLogger(args.log + '-' + exp_name, args.phase)
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
    train_source_transform = utils.get_train_transform(args.train_resizing, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=not args.no_hflip, random_color_jitter=False, resize_size=args.resize_size, norm_mean=args.norm_mean, norm_std=args.norm_std)
    
    weak_augment = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                             random_horizontal_flip=not args.no_hflip,
                                             random_color_jitter=False, resize_size=args.resize_size,
                                             norm_mean=args.norm_mean, norm_std=args.norm_std)
    

    train_target_transform = weak_augment
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)

    print("train_source_transform: ", train_source_transform)
    print("train_target_transform: ", train_target_transform)
    print("val_transform: ", val_transform)

    _, train_target_dataset, _, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_source_transform, val_transform, train_target_transform=train_target_transform)
    
    ## Split the dataset ==============================================================================
    generator = torch.Generator()
    generator.manual_seed(args.val_seed)
    VAL_SPLIT_RATIO = args.val_split
    
    dataset_size = len(train_target_dataset)
    num_val = int(np.floor(0.1 * dataset_size))
    train_target_dataset, valid_dataset = torch.utils.data.random_split(
        train_target_dataset, [dataset_size-num_val, num_val], generator=generator)
    #
    train_target_dataset = convert_dataset(train_target_dataset)

    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)

    if 'vit' in args.arch:
        pool_layer = nn.Identity()
    else:
        pool_layer = None
    # pool_layer = nn.Identity() if args.no_pool else None
    # classifier =SourceFreeClassifier(backbone, num_classes, bottleneck_dim=256).to(device)
    # print(classifier)
    classifier = SourceFreeClassifier(backbone, num_classes, pool_layer=pool_layer, finetune=False).to(device)

    # Load a source pretrained model
    md_name = 'logs/benchmark_erm/OfficeHome_{}-{}'.format(args.source[0], args.arch)
    base_dir = '{}/checkpoints/best.pth'.format(md_name)
    classifier.load_state_dict(torch.load(base_dir))

    if 'vit' in args.arch:
        for k,v in classifier.module.fc_final.named_parameters():
            print(k)
            v.requires_grad= False

        optimizer = SGD(classifier.module.get_parameters(), args.lr, momentum=args.momentum, 
                    weight_decay=1e-3, nesterov=True)
    else:
        for k,v in classifier.fc_final.named_parameters():
            print(k)
            v.requires_grad= False

        optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, 
                    weight_decay=1e-3, nesterov=True)

    # define optimizer and lr scheduler
    max_iter = float(args.epochs * int(len(train_target_loader)))
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + 10.0 * float(x) / max_iter) ** (-args.lr_decay))

    if args.elr_coeff > 0:
        target_early_learning_criterion = ELRloss(0.9, args.elr_coeff, len(train_target_dataset), num_classes)
    else:
        target_early_learning_criterion = AnConLoss(len(train_target_dataset), num_classes, args)
    
    def save_file(name, data, time):
        dir_name = args.log + '-' + exp_name + '/{}'.format(name)
        time_idx = 'ep{}'.format(time)
        fast_name = args.log + '-' + exp_name + '/{}/{}'.format(name, time_idx)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        np.save(fast_name, data)

    accs = []
    acc_name = args.log + '-' + exp_name + '/accs'
    acc1 = utils.validate(test_loader, classifier, args, device)
    accs.append(acc1)
    
    epoch = 512
    # compute scores for the validation
    logits, labels = gather_features(val_loader, classifier)        
    save_file('logit-val', logits, epoch)
    
    logits, labels = gather_features(test_loader, classifier)
    save_file('logit-te', logits, epoch)
    
    if args.nrc:
        fea_bank=torch.randn(len(train_target_dataset), 256)
        score_bank = torch.randn(len(train_target_dataset), num_classes).cuda()

        fea_bank, score_bank = initialize_nrc(train_target_loader,
          classifier, args, num_classes, fea_bank, score_bank)

    else:
        fea_bank = None
        score_bank = None
    
    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr())

        fea_bank, score_bank = train(train_target_loader, classifier,
              optimizer, lr_scheduler, epoch, args, num_classes,
               target_early_learning_criterion, fea_bank, score_bank)
        
        # compute scores for the validation
        logits, labels = gather_features(val_loader, classifier)        
        save_file('logit-val', logits, epoch)
        if epoch == 0:
            save_file('label-val', labels, epoch)
            
        logits, labels = gather_features(test_loader, classifier)
        save_file('logit-te', logits, epoch)
        if epoch == 0:
            save_file('label-te', labels, epoch)
            
        acc1 = utils.validate(test_loader, classifier, args, device)
        accs.append(acc1)

        np.save(acc_name, accs)
        
        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)
        print('='*100)
        print('='*100)
        print('LR: ', args.lr)
        print('Thres: ', args.threshold)
        print(acc1)
        print('='*100)
        print('='*100)

    print("best_acc1 = {:3.1f}".format(best_acc1))
    logger.close()
    

def train(train_target_loader,
          model, optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace,
          num_classes, temporal_consistency, fea_bank, score_bank):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    self_training_losses = AverageMeter('Self Training Loss', ':6.2f')
    pseudo_label_ratios = AverageMeter('Pseudo Label Ratio', ':3.1f')
    pseudo_label_accs = AverageMeter('Pseudo Label Acc', ':3.1f')
    mean_conf = AverageMeter('Conf (weak)', ':3.1f')
    mean_conf_strong = AverageMeter('Conf (strong)', ':3.1f')
    consistency = AverageMeter('Consistency', ':3.1f')

    progress = ProgressMeter(
            len(train_target_loader),
        [batch_time, data_time,   self_training_losses,  pseudo_label_accs, 
         pseudo_label_ratios, mean_conf, mean_conf_strong, consistency],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, xxp in enumerate(train_target_loader):
        idx_u, xx = xxp
        x_t, labels_t = xx[:2]
        idx_u = idx_u.to(device)

        x_t = x_t.to(device)
        labels_t = labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # clear grad
        optimizer.zero_grad()

        # compute output
        y_t_undetached, feature = model(x_t)
        y_t = y_t_undetached.detach()

        confidence, pseudo_labels = torch.softmax(y_t, dim=1).max(dim=1)
        mask = (confidence > args.threshold).float()

        with torch.no_grad():
            norm_y_bar2 = None
            if args.baseline_train or args.elr_coeff > 0:
                smooth_target = F.one_hot(pseudo_labels, num_classes=num_classes).float()
            elif args.ancon:
                # Convert PL to 
                pl = F.one_hot(pseudo_labels, num_classes=num_classes).float()
                y_bar2 = temporal_consistency.get_y_bar(idx_u)
                norm_y_bar2 = y_bar2
                smooth_target = norm_y_bar2 * (args.ls_coeff) + pl * (1 - args.ls_coeff)

            else:
                raise NotImplementedError

        if args.rpl:
            self_training_loss = (gce_gen(y_t_undetached, smooth_target)* mask).mean()
        elif args.nrc:
            nrc_loss, fea_bank, score_bank = get_nrc_loss(feature, y_t_undetached, idx_u, fea_bank, score_bank, num_classes, from_ancon=norm_y_bar2)
            self_training_loss = nrc_loss
        else:
            self_training_loss = (F.cross_entropy(y_t_undetached, smooth_target, reduction='none') * mask).mean()
        
        if args.elr_coeff > 0:
            elr_loss = temporal_consistency(idx_u, y_t_undetached)
            self_training_loss = elr_loss + self_training_loss
            
        self_training_loss = args.trade_off * self_training_loss
        self_training_loss.backward()

        # ## Update 
        if args.ancon:
            temporal_consistency.update_y_bar(idx_u, y_t)
            
        # measure accuracy and record loss
        loss = self_training_loss
        self_training_losses.update(self_training_loss.item(), x_t.size(0))

        mean_conf.update(confidence.mean().item(), x_t.size(0))
#         mean_conf_strong.update(confidence_strong.mean().item(), x_t.size(0))
#         consistency.update((pseudo_labels == pl_strong).float().mean().item(), x_t.size(0))

        # ratio of pseudo labels
        n_pseudo_labels = mask.sum()
        ratio = n_pseudo_labels / x_t.size(0)
        pseudo_label_ratios.update(ratio.item() * 100, x_t.size(0))

        # accuracy of pseudo labels
        if n_pseudo_labels > 0:
            pseudo_labels = pseudo_labels * mask - (1 - mask)
            n_correct = (pseudo_labels == labels_t).float().sum()
            pseudo_label_acc = n_correct / n_pseudo_labels * 100
            pseudo_label_accs.update(pseudo_label_acc.item(), n_pseudo_labels)
#         temp_pseudo_label_accs.update((temp_pseudo_labels == labels_t).float().mean().item(), x_t.size(0))

        # compute gradient and do SGD step
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return fea_bank, score_bank

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FixMatch for Unsupervised Domain Adaptation')
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
    parser.add_argument('--scale', type=float, nargs='+', default=[0.5, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.5 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    parser.add_argument('--auto-augment', default='rand-m10-n2-mstd2', type=str,
                        help='AutoAugment policy (default: rand-m10-n2-mstd2)')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('-ub', '--unlabeled-batch-size', default=32, type=int,
                        help='mini-batch size of unlabeled data (target domain) (default: 32)')
    parser.add_argument('--threshold', default=0.0, type=float,
                        help='confidence threshold')
    parser.add_argument('--lr', '--learning-rate', default=3e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0004, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=30, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='fixmatch',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")

    parser.add_argument("--loss", type=str, default='ce')
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--val-seed", type=int, default=0)
    parser.add_argument("--acc", type=float, default=-1)

    parser.add_argument('--ancon', action='store_true', default=False)
    parser.add_argument("--memory-mode", default='prob', type=str)
    parser.add_argument("--ensemble", default='time', type=str)
    parser.add_argument("--ls-coeff", default=0.3, type=float)

    parser.add_argument('--baseline-train', action='store_true', default=False)
    
    parser.add_argument("--elr-coeff", default=-1, type=float)

    parser.add_argument('--rpl',action='store_true', default=False)
    parser.add_argument('--nrc',action='store_true', default=False)

    args = parser.parse_args()

    exp_name = 'nrc-based-benchmark-bottle256-{}-seed{}-label_smoothing-online_pl'.format(args.arch, args.seed)
    
    if args.baseline_train:
        exp_name += '-baseline'
    elif args.elr_coeff > 0:
        exp_name += '-elr{}'.format(args.elr_coeff)
    elif args.ancon:
        exp_name += '-ancon{}'.format(args.ls_coeff)
    else:
        raise NotImplementedError
            
    if args.rpl:
        exp_name += '-rpl'
    elif args.nrc:
        exp_name += '-nrc'
            
    exp_name += '-lr_{}-wd_{}-epoch_{}'.format(args.lr, 1e-3, args.epochs)
    main(args, exp_name)

