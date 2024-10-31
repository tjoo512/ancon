import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np

import torch.nn.utils.weight_norm as weightNorm


def gather_features(selected_loader, model):
    model.eval()
    with torch.no_grad():
        start_test = True
        for i, dx in enumerate(selected_loader):
            inputs, labels = dx[:2]
            inputs = inputs.cuda()
            logit = model(inputs, return_feature=False)

            if start_test:
                outputs_ = logit.float().cpu()
                labels_ = labels
                start_test = False
            else:
                outputs_ = torch.cat((outputs_, logit.float().cpu()), 0)
                labels_ = torch.cat((labels_, labels), 0)
    return outputs_, labels_

def gce_gen(logits, target, q=0.8):
    """Generalized cross entropy.
    Reference: https://arxiv.org/abs/1805.07836
    """
    probs = F.softmax(logits, dim=1)
    probs_with_correct_idx = (probs * target).sum(-1)
    loss = (1.0 - probs_with_correct_idx**q) / q
    return loss

def entropy(input_):
    epsilon = 1e-5
    entropy = input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=-1)
    return entropy


def convert_dataset(dataset):
    """
    Converts a dataset which returns (img, label) pairs into one that returns (index, img, label) triplets.
    """
    class DatasetWrapper:
        def __init__(self):
            self.dataset = dataset

        def __getitem__(self, index):
            return index, self.dataset[index]

        def __len__(self):
            return len(self.dataset)

    return DatasetWrapper()

class LabelStorage(object):
    def __init__(self, n_labeled_samples, num_classes, args):
        self.device = 'cuda'
        self.num_classes = num_classes
        self.len = n_labeled_samples
        self.labels = torch.zeros(self.len, dtype=torch.int).to(self.device).long()
    def update(self, idxes, out):
        self.labels[idxes] = out
    def get_pl(self, idxes):
        return self.labels[idxes]

class AnConLoss(object):
    def __init__(self, n_labeled_samples, num_classes, args):
        self.device = 'cuda'
        self.num_classes = num_classes
        self.len = n_labeled_samples
        self.memory_mode = args.memory_mode
        self.ensemble = args.ensemble

        self._init_memory()

    def _init_memory(self):
        self.scores = torch.zeros((self.len, 1), dtype=torch.float).to(self.device)
        self.y_bar = torch.zeros((self.len, self.num_classes), dtype=torch.float).to(self.device)
        self.mean_conf = None

    def update_y_bar(self, idxes, out):
        ## Preproc predictions
        probs = F.softmax(out.detach(), 1)
        conf, pl = probs.max(dim=1)
        obj = F.one_hot(pl, self.num_classes)
        
        ## Get EMA Conf
        vval = torch.mean(conf)
        if self.mean_conf is None:
            self.mean_conf = vval
        else:
            self.mean_conf = 0.9 * self.mean_conf + 0.1 * vval
            
        ## Update Memory
        med_thres = self.mean_conf
        is_uncertain = (conf< med_thres)
        obj[is_uncertain] = 0.
        self.y_bar[idxes] += obj
        

    def get_y_bar(self, idxes):
        vals = self.y_bar[idxes]
        return vals
    
    


