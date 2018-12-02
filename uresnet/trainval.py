from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import time
import os
import sys
from uresnet.ops import GraphDataParallel
import uresnet.models as models
import uresnet.utils as utils

class trainval(object):
    def __init__(self, flags):
        self._flags  = flags
        self.tspent = {}
        self.tspent_sum = {}

    def backward(self):
        self._optimizer.zero_grad()
        self._loss.backward()
        self._optimizer.step()
        # self._loss = None
        # del self._loss
        # torch.cuda.empty_cache()

    def save_state(self, iteration):
        tstart = time.time()
        filename = '%s-%d.ckpt' % (self._flags.WEIGHT_PREFIX, iteration)
        torch.save({
            'global_step': iteration,
            'state_dict': self._net.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }, filename)
        self.tspent['save'] = time.time() - tstart

    def train_step(self, data_blob,
                   display_intermediate=True, epoch=None):
        tstart = time.time()
        res_combined = {}
        for idx in range(len(data_blob['data'])):
            blob = {}
            for key in data_blob.keys(): blob[key] = data_blob[key][idx]
            res = self.forward(blob, display_intermediate, epoch)
            for key in res.keys():
                if not key in res_combined: res_combined[key]=[res[key]]
                else: res_combined[key].append(res[key])
        self.backward()
        # torch.cuda.empty_cache()
        self.tspent['train'] = time.time() - tstart
        self.tspent_sum['train'] += self.tspent['train']
        return res_combined

    def forward(self, data_blob, display_intermediate=True, epoch=None):
        """
        data/label/weight are lists of size batch size.
        For sparse uresnet:
        data[0]: shape=(N, 5)
        where N = total nb points in all events of the minibatch
        For dense uresnet:
        data[0]: shape=(minibatch size, spatial size, spatial size, spatial size, channel)
        """
        data = data_blob['data']
        label = data_blob.get('label', None)
        weight = data_blob.get('weight', None)

        tstart = time.time()
        with torch.set_grad_enabled(self._flags.TRAIN):
            # Segmentation
            data = [torch.as_tensor(d).cuda() for d in data]
            segmentation, = self._net(data)
            if not isinstance(segmentation, list):
                segmentation = [segmentation]

            # If label is given, compute the loss
            loss_seg, acc = 0., 0.
            if label is not None:
                label = [torch.as_tensor(l).cuda() for l in label]
                for l in label:
                    l.requires_grad = False
                # Weight is optional for loss
                if weight is not None:
                    weight = [torch.as_tensor(w).cuda() for w in weight]
                    for w in weight:
                        w.requires_grad = False
                loss_seg, acc = self._criterion(segmentation, data, label, weight)
                if self._flags.TRAIN:
                    self._loss = loss_seg
            res = {
                'segmentation': [s.cpu().detach().numpy() for s in segmentation],
                'softmax': [self._softmax(s).cpu().detach().numpy() for s in segmentation],
                'accuracy': acc,
                'loss_seg': loss_seg.cpu().item() if not isinstance(loss_seg, float) else loss_seg
            }
            self.tspent['forward'] = time.time() - tstart
            self.tspent_sum['forward'] += self.tspent['forward']
            return res

    def initialize(self):
        # To use DataParallel all the inputs must be on devices[0] first
        model = None
        if self._flags.MODEL_NAME == 'uresnet_sparse':
            model = models.SparseUResNet
            self._criterion = models.SparseSegmentationLoss(self._flags).cuda()
        elif self._flags.MODEL_NAME == 'uresnet_dense':
            model = models.DenseUResNet
            self._criterion = models.DenseSegmentationLoss(self._flags).cuda()
        else:
            raise Exception("Unknown model name provided")

        self.tspent_sum['forward'] = self.tspent_sum['train'] = self.tspent_sum['save'] = 0.
        self.tspent['forward'] = self.tspent['train'] = self.tspent['save'] = 0.

        self._net = GraphDataParallel(model(self._flags), device_ids=self._flags.GPUS)

        if self._flags.TRAIN:
            self._net.train().cuda()
        else:
            self._net.eval().cuda()

        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._flags.LEARNING_RATE)
        self._softmax = torch.nn.LogSoftmax(dim=1)

        #self._criterion = SegmentationLoss(self._flags).cuda()

        iteration = 0
        if self._flags.MODEL_PATH:
            if not os.path.isfile(self._flags.MODEL_PATH):
                sys.stderr.write('File not found: %s\n' % self._flags.MODEL_PATH)
                raise ValueError
            print('Restoring weights from %s...' % self._flags.MODEL_PATH)
            with open(self._flags.MODEL_PATH, 'rb') as f:
                checkpoint = torch.load(f)
                self._net.load_state_dict(checkpoint['state_dict'])
                self._optimizer.load_state_dict(checkpoint['optimizer'])
                iteration = checkpoint['global_step'] + 1
            print('Done.')

        return iteration
