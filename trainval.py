from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import nu_net
import torch
import time
import os
import sys


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

        res = self.forward(data_blob, display_intermediate, epoch)

        self.backward()
        # torch.cuda.empty_cache()
        self.tspent['train'] = time.time() - tstart
        self.tspent_sum['train'] += self.tspent['train']
        return res

    # FIXME this function should leave grp/pdg/alpha optional, or make a separate function
    #       to support when one runs on data with no label
    def forward(self, data_blob, display_intermediate=True, epoch=None):
        """
        data: shape=(self._flags.MINIBATCH_SIZE,None,self._flags.NUM_CHANNEL)
        grp: shape=(self._flags.MINIBATCH_SIZE,None,1)
        pdf: shape=(self._flags.MINIBATCH_SIZE,None,1)
        alpha: shape=()
        """
        data = data_blob['data']
        grp = data_blob.get('grp', None)
        pdg = data_blob.get('pdg', None)
        label = data_blob.get('label', None)
        weight = data_blob.get('weight', None)
        alpha = data_blob.get('alpha', 0.0)

        tstart = time.time()
        with torch.set_grad_enabled(self._flags.TRAIN):
            point_cloud = [torch.as_tensor(d).cuda() for d in data]
            if grp is not None:
                grp = [torch.as_tensor(g).cuda() for g in grp]
                for g in grp:
                    g.requires_grad = False
            if pdg is not None:
                pdg = [torch.as_tensor(p).cuda() for p in pdg]
                for p in pdg:
                    p.requires_grad = False
            if label is not None:
                label = [torch.as_tensor(l).cuda() for l in label]
                for l in label:
                    l.requires_grad = False
            if weight is not None:
                weight = [torch.as_tensor(w).cuda() for w in weight]
                for w in weight:
                    w.requires_grad = False
            alpha = torch.as_tensor(alpha).cuda()

            alpha.requires_grad = False

            if self._flags.MODEL_NAME == 'uresnet':
                segmentation = self._net(point_cloud)
                # softmax      = self._softmax(segmentation)
                loss_seg, acc = 0., 0.
                if self._flags.TRAIN:
                    loss_seg, acc = self._criterion(segmentation, label, weight)
                    self._loss = loss_seg
                res = {
                    'segmentation': [segmentation.cpu().detach().numpy()],
                    'softmax': [self._softmax(segmentation).cpu().detach().numpy()],
                    'accuracy': acc,
                    'loss_seg': loss_seg.cpu().item() if not isinstance(loss_seg, float) else loss_seg
                }
                self.tspent['forward'] = time.time() - tstart
                self.tspent_sum['forward'] += self.tspent['forward']
                return res

            # preds, scores, segmentation, tensors = self._net(point_cloud)
            outputs = self._net(point_cloud)
            preds = outputs['predictions']
            scores = outputs['scores']
            segmentation = outputs['segmentation']
            tensors = outputs['tensors']

            if self._flags.MINIBATCH_SIZE > 1:
                new_labels, new_grp, new_pdg = [], [], []
                i = 0
                for p in preds:
                    npoints = len(p[0])
                    new_labels.append(label[0][i:i+npoints])
                    new_grp.append(grp[0][i:i+npoints])
                    new_pdg.append(pdg[0][i:i+npoints])
                    i += npoints
                label = new_labels
                grp = new_grp
                pdg = new_pdg

            loss0, loss1, loss2, loss_cluster, \
            loss_conf, loss_seg = self._criterion(outputs, grp, pdg, alpha,
                                                  label, weight)
            if epoch is not None and epoch <= 2.0:
                self._loss = loss_cluster + 10.0 * loss_seg
            else:
                self._loss = loss_cluster + 10.0 * loss_conf + 10.0 * loss_seg

            intermediate = []
            if display_intermediate and not self._flags.TRAIN:
                # FIXME for several GPUS
                for i, t in enumerate(tensors):
                    t = torch.squeeze(t, dim=-1)
                    if i < len(tensors)-1:  # not for scores!
                        t = self._dist_nn(t.permute([0, 2, 1]))
                    if i == len(tensors)-1:
                        name = 'score'
                    elif i == len(tensors)-2:
                        print(torch.max(t))
                        name = 'final'
                    elif self._flags.MODEL_NAME != 'residual-dgcnn-nofc' and i == len(tensors)-3:
                        name = 'max_pool'
                    elif self._flags.MODEL_NAME != 'residual-dgcnn-nofc' and i == len(tensors)-4:
                        name = 'concat'
                    else:
                        name = 'layer%d' % i
                    intermediate.append((t, name))

            res = {
                'loss0': loss0.cpu().item(),
                'loss1': loss1.cpu().item(),
                'loss2': loss2.cpu().item(),
                'loss_cluster': loss_cluster.cpu().item(),
                'loss_conf': loss_conf.cpu().item(),
                'loss': self._loss.cpu().item(),
                'predictions': [pred.cpu().detach().numpy() for pred in preds],
                'scores': [score.cpu().detach().numpy() for score in scores],
                'tensors': [(t.cpu().detach(), name) for t, name in intermediate],
                'segmentation': [s.cpu().detach().numpy() for s in segmentation],
                'segmentation_pred': [torch.argmax(s, dim=1).cpu().detach().numpy() for s in segmentation],
                'loss_seg': loss_seg.cpu().item()
            }
            self.tspent['forward'] = time.time() - tstart
            self.tspent_sum['forward'] += self.tspent['forward']
            return res

    def initialize(self):
        # To use DataParallel all the inputs must be on devices[0] first
        model = None
        if 'dgcnn' in self._flags.MODEL_NAME:
            model = nu_net.model.NuNet
        elif self._flags.MODEL_NAME == 'nunetC':
            model = nu_net.model.NuNetC
        elif self._flags.MODEL_NAME == 'nunetA':
            model = nu_net.model.NuNetA
        elif self._flags.MODEL_NAME == 'uresnet':
            model = nu_net.model.UResNet
        else:
            raise Exception("Unknown model name provided")

        self.tspent_sum['forward'] = self.tspent_sum['train'] = self.tspent_sum['save'] = 0.
        self.tspent['forward'] = self.tspent['train'] = self.tspent['save'] = 0.

        self._net = nu_net.ops.GraphDataParallel(model(self._flags), device_ids=self._flags.GPUS)

        if self._flags.TRAIN:
            self._net.train().cuda()
        else:
            self._net.eval().cuda()

        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._flags.LEARNING_RATE)
        self._softmax = torch.nn.LogSoftmax(dim=1)

        if self._flags.MODEL_NAME == 'uresnet':
            self._criterion = nu_net.ops.SegmentationLoss(self._flags).cuda()
        else:
            self._criterion = nu_net.ops.DGCNNLoss(self._flags).cuda()
            self._dist_nn = nu_net.ops.DistNN()

        iteration = 0
        if self._flags.MODEL_PATH:
            if not os.path.isfile(self._flags.MODEL_PATH):
                sys.stderr.write('File not found: %s\n' % self._flags.MODEL_PATH)
                raise ValueError
            print('Restoring weights from %s...' % self._flags.MODEL_PATH)
            with open(self._flags.MODEL_PATH, 'rb') as f:
                #checkpoint = torch.load(f, map_location=lambda storage, location: storage)  # FIXME
                checkpoint = torch.load(f)
                self._net.load_state_dict(checkpoint['state_dict'])  # FIXME state_dict
                self._optimizer.load_state_dict(checkpoint['optimizer'])
                iteration = checkpoint['global_step'] + 1
                #iteration = int((self._flags.MODEL_PATH.split('-'))[-1])
            print('Done.')

        return iteration
