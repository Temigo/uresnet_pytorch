from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from sklearn.cluster import DBSCAN


class CSVData:

    def __init__(self,fout):
        self._fout = fout
        self._str  = None
        self._dict = {}

    def record(self, keys, vals):
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]

    def write(self):

        if self._str is None:
            self._fout=open(self._fout,'w')
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str+='{:f}'
            self._fout.write('\n')
            self._str+='\n'

        self._fout.write(self._str.format(*(self._dict.values())))

    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()


def store_segmentation(io,idx_vv,softmax_vv):
    res_softmax_v=[]
    for batch,idx_v in enumerate(idx_vv):
        start,end = (0,0)
        res_softmax = []
        softmax_v = softmax_vv[batch]
        for i,idx in enumerate(idx_v):
            voxels = io.blob()['voxels'][idx]
            end    = start + len(voxels)
            softmax = softmax_v[start:end,:]
            start = end
            io.store_segment(idx,softmax)
            res_softmax.append(softmax)
        res_softmax_v.append(res_softmax)
        start = end
    return res_acc_v, res_softmax_v


def round_decimals(val, digits):
    factor = float(np.power(10, digits))
    return int(val * factor+0.5) / factor


def print_memory(msg=''):
    max_allocated = round_decimals(torch.cuda.max_memory_allocated()/1.e9, 3)
    allocated = round_decimals(torch.cuda.memory_allocated()/1.e9, 3)
    max_cached = round_decimals(torch.cuda.max_memory_cached()/1.e9, 3)
    cached = round_decimals(torch.cuda.memory_cached()/1.e9, 3)
    print(max_allocated, allocated, max_cached, cached, msg)


def compute_metrics_sparse(data_v, label_v, softmax_v):
    assert len(data_v) == len(label_v)
    assert len(data_v) == len(softmax_v)
    res = {
        'acc': [],
        'correct_softmax': [],
        'id': [],
        'nonzero_pixels': [],
        'class_acc': [],
        'class_pixel': [],
        'class_mean_softmax': [],
        'cluster_acc': [],
        'class_cluster_acc': [],
    }
    dbscan_vv = []
    for i, label in enumerate(label_v):
        data = data_v[i]
        softmax = softmax_v[i]
        # For each event
        for batch_id in np.unique(data[:, 3]):
            event_index = data[:, 3] == batch_id

            event_data = data[event_index]
            db = DBSCAN(eps=1, min_samples=15).fit(event_data[:, :3]).labels_
            dbscan_vv.append(db[:, None])

            event_softmax = softmax[event_index]
            event_label = label[event_index]
            # Non-zero Accuracy
            predictions = np.argmax(event_softmax, axis=1)[:, None]
            acc = (event_label == predictions).astype(np.int32).sum() / float(len(event_label))
            res['acc'].append(acc)
            # Softmax score of correct labels
            correct_softmax = event_softmax[np.arange(len(event_label)), event_label.reshape((-1,)).astype(np.int32)][:, None]
            res['correct_softmax'].append(np.mean(correct_softmax))
            res['id'].append(batch_id)
            res['nonzero_pixels'].append(event_label.shape[0])

            clusters_index = db > -1
            # print(np.unique(db))
            cluster_acc = (event_label[clusters_index] == predictions[clusters_index]).astype(np.int32).sum() / clusters_index.astype(np.int32).sum()
            res['cluster_acc'].append(cluster_acc)

            classes, class_count = np.unique(event_label, return_counts=True)
            class_pixel = []
            class_acc = []
            class_mean_softmax = []
            class_cluster_acc = []
            for c in range(event_softmax.shape[1]):
                class_index = event_label == c
                class_acc.append((event_label[class_index] == predictions[class_index]).astype(np.int32).sum() / float(len(event_label[class_index])))
                class_cluster_index = event_label[clusters_index] == c
                class_cluster_acc.append((event_label[clusters_index][class_cluster_index] == predictions[clusters_index][class_cluster_index]).astype(np.int32).sum() / float(len(event_label[clusters_index][class_cluster_index])))
                class_mean_softmax.append(np.mean(correct_softmax[class_index]))
                if c in classes:
                    class_pixel.append(class_count[classes == c])
                else:
                    class_pixel.append(0)
            res['class_acc'].append(class_acc)
            res['class_mean_softmax'].append(class_mean_softmax)
            res['class_pixel'].append(np.hstack(class_pixel))
            res['class_cluster_acc'].append(class_cluster_acc)
    return res, dbscan_vv

def compute_metrics_dense(data_v, label_v, softmax_v):
    print(len(data_v), len(label_v), len(softmax_v[0]))
    assert len(data_v) == len(label_v)
    assert len(data_v) == len(softmax_v)
    res = {
        'acc': [],
        'correct_softmax': [],
        'id': [],
        'nonzero_pixels': [],
        'class_acc': [],
        'class_pixel': [],
        'class_mean_softmax': [],
        'cluster_acc': [],
        'class_cluster_acc': [],
    }
    dbscan_vv = []
    for i, label in enumerate(label_v):
        data = data_v[i]
        softmax = softmax_v[i]
        print(data.shape, softmax.shape)
        batch_size = data.shape[0]
        for j in range(batch_size):
            event_data = data[j, ...]
            event_softmax = data[j, ...]
            event_label = label_v[i][j, ...]

            # Non-zero Accuracy
            predictions = np.argmax(event_softmax, axis=1)[:, None]
            acc = (event_label == predictions).astype(np.int32).sum() / float(len(event_label))
            res['acc'].append(acc)
            # Softmax score of correct labels
            # correct_softmax = event_softmax[np.arange(len(event_label)), event_label.reshape((-1,)).astype(np.int32)][:, None]
            # res['correct_softmax'].append(np.mean(correct_softmax))
            res['id'].append(j)
            res['nonzero_pixels'].append(np.count_nonzero(event_label))
    return res
