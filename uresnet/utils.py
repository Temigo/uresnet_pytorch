from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch


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
