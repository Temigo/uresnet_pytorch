from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import time

class io_base(object):

    def __init__(self, flags):

        if not flags.BATCH_SIZE % (flags.MINIBATCH_SIZE * len(flags.GPUS)) == 0:
            msg = 'BATCH_SIZE (%d) must be divisible by GPU count (%d) times MINIBATCH_SIZE(%d)'
            msg = msg % (flags.BATCH_SIZE, len(flags.GPUS), flags.MINIBATCH_SIZE)
            print(msg)
            raise ValueError
        
        self._minibatch_per_step = flags.MINIBATCH_SIZE * len(flags.GPUS)
        self._minibatch_per_gpu  = flags.MINIBATCH_SIZE
        self._num_entries  = -1
        self._num_channels = -1
        self._flags = flags
        self._blob = {}
        self.tspent_io = 0
        self.tspent_sum_io = 0
        
    def blob(self):
        return self._blob

    def batch_per_step(self):
        return self._minibatch_per_step

    def batch_per_gpu(self):
        return self._minibatch_per_gpu

    def num_entries(self):
        return self._num_entries

    def num_channels(self):
        return self._num_channels

    def initialize(self):
        raise NotImplementedError

    def set_index_start(self,idx):
        raise NotImplementedError

    def start_threads(self):
        raise NotImplementedError

    def stop_threads(self):
        raise NotImplementedError

    def next(self,buffer_id=-1,release=True):
        tstart = time.time()
        res = self._next(buffer_id,release)
        self.tspent_io = time.time() - tstart
        self.tspent_sum_io += self.tspent_io
        return res
    
    def _next(self,buffer_id=-1,release=True):
        raise NotImplementedError
    
    def store_segment(self, idx, data, softmax):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError
