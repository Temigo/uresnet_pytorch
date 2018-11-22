import sys
import time

class io_base(object):

    def __init__(self, flags):
        self._batch_size   = flags.BATCH_SIZE
        self._num_entries  = -1
        self._num_channels = -1
        self._flags = flags
        self._blob = {}
        self.tspent_io = 0
        self.tspent_sum_io = 0
        
    def blob(self):
        return self._blob

    def batch_size(self, size=None):
        if size is None: return self._batch_size
        self._batch_size = int(size)

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
    
    def store_segment(self, idx, group):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError
        
class io_base_sparse(io_base):

    def __init__(self, flags):
        super(io_base_sparse, self).__init__(flags)
