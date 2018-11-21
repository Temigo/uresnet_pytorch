import threading
import time
import sys

class io_base(object):

    def __init__(self, flags):
        self._batch_size   = flags.BATCH_SIZE
        self._num_entries  = -1
        self._num_channels = -1

    def batch_size(self, size=None):
        if size is None: return self._batch_size
        self._batch_size = int(size)

    def num_entries(self):
        return self._num_entries

    def num_channels(self):
        return self._num_channels

    def initialize(self):
        raise NotImplementedError

    def store_segment(self, idx, group):
        raise NotImplementedError

    def store_cluster(self, idx, group):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

        
class io_base_sparse(io_base):

    def __init__(self, flags):
        super(io_base_sparse, self).__init__(flags)
        self._batch_size   = flags.MINIBATCH_SIZE
        self._voxel        = [] # should be a list of numpy arrays
        self._feature      = [] # should be a list of numpy arrays, same length as self._voxel
        self._label        = [] # should be a list of numpy arrays, same length as self._voxel
        # For circular buffer / thread function controls
        self._locks   = [False] * flags.NUM_THREADS
        self._buffs   = [None ] * flags.NUM_THREADS
        self._threads = [None ] * flags.NUM_THREADS
        self._start_idx = [-1 ] * flags.NUM_THREADS
        self._last_buffer_id = -1
        self.set_index_start(0)
        self.tspent_io = 0
        self.tspent_sum_io = 0

    def voxel   (self): return self._voxel
    def feature (self): return self._feature
    def label   (self): return self._label

    def stop_threads(self):
        if self._threads[0] is None:
            return
        for i in range(len(self._threads)):
            while self._locks[buffer_id]:
                time.sleep(0.000001)
            self._buffs[i] = None
            self._start_idx[i] = -1

    def set_index_start(self,idx):
        self.stop_threads()
        for i in range(len(self._threads)):
            self._start_idx[i] = idx + i*self._batch_size

    def start_threads(self):
        if self._threads[0] is not None:
            return
        for thread_id in range(len(self._threads)):
            print('Starting thread',thread_id)
            self._threads[thread_id] = threading.Thread(target = threadio_func, args=[self,thread_id])
            self._threads[thread_id].daemon = True
            self._threads[thread_id].start()

    def next(self,buffer_id=-1,release=True):
        tstart = time.time()
        if buffer_id >= len(self._locks):
            sys.stderr.write('Invalid buffer id requested: {:d}\n'.format(buffer_id))
            raise ValueError
        if buffer_id < 0: buffer_id = self._last_buffer_id + 1
        if buffer_id >= len(self._locks):
            buffer_id = 0
        if self._threads[buffer_id] is None:
            sys.stderr.write('Read-thread does not exist (did you initialize?)\n')
            raise ValueError
        while not self._locks[buffer_id]:
            time.sleep(0.000001)
        res = self._buffs[buffer_id]
        if release:
            self._buffs[buffer_id] = None
            self._locks[buffer_id] = False
            self._last_buffer_id   = buffer_id
        self.tspent_io = time.time() - tstart
        self.tspent_sum_io += self.tspent_io
        return res
