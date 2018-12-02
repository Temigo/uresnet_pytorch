from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
import numpy as np
from uresnet.iotools.io_base import io_base

def make_input_larcv_cfg(flags):
    input_filelist = 'InputFiles: ['
    for i,f in enumerate(flags.INPUT_FILE):
        input_filelist += '"%s"' % f
        if (i+1) < len(flags.INPUT_FILE):
            input_filelist += ','
    input_filelist += ']'
    proctypes = 'ProcessType: ['
    procnames = 'ProcessName: ['
    proccfg   = ''
    if flags.DATA_DIM==3:
        proctypes += '"EmptyTensorFilter",'
        procnames += '"EmptyTensorFilter",'
        cfg = 'EmptyTensorFilter: { MinVoxel%dDCount: 10 Tensor%dDProducer: "%s" }\n'
        cfg = cfg % (flags.DATA_DIM,flags.DATA_DIM,flags.DATA_KEYS[0])
        proccfg   += cfg 
        for i,key in enumerate(flags.DATA_KEYS):
            proctypes += '"BatchFillerTensor%dD",' % flags.DATA_DIM
            procnames += '"%s",' % key
            if i == 1:
                # special treatment for "label"
                cfg = '        %s: { Tensor%dDProducer: "%s" EmptyVoxelValue: %d }\n'
                cfg = cfg % (key,flags.DATA_DIM,key,flags.NUM_CLASS-1)
                proccfg += cfg 
            else:
                cfg = '        %s: { Tensor%dDProducer: "%s" }\n'
                cfg = cfg % (key,flags.DATA_DIM,key)
                proccfg += cfg
    else:
        for i,key in enumerate(flags.DATA_KEYS):
            proctypes += '"BatchFillerImage2D",'
            procnames += '"%s",' % key
            proccfg += ' %s: { ImageProducer: "%s" }\n' % (key,key)
    proctypes=proctypes[0:proctypes.rfind(',')] + ']'
    procnames=procnames[0:procnames.rfind(',')] + ']'
    random = 0
    if flags.SHUFFLE: random=2

    cfg = '''
MainIO: {
    Verbosity:    2
    EnableFilter: true
    RandomAccess: %d
    %s
    %s
    %s
    NumThreads: 2
    NumBatchStorage: 2
    ProcessList: {
       %s
    }
}
'''
    cfg = cfg % (random,input_filelist, proctypes, procnames, proccfg)
    cfg_file = tempfile.NamedTemporaryFile('w')
    cfg_file.write(cfg)
    cfg_file.flush()
    return cfg_file

def make_output_larcv_cfg(flags):
    # Output
    if not flags.OUTPUT_FILE:
        print('Output file not specified!')
        raise ValueError
    input_filelist = ''
    for i,f in enumerate(flags.INPUT_FILE):
        input_filelist += '"%s"' % f
        if (i+1) < len(flags.INPUT_FILE):
            input_filelist += ','

    readonlyname = 'ReadOnlyName: ['
    readonlytype = 'ReadOnlyType: ['
    for i,key in enumerate(flags.DATA_KEYS):
        readonlyname += '"%s"' % key
        if flags.DATA_DIM == 2:
            readonlytype += '"sparse2d"'
        else:
            readonlytype += '"sparse3d"'
        if (i+1) < len(flags.DATA_KEYS):
            readonlytype += ','
            readonlyname += ','
    readonlytype += ']'
    readonlyname += ']'
    cfg = '''
IOManager: {
      Verbosity:   2
      Name:        "IOManager"
      IOMode:      2
      OutFileName: "%s"
      InputFiles:  [%s]
      %s
      %s
    }
'''
    cfg = cfg % (flags.OUTPUT_FILE,input_filelist,readonlytype,readonlyname)
    cfg_file = tempfile.NamedTemporaryFile('w')
    cfg_file.write(cfg)
    cfg_file.flush()
    return cfg_file

class io_larcv_dense(io_base):

    def __init__(self,flags):
        super(io_larcv_dense,self).__init__(flags=flags)
        self._fout    = None

    def initialize(self):
        from larcv import larcv
        from larcv.dataloader2 import larcv_threadio
        self._input_cfg = make_input_larcv_cfg(self._flags)
        cfg = {'filler_name' : 'MainIO',
               'verbosity'   : 0,
               'filler_cfg'  : self._input_cfg.name}
        self._ihandler = larcv_threadio()
        self._ihandler.configure(cfg)
        self._ihandler.start_manager(self.batch_per_step())
        self._ihandler.next(store_entries=True,store_event_ids=True)
        self._next_counter = 0
        self._num_entries = self._ihandler._proc.pd().io().get_n_entries()
        self._num_channels = self._ihandler.fetch_data(self._flags.DATA_KEYS[0]).dim()[-1]

        if self._flags.OUTPUT_FILE:
            self._output_cfg = make_output_larcv_cfg(self._flags)
            self._fout = larcv.IOManager(self._output_cfg.name)
            self._fout.initialize()

    def stop_threads(self):
        self._ihandler.stop_manager()

    def set_index_start(self,idx):
        self._ihandler.set_next_index(idx)

    def start_threads(self):
        self._ihandler.start_manager(self.batch_per_step())

    def store_segment(self, idx_v, data_v, softmax_v):
        for i,idx in enumerate(idx_v):
            self.store_one_segment(idx,data_v[i],softmax_v[i])

    def store_one_segment(self, idx, data, softmax):
        from larcv import larcv
        if self._fout is None:
            return
        idx=int(idx)
        if idx >= self.num_entries():
            raise ValueError
        self._fout.read_entry(idx)

        datatype = 'sparse2d'
        meta,to_voxelset=(None,None)
        if self._flags.DATA_DIM == 2:
            datatype = 'sparse2d'
            meta = self._fout.get_data(datatype,self._flags.DATA_KEYS[0]).as_vector().front().meta()
            to_voxelset = larcv.as_tensor2d
        elif self._flags.DATA_DIM == 3:
            datatype = 'sparse3d'
            meta = self._fout.get_data(datatype,self._flags.DATA_KEYS[0]).meta()
            to_voxelset = larcv.as_tensor3d

        nonzero = (data > 0).astype(np.float32).squeeze(0)
        score = np.max(softmax,axis=0) * nonzero
        prediction = np.argmax(softmax,axis=0).astype(np.float32) * nonzero

        larcv_softmax = self._fout.get_data(datatype,'softmax')
        vs = to_voxelset(score)
        larcv_softmax.set(vs,meta)

        larcv_prediction = self._fout.get_data(datatype,'prediction')
        vs = to_voxelset(prediction)
        larcv_prediction.set(vs,meta)

        self._fout.save_entry()

    def _next(self,buffer_id=-1,release=True):
        import numpy as np
        if self._next_counter:
            self._ihandler.next(store_entries=True,store_event_ids=True)

        blob = {}
        for key in self._flags.DATA_KEYS:
            data = self._ihandler.fetch_data(key)
            dim  = data.dim()
            data = data.data().reshape(dim)
            if self._flags.DATA_DIM == 3:
                blob[key] = np.array(np.swapaxes(np.swapaxes(np.swapaxes(data,4,3),3,2),2,1))
            else:
                blob[key] = np.array(np.swapaxes(np.swapaxes(data,3,2),2,1))
            #blob[key] = []
            #for gpu in range(len(self._flags.GPUS)):
            #    blob[key].append(data[gpu*self.batch_per_gpu():(gpu+1)*self.batch_per_gpu()])
        idx = np.array(self._ihandler.fetch_entries())
        self._next_counter += 1
        return idx, blob

    def finalize(self):
        self._ihandler.reset()
        if self._fout:
            self._fout.finalize()
