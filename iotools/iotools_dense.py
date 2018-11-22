import tempfile
from io_base import io_base

def make_input_larcv_cfg(flags):
    input_filelist = ''
    for i,f in enumerate(flags.INPUT_FILE):
        input_filelist += '"%s"' % f
        if (i+1) < len(flags.INPUT_FILE):
            input_filelist += ','
    weight_proc = ''
    weight_name = ''
    weight_cfg  = ''
    if len(self._flags.DATA_KEYS)>=3:
        weight_cfg = 'weight: { Tensor3DProducer: "%s" }' % self._flags.DATA_KEYS[2]
        weight_proc = ',"BatchFillerTensor3D"'
        weight_name = ',"weight"'
    cfg = '''
MainIO: {
   Verbosity:    2
   EnableFilter: true
   RandomAccess: %d
   InputFiles: [%s]
   ProcessType:  ["BatchFillerTensor3D","BatchFillerTensor3D"%s]
   ProcessName:  ["data","label"%s]
   NumThreads: 4
   NumBatchStorage: 8
   ProcessList: {
       data:  { Tensor3DProducer: "%s" }
      label:  { Tensor3DProducer: "%s" }
     %s
   }
}
'''
    cfg = cfg % (flags.SHUFFLE,input_filelist, weight_proc, weight_name, flags.DATA_KEYS[0], flags.DATA_KEYS[1], weight_cfg)
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

    readonlyname = 'ReadOnlyType: ['
    readonlytype = 'ReadOnlyName: ['
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
    cfg = cfg % (input_filelist,flags.OUTPUT_FILE,readonlytype,readonlyname)
    cfg_file = tempfile.NamedTemporaryFile('w')
    cfg_file.write(cfg)
    cfg_file.flush()
    return cfg_file

class io_larcv_dense(io_base):

    def __init__(self,flags):
        super(io_larcv_dense,self).__init__(flags=flags)
        self._flags   = flags
        self._blob = {}
        self._fout    = None
        self._last_entry = -1
        self._event_keys = []
        self._metas      = []

    def initialize(self):

        self._input_cfg = make_input_larcv_cfg(self._flags)
        cfg = {'filler_name' : 'MainIO',
               'verbosity'   : 0,
               'filler_cfg'  : self._input_cfg.name}
        self._ihandler = larcv_threadio()
        self._ihandler.configure(cfg)
        self._ihandler.start_manager(self._flags.MINIBATCH_SIZE)
        self.next(store_entries=True,store_event_ids=True)
        self._next_counter = 0
        self._num_entries = self._ihandler._proc.pd().io().get_n_entries()
        self._num_channels = self._input_main.fetch_data(self._flags.DATA_KEYS[0]).dim()[-1]
        
        if self._flags.OUTPUT_FILE:
            self._output_cfg = make_output_larcv_cfg(self._flags)
            self._fout = larcv.IOManager(self._output_cfg.name)
            self._fout.initialize()
        
    def stop_threads(self):
        self._ihandler.stop_manager()

    def set_index_start(self,idx):
        self._ihandler.set_next_index(idx)

    def start_threads(self):
        self._ihandler.start_manager(self._flags.MINIBATCH_SIZE)

    def store_segment(self, idx, group):
        from larcv import larcv
        if self._fout is None:
            return
        idx=int(idx)
        if idx >= self.num_entries():
            raise ValueError
        self._fout.read_entry(idx)

        datatype = 'sparse2d'
        meta=None
        to_voxelset = None
        if self._flags.DATA_DIM == 2:
            datatype = 'sparse2d'
            meta = self._fout.get_data(datatype,self._flags.DATA_KEYS[0]).as_vector().front().meta()
            to_voxelset = larcv.as_tensor2d
        elif self._flags.DATA_DIM == 3:
            datatype = 'sparse3d'
            meta = self._fout.get_data(datatype,self._flags.DATA_KEYS[0]).meta()
            to_voxelset = larcv.as_tensor3d
            
        score = np.max(softmax,axis=1).reshape([-1])
        prediction = np.argmax(softmax,axis=1).astype(np.float32).reshape([-1])

        larcv_softmax = self._fout.get_data(datatype,'softmax')
        vs = to_voxelset(voxel,score,meta,-1.)
        larcv_softmax.set(vs,meta)

        larcv_prediction = self._fout.get_data(datatype,'prediction')
        vs = to_voxelset(voxel,prediction,meta,-1.)
        larcv_prediction.set(vs,meta)

        self._fout.save_entry()

    def next(self,buffer_id=-1,release=True):
        
        if self._next_counter:
            self._ihandler.next(store_entries=True,store_event_ids=True)
            
        blob = {}
        for key in self._flags.DATA_KEYS:
            blob[key] = self._ihandler.fetch_data(key)
        idx = self._ihandler.fetch_entries()
        self._next_counter += 1
        return idx, blob

    def finalize(self):
        self._ihandler.reset()
