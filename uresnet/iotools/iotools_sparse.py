from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import threading
import multiprocessing
import time
from uresnet.iotools.io_base import io_base


def get_particle_info(particle_v):
    from larcv import larcv
    num_particles = particle_v.size()
    part_info = {'particle_idx' : np.arange(0,num_particles),
                 'primary'      : np.zeros(num_particles,np.int8),
                 'pdg_code'     : np.zeros(num_particles,np.int32),
                 'mass'         : np.zeros(num_particles,np.float32),
                 'creation_x'   : np.zeros(num_particles,np.float32),
                 'creation_y'   : np.zeros(num_particles,np.float32),
                 'creation_z'   : np.zeros(num_particles,np.float32),
                 'direction_x'  : np.zeros(num_particles,np.float32),
                 'direction_y'  : np.zeros(num_particles,np.float32),
                 'direction_z'  : np.zeros(num_particles,np.float32),
                 'start_x'      : np.zeros(num_particles,np.float32),
                 'start_y'      : np.zeros(num_particles,np.float32),
                 'start_z'      : np.zeros(num_particles,np.float32),
                 'end_x'        : np.zeros(num_particles,np.float32),
                 'end_y'        : np.zeros(num_particles,np.float32),
                 'end_z'        : np.zeros(num_particles,np.float32),
                 'creation_energy'   : np.zeros(num_particles,np.float32),
                 'creation_momentum' : np.zeros(num_particles,np.float32),
                 'deposited_energy'  : np.zeros(num_particles,np.float32),
                 'npx'               : np.zeros(num_particles,np.int32),
                 'creation_process'  : ['']*num_particles,
                 'category'          : np.zeros(num_particles,np.int8)
                 }

    for idx in range(num_particles):
        particle = particle_v[idx]
        pdg_code = particle.pdg_code()
        mass     = larcv.ParticleMass(pdg_code)
        momentum = np.float32(np.sqrt(np.power(particle.px(),2)+
                                      np.power(particle.py(),2)+
                                      np.power(particle.pz(),2)))

        part_info[ 'primary'     ][idx] = np.int8(particle.track_id() == particle.parent_track_id())
        part_info[ 'pdg_code'    ][idx] = np.int32(pdg_code)
        part_info[ 'mass'        ][idx] = np.float32(mass)
        part_info[ 'creation_x'  ][idx] = np.float32(particle.x())
        part_info[ 'creation_y'  ][idx] = np.float32(particle.y())
        part_info[ 'creation_z'  ][idx] = np.float32(particle.z())
        part_info[ 'direction_x' ][idx] = np.float32(particle.px()/momentum)
        part_info[ 'direction_y' ][idx] = np.float32(particle.py()/momentum)
        part_info[ 'direction_z' ][idx] = np.float32(particle.pz()/momentum)
        part_info[ 'start_x'     ][idx] = np.float32(particle.first_step().x())
        part_info[ 'start_y'     ][idx] = np.float32(particle.first_step().y())
        part_info[ 'start_z'     ][idx] = np.float32(particle.first_step().z())
        part_info[ 'end_x'       ][idx] = np.float32(particle.last_step().x())
        part_info[ 'end_y'       ][idx] = np.float32(particle.last_step().y())
        part_info[ 'end_z'       ][idx] = np.float32(particle.last_step().z())
        part_info[ 'creation_energy'   ][idx] = np.float32(particle.energy_init() - mass)
        part_info[ 'creation_momentum' ][idx] = momentum
        part_info[ 'deposited_energy'  ][idx] = np.float32(particle.energy_deposit())
        part_info[ 'npx'               ][idx] = np.int32(particle.num_voxels())

        category = -1
        process  = particle.creation_process()
        if(pdg_code == 2212 or pdg_code == -2212): category = 0
        elif not pdg_code in [11,-11,22]: category = 1
        elif pdg_code == 22: category = 2
        else:
            if process in ['primary','nCapture','conv','compt']: category = 2
            elif process in ['muIoni','hIoni']: category = 3
            elif process in ['muMinusCaptureAtRest','muPlusCaptureAtRest','Decay']: category = 4
            else:
                print('Unidentified process found: PDG=%d creation_process="%s"' % (pdg_code,process))
                raise ValueError

        part_info[ 'creation_process'  ][idx] = process
        part_info[ 'category'          ][idx] = category
    return part_info


def threadio_func(io_handle, thread_id):
    """
    Structure of returned blob:
        - voxels = [(N, 4)] * batch size
        - feature = [(N, 1)] * batch_per_step
        - data = [(N, 5)] * batch_per_step
        - label = [(N, 1)] * batch size
    where N = total number of points across minibatch_size events
    """
    num_gpus = len(io_handle._flags.GPUS)
    batch_per_step = io_handle.batch_per_step()
    batch_per_gpu = io_handle.batch_per_gpu()
    while 1:
        time.sleep(0.000001)
        while not io_handle._locks[thread_id]:
            idx_v     = []
            voxel_v   = []
            feature_v = []
            new_idx_v = []
            particles_v = []
            # label_v   = []
            blob = {}
            for key, val in io_handle.blob().iteritems():
                blob[key] = []
            if io_handle._flags.SHUFFLE:
                idx_v = np.random.random([batch_per_step])*io_handle.num_entries()
                idx_v = idx_v.astype(np.int32)
                # for key, val in io_handle.blob().iteritems():
                #     blob[key] = val  # fixme start, val?
            else:
                start = io_handle._start_idx[thread_id]
                end   = start + batch_per_step
                if end < io_handle.num_entries():
                    idx_v = np.arange(start,end)
                    # for key, val in io_handle.blob().iteritems():
                    #     blob[key] = val[start:end]
                else:
                    idx_v = np.arange(start, io_handle.num_entries())
                    idx_v = np.concatenate([idx_v,np.arange(0,end-io_handle.num_entries())])
                    # for key, val in io_handle.blob().iteritems():
                    #     blob[key] = val[start:] + val[0:end-io_handle.num_entries()]
                next_start = start + len(io_handle._threads) * batch_per_step
                if next_start >= io_handle.num_entries():
                    next_start -= io_handle.num_entries()
                io_handle._start_idx[thread_id] = next_start

            for i in range(num_gpus):
                voxel_v.append([])
                feature_v.append([])
                new_idx_v.append([])
                particles_v.append([])
                for key in io_handle._flags.DATA_KEYS:
                    blob[key].append([])

            for data_id, idx in enumerate(idx_v):
                voxel  = io_handle.blob()['voxels'][idx]
                new_id = int(data_id / batch_per_gpu)
                voxel_v[new_id].append(np.pad(voxel, [(0,0),(0,1)],'constant',constant_values=data_id))
                feature_v[new_id].append(io_handle.blob()['feature'][idx])
                new_idx_v[new_id].append(idx)
                if 'particles' in io_handle.blob():
                    particles = io_handle.blob()['particles'][idx]
                    particles['batch_id'] = data_id
                    particles_v[new_id].append(particles)
                for key in io_handle._flags.DATA_KEYS:
                    blob[key][new_id].append(io_handle.blob()[key][idx])
                # if len(io_handle._label):
                #     label_v.append(io_handle._label[idx])
            blob['voxels']  = [np.vstack(voxel_v[i]) for i in range(num_gpus)]
            blob['feature'] = [np.vstack(feature_v[i]) for i in range(num_gpus)]
            if len(particles_v) > 0:
                blob['particles'] = particles_v
            new_idx_v = [np.array(x) for x in new_idx_v]
            # if len(label_v): label_v = np.hstack(label_v)
            for key in io_handle._flags.DATA_KEYS:
                blob[key] = [np.vstack(minibatch) for minibatch in blob[key]]
            blob[io_handle._flags.DATA_KEYS[0]] = [np.concatenate([blob['voxels'][i], blob['feature'][i]], axis=1) for i in range(num_gpus)]
            io_handle._buffs[thread_id] = (new_idx_v, blob)
            io_handle._locks[thread_id] = True
    return


def threadio_init_func(io_handle, thread_id):
    br_blob = {}
    data_key = io_handle._flags.DATA_KEYS[0]
    while 1:
        # Fetch current entry number
        entry = io_handle._entry_idx[thread_id]
        if entry >= io_handle.num_entries() or (io_handle._flags.LIMIT_NUM_SAMPLE > 0 and entry == io_handle._flags.LIMIT_NUM_SAMPLE):
            io_handle._is_done[thread_id] = True
            break
        io_handle._entry_idx[thread_id] = entry + len(io_handle._threads_init)

        # Get entry from all TChains
        for key, ch in io_handle._ch_blobs[thread_id].iteritems():
            ch.GetEntry(entry)
            if key not in br_blob:
                if key == 'mcst':
                    br_blob[key] = getattr(ch, 'particle_mcst_branch')
                else:
                    br_blob[key] = getattr(ch, '%s_%s_branch' % (io_handle._dtype_keyword, key))

        io_handle._event_keys[entry] = (br_blob[data_key].run(),
                                        br_blob[data_key].subrun(),
                                        br_blob[data_key].event())

        if io_handle._flags.DATA_DIM == 2:
            num_point = br_blob[data_key].as_vector().front().as_vector().size()
        else:
            num_point = br_blob[data_key].as_vector().size()
        if num_point < 1: return  # FIXME what to do here

        # special treatment for the data
        br_data = br_blob[data_key]
        if io_handle._flags.DATA_DIM == 2:
            br_data = br_data.as_vector().front()
        np_data  = np.zeros(shape=(num_point, io_handle._flags.DATA_DIM+1), dtype=np.float32)
        io_handle._as_numpy_pcloud(br_data, np_data)
        io_handle._blob[data_key][entry] = np_data
        io_handle._total_data[thread_id] += np_data.size

        io_handle._metas[entry] = io_handle._as_meta(br_data.meta())

        np_voxel   = np.zeros(shape=(num_point, io_handle._flags.DATA_DIM), dtype=np.int32)
        io_handle._as_numpy_voxels(br_data, np_voxel)
        io_handle._blob['voxels'][entry] = np_voxel
        io_handle._total_data[thread_id] += np_voxel.size

        np_feature = np.zeros(shape=(num_point, 1), dtype=np.float32)
        io_handle._as_numpy_pcloud(br_data, np_feature)
        io_handle._blob['feature'][entry] = np_feature
        io_handle._total_data[thread_id] += np_feature.size

        # for the rest, different treatment
        for key in io_handle._flags.DATA_KEYS[1:]:
            if io_handle._flags.COMPUTE_WEIGHT and key == io_handle._flags.DATA_KEYS[2]:
                continue
            br = br_blob[key]
            if io_handle._flags.DATA_DIM == 2:
                br = br.as_vector().front()
            np_data = np.zeros(shape=(num_point, 1), dtype=np.float32)
            io_handle._as_numpy_pcloud(br, np_data)
            io_handle._blob[key][entry] = np_data
            io_handle._total_data[thread_id] += np_data.size

        # if weights need to be computed, compute here using label (index 1)
        if io_handle._flags.COMPUTE_WEIGHT:
            labels  = io_handle._blob[io_handle._flags.DATA_KEYS[1]][entry]
            weights = np.zeros(shape=labels.shape, dtype=np.float32)
            classes, counts = np.unique(labels, return_counts=True)
            for c in range(len(classes)):
                idx = np.where(labels == float(c))[0]
                weights[idx] = float(len(labels))/(len(classes))/counts[c]
            io_handle._blob[io_handle._flags.DATA_KEYS[2]][entry] = weights
            io_handle._total_data[thread_id] += weights.size

        if io_handle._flags.PARTICLE:
            particle_v = br_blob['mcst'].as_vector()
            part_info = get_particle_info(particle_v)
            io_handle._blob['particles'][entry] = part_info

        io_handle._num_entries_processed[thread_id] += 1
        io_handle._num_points_processed[thread_id] += num_point
    return


class io_larcv_sparse(io_base):

    def __init__(self, flags):
        super(io_larcv_sparse, self).__init__(flags=flags)
        self._fout    = None
        self._event_keys = []
        self._metas      = []
        # For circular buffer / thread function controls
        self._locks   = [False] * flags.NUM_THREADS
        self._buffs   = [None ] * flags.NUM_THREADS
        self._threads = [None ] * flags.NUM_THREADS
        self._start_idx = [-1 ] * flags.NUM_THREADS
        self._last_buffer_id = -1
        self.set_index_start(0)

        # For initialization threads
        self._threads_init = [None] * flags.NUM_THREADS_INIT
        self._entry_idx = [-1] * flags.NUM_THREADS_INIT
        self._ch_blobs = [None] * flags.NUM_THREADS_INIT
        self._total_data = [0.0] * flags.NUM_THREADS_INIT
        self._num_entries_processed = [0.0] * flags.NUM_THREADS_INIT
        self._num_points_processed = [0.0] * flags.NUM_THREADS_INIT
        self._is_done = [False] * flags.NUM_THREADS_INIT

        # For LArCV functions in 2D/3D
        self._dtype_keyword = ''
        self._as_numpy_voxels = None
        self._as_numpy_pcloud = None
        self._as_meta = None
        self._as_tensor = None

    def init_larcv(self):
        from larcv import larcv
        self._IOManager = larcv.IOManager
        # set 2d vs. 3d functions
        if self._flags.DATA_DIM == 3:
            self._as_numpy_voxels = larcv.fill_3d_voxels
            self._as_numpy_pcloud = larcv.fill_3d_pcloud
            self._dtype_keyword   = 'sparse3d'
            self._as_meta = larcv.Voxel3DMeta
            self._as_tensor = larcv.as_tensor3d
        elif self._flags.DATA_DIM == 2:
            self._as_numpy_voxels = larcv.fill_2d_voxels
            self._as_numpy_pcloud = larcv.fill_2d_pcloud
            self._dtype_keyword   = 'sparse2d'
            self._as_meta = larcv.ImageMeta
            self._as_tensor = larcv.as_tensor2d
        else:
            print('larcv IO not implemented for data dimension', self._flags.DATA_DIM)
            raise NotImplementedError

    def initialize_buffer(self, num_entries):
        # Prepare output buffers
        self._blob['voxels'] = [None] * num_entries
        self._blob['feature'] = [None] * num_entries
        if self._flags.PARTICLE:
            self._blob['particles'] = [None] * num_entries
        for key in self._flags.DATA_KEYS:
            self._blob[key] = [None] * num_entries
        self._event_keys = [None] * num_entries
        self._metas = [None] * num_entries

    def initialize(self):
        # configure the input
        from ROOT import TChain
        self.init_larcv()

        # start threads
        if self._threads_init[0] is not None:
            return
        for thread_id in range(len(self._threads_init)):
            print('Starting thread init', thread_id)
            self._ch_blobs[thread_id] = {}
            for key in self._flags.DATA_KEYS:
                if self._flags.COMPUTE_WEIGHT and key == self._flags.DATA_KEYS[2]:
                    continue
                self._ch_blobs[thread_id][key] = TChain('%s_%s_tree' % (self._dtype_keyword, key))
            if self._flags.PARTICLE:
                self._ch_blobs[thread_id]['mcst'] = TChain('particle_mcst_tree')
            for f in self._flags.INPUT_FILE:
                for ch in self._ch_blobs[thread_id].values():
                    ch.AddFile(f)
            if self._num_entries < 0:
                self._num_entries = self._ch_blobs[0].values()[0].GetEntries()
                if self._flags.LIMIT_NUM_SAMPLE > 0:
                    self._num_entries = self._flags.LIMIT_NUM_SAMPLE
                self.initialize_buffer(self._num_entries)
            self._entry_idx[thread_id] = thread_id
            self._threads_init[thread_id] = threading.Thread(target=threadio_init_func,
                                                             args=[self, thread_id])
            self._threads_init[thread_id].daemon = True
            self._threads_init[thread_id].start()

        if self._num_entries == 0:
            raise Exception("No entries found in data file.")
        event_fraction = 1./self._num_entries * 100.
        if self._flags.LIMIT_NUM_SAMPLE > 0:
            event_fraction = 1./self._flags.LIMIT_NUM_SAMPLE * 100.
        total_point = 0.
        total_data = 0.
        entry = 0
        is_done = False
        while entry < self._num_entries and not is_done:
            entry = 0
            total_point = 0.0
            total_data = 0.0
            time.sleep(0.001)
            is_done = True
            for buffer_id in range(len(self._threads_init)):
                # Process blob
                entry += self._num_entries_processed[thread_id]
                total_point  += self._num_points_processed[thread_id]
                total_data += self._total_data[thread_id]
                is_done = is_done & self._is_done[thread_id]

            sys.stdout.write('Processed %d samples (%d%% ... %d MB\r' % (entry,int(event_fraction*entry),int(total_data*4/1.e6)))
            sys.stdout.flush()
            # print(self._is_done)

        sys.stdout.write('\n')
        sys.stdout.write('Total: %d samples (%d points) ... %d MB\n' % (entry,total_point,total_data*4/1.e6))
        sys.stdout.flush()

        # Prepare output file if necessary
        data = self._blob[self._flags.DATA_KEYS[0]]
        self._num_channels = data[0].shape[-1]
        # Output
        if self._flags.OUTPUT_FILE:
            import tempfile
            cfg = '''
IOManager: {
      Verbosity:   2
      Name:        "IOManager"
      IOMode:      1
      OutFileName: "%s"
      InputFiles:  []
      InputDirs:   []
      StoreOnlyType: []
      StoreOnlyName: []
    }
                  '''
            cfg = cfg % self._flags.OUTPUT_FILE
            cfg_file = tempfile.NamedTemporaryFile('w')
            cfg_file.write(cfg)
            cfg_file.flush()
            self._fout = self._IOManager(cfg_file.name)
            self._fout.initialize()

    def set_index_start(self, idx):
        self.stop_threads()
        for i in range(len(self._threads)):
            self._start_idx[i] = idx + i * self.batch_per_step()

    def start_threads(self):
        if self._threads[0] is not None:
            return
        for thread_id in range(len(self._threads)):
            print('Starting thread', thread_id)
            self._threads[thread_id] = threading.Thread(target=threadio_func,
                                                        args=[self, thread_id])
            self._threads[thread_id].daemon = True
            self._threads[thread_id].start()

    def stop_threads(self):
        if self._threads[0] is None:
            return
        for i in range(len(self._threads)):
            while self._locks[i]:
                time.sleep(0.000001)
            self._buffs[i] = None
            self._start_idx[i] = -1

    def _next(self, buffer_id=-1, release=True):
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
        return res

    def store_segment(self, idx_vv, data_vv, softmax_vv, **kwargs):
        for batch, idx_v in enumerate(idx_vv):
            start, end = (0, 0)
            softmax_v = softmax_vv[batch]
            args_v = [kwargs[keyword][batch] for keyword in kwargs]
            for i, idx in enumerate(idx_v):
                voxels = self.blob()['voxels'][idx]
                end    = start + len(voxels)
                softmax = softmax_v[start:end, :]
                args_event = [arg_v[start:end, :] for arg_v in args_v]
                start = end
                self.store_one_segment(idx, softmax, **dict(zip(kwargs.keys(), args_event)))
            start = end

    def store_one_segment(self, idx, softmax, **kwargs):
        if self._fout is None:
            return
        idx = int(idx)
        if idx >= self.num_entries():
            raise ValueError
        keys = self._event_keys[idx]
        meta = self._metas[idx]
        if self._as_tensor is None:
            self.init_larcv()

        data_key = self._flags.DATA_KEYS[0]

        larcv_data = self._fout.get_data(self._dtype_keyword, data_key)
        voxel   = self._blob['voxels'][idx]
        feature = self._blob['feature'][idx].reshape([-1])
        if self._flags.DATA_DIM == 3:
            vs = self._as_tensor(voxel, feature, meta, 0.)
        elif self._flags.DATA_DIM == 2:
            data = self._blob[self._flags.DATA_KEYS[0]][idx].reshape((-1,))
            vs = self._as_tensor(data, np.arange(data.shape[0]))
        larcv_data.set(vs, meta)

        score = np.max(softmax, axis=1).reshape([-1])
        prediction = np.argmax(softmax, axis=1).astype(np.float32).reshape([-1])

        larcv_softmax = self._fout.get_data(self._dtype_keyword, 'softmax')
        vs = self._as_tensor(voxel, score, meta, -1.)
        larcv_softmax.set(vs, meta)

        larcv_prediction = self._fout.get_data(self._dtype_keyword, 'prediction')
        vs = self._as_tensor(voxel, prediction, meta, -1.)
        larcv_prediction.set(vs, meta)

        for keyword in kwargs:
            values = kwargs[keyword].reshape([-1]).astype(np.float32)
            larcv_arg = self._fout.get_data(self._dtype_keyword, keyword)
            vs = self._as_tensor(voxel, values, meta, -1.)
            larcv_arg.set(vs, meta)

        if len(self._flags.DATA_KEYS) > 1:
            label = self.blob()[self._flags.DATA_KEYS[1]][idx]
            label = label.astype(np.float32).reshape([-1])
            larcv_label = self._fout.get_data(self._dtype_keyword, 'label')
            vs = self._as_tensor(voxel, label, meta, -1.)
            larcv_label.set(vs, meta)
        self._fout.set_id(keys[0], keys[1], keys[2])
        self._fout.save_entry()

    def finalize(self):
        if self._fout:
            self._fout.finalize()
