import numpy as np
from uresnet.utils import get_particle_info
from torch.utils.data import Dataset


class SparseDataset(Dataset):
    def __init__(self, flags):
        self._flags = flags
        self._blob = {}
        self._ch_blob = {}
        self._br_blob = {}
        self._num_entries = -1
        self._num_channels = -1

        self._as_numpy_voxels = None
        self._as_numpy_pcloud = None
        self._dtype_keyword = ''
        self._as_meta = None

        self._fout = None

        self.initialize()

    def num_channels(self):
        return self._num_channels

    def init_larcv(self):
        from larcv import larcv
        self._IOManager = larcv.IOManager
        # set 2d vs. 3d functions
        if self._flags.DATA_DIM == 3:
            self._as_numpy_voxels = larcv.fill_3d_voxels
            self._as_numpy_pcloud = larcv.fill_3d_pcloud
            self._dtype_keyword   = 'sparse3d'
            self._as_meta = larcv.Voxel3DMeta
        elif self._flags.DATA_DIM == 2:
            self._as_numpy_voxels = larcv.fill_2d_voxels
            self._as_numpy_pcloud = larcv.fill_2d_pcloud
            self._dtype_keyword   = 'sparse2d'
            self._as_meta = larcv.ImageMeta
        else:
            print('larcv IO not implemented for data dimension', self._flags.DATA_DIM)
            raise NotImplementedError

    def initialize(self):
        self.init_larcv()
        from ROOT import TChain
        # Init ch_blob
        for key in self._flags.DATA_KEYS:
            if self._flags.COMPUTE_WEIGHT and key == self._flags.DATA_KEYS[2]:
                continue
            self._ch_blob[key] = TChain('%s_%s_tree' % (self._dtype_keyword, key))
        if self._flags.PARTICLE:
            self._ch_blob['mcst'] = TChain('particle_mcst_tree')
        # Add input files
        for f in self._flags.INPUT_FILE:
            for ch in self._ch_blob.values():
                ch.AddFile(f)
        # Set num entries
        self._num_entries = self._ch_blob.values()[0].GetEntries()
        if self._num_entries < 1:
            raise Exception('No event in data files')
        if self._flags.LIMIT_NUM_SAMPLE > 0:
            self._num_entries = min(self._num_entries,
                                    self._flags.LIMIT_NUM_SAMPLE)

        # Init self._br_blob
        for key, ch in self._ch_blob.iteritems():
            ch.GetEntry(0)
            if key == 'mcst':
                self._br_blob[key] = getattr(ch, 'particle_mcst_branch')
            else:
                self._br_blob[key] = getattr(ch, '%s_%s_branch' % (self._dtype_keyword, key))

        # Init self._blob
        self._blob['voxels'] = [None] * self._num_entries
        self._blob['feature'] = [None] * self._num_entries
        if self._flags.PARTICLE:
            self._blob['particles'] = [None] * self._num_entries
        for key in self._flags.DATA_KEYS:
            self._blob[key] = [None] * self._num_entries
        # Init other quantities
        self._event_keys = [None] * self._num_entries
        self._metas = [None] * self._num_entries

        # Init num channels
        data = self.__getitem__(0)[1][self._flags.DATA_KEYS[0]]
        self._num_channels = data.shape[-1]

        # Prepare Output File if necessary
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

    def __len__(self):
        return self._num_entries

    def __getitem__(self, index):
        # if self._flags.LIMIT_NUM_SAMPLE > 0 and index >= self._flags.LIMIT_NUM_SAMPLE:
        if self._blob['voxels'][index] is None:
            # Load corresponding data
            for key, ch in self._ch_blob.iteritems():
                ch.GetEntry(index)

            self._event_keys[index] = (self._br_blob[self._flags.DATA_KEYS[0]].run(),
                                       self._br_blob[self._flags.DATA_KEYS[0]].subrun(),
                                       self._br_blob[self._flags.DATA_KEYS[0]].event())

            if self._flags.DATA_DIM == 2:
                num_point = self._br_blob[self._flags.DATA_KEYS[0]].as_vector().front().as_vector().size()
            else:
                num_point = self._br_blob[self._flags.DATA_KEYS[0]].as_vector().size()
            if num_point < 1: return None

            # special treatment for the data
            data_key = self._flags.DATA_KEYS[0]
            br_data = self._br_blob[data_key]
            if self._flags.DATA_DIM == 2:
                br_data = br_data.as_vector().front()
            np_data  = np.zeros(shape=(num_point, self._flags.DATA_DIM+1),dtype=np.float32)
            self._as_numpy_pcloud(br_data, np_data)
            self._blob[data_key][index] = np_data

            self._metas.append(self._as_meta(br_data.meta()))

            # Voxels
            np_voxel   = np.zeros(shape=(num_point,self._flags.DATA_DIM),dtype=np.int32)
            self._as_numpy_voxels(br_data, np_voxel)
            self._blob['voxels'][index] = np_voxel

            # Features (energy depositions)
            np_feature = np.zeros(shape=(num_point,1),dtype=np.float32)
            self._as_numpy_pcloud(br_data,  np_feature)
            self._blob['feature'][index] = np_feature

            # for the rest, different treatment
            for key in self._flags.DATA_KEYS[1:]:
                if self._flags.COMPUTE_WEIGHT and key == self._flags.DATA_KEYS[2]:
                    continue
                br = self._br_blob[key]
                if self._flags.DATA_DIM == 2:
                    br = br.as_vector().front()
                np_data = np.zeros(shape=(num_point,1),dtype=np.float32)
                self._as_numpy_pcloud(br,np_data)
                self._blob[key][index] = np_data

            # if weights need to be computed, compute here using label (index 1)
            if self._flags.COMPUTE_WEIGHT:
                labels  = self._blob[self._flags.DATA_KEYS[1]][index]
                weights = np.zeros(shape=labels.shape,dtype=np.float32)
                classes,counts = np.unique(labels,return_counts=True)
                for c in range(len(classes)):
                    idx = np.where(labels == float(c))[0]
                    weights[idx] = float(len(labels))/(len(classes))/counts[c]
                self._blob[self._flags.DATA_KEYS[2]][index] = weights

            if self._flags.PARTICLE:
                particle_v = self._br_blob['mcst'].as_vector()
                part_info = get_particle_info(particle_v)
                self._blob['particles'][index] = part_info

        # Return already loaded data
        res = {}
        for key in self._blob:
            res[key] = self._blob[key][index]
        return index, res

    def collate_fn(self, batch):
        num_gpus = len(self._flags.GPUS)  # FIXME
        batch_per_gpu =  self._flags.MINIBATCH_SIZE# FIXME
        batch_per_step = self._flags.MINIBATCH_SIZE * len(self._flags.GPUS)

        voxel_v   = []
        feature_v = []
        new_idx_v = []
        particles_v = []
        blob = {}
        for key in batch[0][1]:
            blob[key] = []

        for i in range(num_gpus):
            voxel_v.append([])
            feature_v.append([])
            new_idx_v.append([])
            particles_v.append([])
            for key in self._flags.DATA_KEYS:
                blob[key].append([])

        for data_id in range(batch_per_step):
            blob_i = batch[data_id][1]
            idx = batch[data_id][0]
            voxel  = blob_i['voxels']
            new_id = int(data_id / batch_per_gpu)
            voxel_v[new_id].append(np.pad(voxel, [(0,0),(0,1)],'constant',constant_values=data_id))
            feature_v[new_id].append(blob_i['feature'])
            new_idx_v[new_id].append(idx)
            if 'particles' in blob_i:
                particles = blob_i['particles']
                particles['batch_id'] = data_id
                particles_v[new_id].append(particles)
            for key in self._flags.DATA_KEYS:
                blob[key][new_id].append(blob_i[key])
        blob['voxels']  = [np.vstack(voxel_v[i]) for i in range(num_gpus)]
        blob['feature'] = [np.vstack(feature_v[i]) for i in range(num_gpus)]
        if len(particles_v) > 0:
            blob['particles'] = particles_v
        new_idx_v = [np.array(x) for x in new_idx_v]
        for key in self._flags.DATA_KEYS:
            blob[key] = [np.vstack(minibatch) for minibatch in blob[key]]
        blob[self._flags.DATA_KEYS[0]] = [np.concatenate([blob['voxels'][i], blob['feature'][i]], axis=1) for i in range(num_gpus)]
        blob['idx'] = new_idx_v
        return blob
