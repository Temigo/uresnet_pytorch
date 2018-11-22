from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
from iotools.io_base import io_base_sparse


class io_larcv_sparse(io_base_sparse):

    def __init__(self, flags):
        super(io_larcv_sparse, self).__init__(flags=flags)
        self._blob = {}
        self._fout    = None
        self._last_entry = -1
        self._event_keys = []
        self._metas      = []

    def blob(self):
        return self._blob

    def initialize(self):
        self._last_entry = -1
        self._event_keys = []
        self._metas = []
        # configure the input
        from larcv import larcv
        from ROOT import TChain
        # set 2d vs. 3d functions
        dtype_keyword  = ''
        if self._flags.DATA_DIM == 3:
            as_numpy_voxels = larcv.fill_3d_voxels
            as_numpy_pcloud = larcv.fill_3d_pcloud
            dtype_keyword   = 'sparse3d'
        elif self._flags.DATA_DIM == 2:
            as_numpy_voxels = larcv.fill_2d_voxels
            as_numpy_pcloud = larcv.fill_2d_pcloud
            dtype_keyword   = 'sparse2d'
        else:
            print('larcv IO not implemented for data dimension', self._flags.DATA_DIM)
            raise NotImplementedError

        ch_blob = {}
        br_blob = {}
        self._blob['voxels'] = []
        self._blob['feature'] = []
        for key in self._flags.DATA_KEYS:
            ch_blob[key] = TChain('%s_%s_tree' % (dtype_keyword, key))
            self._blob[key] = []
        # ch_data   = TChain('%s_%s_tree' % (dtype_keyword,self._flags.DATA_KEY))
        # ch_label  = None
        # if self._flags.LABEL_KEY:
        #     ch_label  = TChain('%s_%s_tree' % (dtype_keyword,self._flags.LABEL_KEY))
        for f in self._flags.INPUT_FILE:
            for ch in ch_blob.values():
                ch.AddFile(f)

        # self._voxel   = []
        # self._feature = []
        # self._label   = []
        # br_data,br_label=(None,None)
        ach = ch_blob.values()[0]
        event_fraction = 1./ach.GetEntries() * 100.
        total_point = 0.
        for i in range(ach.GetEntries()):
            if self._flags.LIMIT_NUM_SAMPLE > 0 and i == self._flags.LIMIT_NUM_SAMPLE:
                break
            for key, ch in ch_blob.iteritems():
                ch.GetEntry(i)
                if i == 0:
                    br_blob[key] = getattr(ch, '%s_%s_branch' % (dtype_keyword, key))
            num_point = br_blob.values()[0].as_vector().size()
            # if num_point < self._flags.KVALUE: continue

            # ch_data.GetEntry(i)
            # if ch_label:  ch_label.GetEntry(i)
            # if br_data is None:
            #     br_data  = getattr(ch_data, '%s_%s_branch' % (dtype_keyword,self._flags.DATA_KEY))
            #     if ch_label:  br_label  = getattr(ch_label, '%s_%s_branch' % (dtype_keyword,self._flags.LABEL_KEY))

            # HACK that should go away when unifying 2d and 3d data reps...
            if self._flags.DATA_DIM == 2:
                for key in br_blob:
                    br_blob[key] = br_blob[key].as_vector().front()

            # special treatment for the data
            br_data = br_blob[self._flags.DATA_KEYS[0]]
            np_data  = np.zeros(shape=(num_point,4),dtype=np.float32)
            larcv.fill_3d_pcloud(br_data, np_data)
            self._blob[self._flags.DATA_KEYS[0]].append(np_data)

            self._event_keys.append((br_data.run(),br_data.subrun(),br_data.event()))
            self._metas.append(larcv.Voxel3DMeta(br_data.meta()))

            # FIXME HACK that should go away when unifying 2d and 3d data reps...
            # if self._flags.DATA_DIM == 2:
            #     self._metas.append(larcv.ImageMeta(br_label.meta()))
            # else:
            #     self._metas.append(larcv.Voxel3DMeta(br_data.meta()))

            np_voxel   = np.zeros(shape=(num_point,self._flags.DATA_DIM),dtype=np.int32)
            as_numpy_voxels(br_data, np_voxel)
            self._blob['voxels'].append(np_voxel)

            np_feature = np.zeros(shape=(num_point,1),dtype=np.float32)
            as_numpy_pcloud(br_data,  np_feature)
            self._blob['feature'].append(np_feature)

            # for the rest, different treatment
            for key in self._flags.DATA_KEYS[1:]:
                br = br_blob[key]
                np_data = np.zeros(shape=(num_point,1),dtype=np.float32)
                larcv.fill_3d_pcloud(br,np_data)
                self._blob[key].append(np_data)

            total_point += num_point * 4 * (4 + len(self._flags.DATA_KEYS)-1)
            sys.stdout.write('Processed %d%% ... %d MB\r' % (int(event_fraction*i),int(total_point/1.e6)))
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()
        data = self._blob[self._flags.DATA_KEYS[0]]
        self._num_channels = data[0].shape[-1]
        self._num_entries = len(data)
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
            self._fout = larcv.IOManager(cfg_file.name)
            self._fout.initialize()

    def store_segment(self, idx, softmax):
        from larcv import larcv
        if self._fout is None:
            return
        idx=int(idx)
        if idx >= self.num_entries():
            raise ValueError
        keys = self._event_keys[idx]
        meta = self._metas[idx]

        data_key = self._flags.DATA_KEYS[0]

        larcv_data = self._fout.get_data('sparse3d',data_key)
        voxel   = self._blob['voxels'][idx]
        feature = self._blob['feature'][idx].reshape([-1])
        vs = larcv.as_tensor3d(voxel,feature,meta,0.)
        larcv_data.set(vs,meta)

        score = np.max(softmax,axis=1).reshape([-1])
        prediction = np.argmax(softmax,axis=1).astype(np.float32).reshape([-1])

        larcv_softmax = self._fout.get_data('sparse3d','softmax')
        vs = larcv.as_tensor3d(voxel,score,meta,-1.)
        larcv_softmax.set(vs,meta)

        larcv_prediction = self._fout.get_data('sparse3d','prediction')
        vs = larcv.as_tensor3d(voxel,prediction,meta,-1.)
        larcv_prediction.set(vs,meta)

        if len(self._label) > 0:
            label = self._label[idx]
            label = label.astype(np.float32).reshape([-1])
            larcv_label = self._fout.get_data('sparse3d','label')
            vs = larcv.as_tensor3d(voxel,label,meta,-1.)
            larcv_label.set(vs,meta)

        self._fout.set_id(keys[0],keys[1],keys[2])
        self._fout.save_entry()

    def store_cluster(self, idx, groups):
        from larcv import larcv
        if self._fout is None:
            raise NotImplementedError
        idx = int(idx)
        if idx >= self.num_entries():
            raise ValueError
        keys = self._event_keys[idx]
        meta = self._metas[idx]

        larcv_data = self._fout.get_data('sparse3d',self._flags.DATA_KEYS[0])
        # voxel   = self._voxel[idx]
        # feature = self._feature[idx].reshape([-1])
        # vs = larcv.as_tensor3d(voxel,feature,meta,0.)
        # larcv_data.set(vs,meta)
        data = self._blob[self._flags.DATA_KEYS[0]][idx]
        vs = larcv.as_tensor3d(data, meta, 0.)
        larcv_data.set(vs, meta)

        pos = data[:, 0:3]
        if isinstance(groups, list):
            for i, g in enumerate(groups):
                group, name = g
                group = np.concatenate([pos, group], axis=1)

                larcv_group = self._fout.get_data('sparse3d', name)
                vs = larcv.as_tensor3d(group, meta, -1.)
                larcv_group.set(vs, meta)
        else:
            group = np.concatenate([pos, groups], axis=1)

            larcv_group = self._fout.get_data('sparse3d', 'prediction')
            vs = larcv.as_tensor3d(group, meta, -1.)
            larcv_group.set(vs, meta)

        self._fout.set_id(keys[0],keys[1],keys[2])
        self._fout.save_entry()

    def finalize(self):
        if self._fout:
            self._fout.finalize()
