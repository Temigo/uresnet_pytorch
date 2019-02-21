from sparse_dataset import SparseDataset
from torch.utils.data import DataLoader
import numpy as np


class SparseDataLoader(DataLoader):
    def __init__(self, flags):
        self._flags = flags
        self.dataset = SparseDataset(flags)
        super(SparseDataLoader, self).__init__(self.dataset,
                          batch_size = flags.MINIBATCH_SIZE * len(flags.GPUS),
                          shuffle=True,
                          num_workers=flags.NUM_WORKERS,
                          collate_fn=self._collate_fn,
                          drop_last=True)

    def num_entries(self):
        return len(self.dataset)

    def num_channels(self):
        return self.dataset.num_channels()

    def cycle(self):
        while True:
            for x in self:
                yield x

    def _collate_fn(self, batch):
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
