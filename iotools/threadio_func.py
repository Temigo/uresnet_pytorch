import time
import numpy as np


def threadio_func(io_handle, thread_id):
    """
    Structure of returned blob:
        - voxels = [(N, 4)] * batch size
        - feature = [(N, 1)] * batch_size
        - data = [(N, 5)] * batch_size
        - segment = [(N, 1)] * batch size
    where N = total number of points across minibatch_size events
    """
    while 1:
        time.sleep(0.000001)
        while not io_handle._locks[thread_id]:
            idx_v     = []
            voxel_v   = []
            feature_v = []
            new_idx_v = []
            # label_v   = []
            blob = {}
            for key, val in io_handle._blob.iteritems():
                blob[key] = []
            if io_handle._flags.SHUFFLE:
                idx_v = np.random.random([io_handle.batch_size() * io_handle.minibatch_size()])*io_handle.num_entries()
                idx_v = idx_v.astype(np.int32)
                # for key, val in io_handle._blob.iteritems():
                #     blob[key] = val  # fixme start, val?
            else:
                start = io_handle._start_idx[thread_id]
                end   = start + io_handle.batch_size() * io_handle.minibatch_size()
                if end < io_handle.num_entries():
                    idx_v = np.arange(start,end)
                    # for key, val in io_handle._blob.iteritems():
                    #     blob[key] = val[start:end]
                else:
                    idx_v = np.arange(start, io_handle.num_entries())
                    idx_v = np.concatenate([idx_v,np.arange(0,end-io_handle.num_entries())])
                    # for key, val in io_handle._blob.iteritems():
                    #     blob[key] = val[start:] + val[0:end-io_handle.num_entries()]
                next_start = start + len(io_handle._threads) * io_handle.batch_size() * io_handle.minibatch_size()
                if next_start >= io_handle.num_entries():
                    next_start -= io_handle.num_entries()
                io_handle._start_idx[thread_id] = next_start

            for i in range(io_handle.batch_size()):
                voxel_v.append([])
                feature_v.append([])
                new_idx_v.append([])
                for key in io_handle._flags.DATA_KEYS:
                    blob[key].append([])
            for data_id, idx in enumerate(idx_v):
                voxel   = io_handle._blob['voxels'][idx]
                voxel_v[data_id / io_handle.minibatch_size()].append(np.pad(voxel, [(0,0),(0,1)],'constant',constant_values=data_id))
                feature_v[data_id / io_handle.minibatch_size()].append(io_handle._blob['feature'][idx])
                new_idx_v[data_id / io_handle.minibatch_size()].append(idx)
                for key in io_handle._flags.DATA_KEYS:
                    blob[key][data_id / io_handle.minibatch_size()].append(io_handle._blob[key][idx])
                # if len(io_handle._label):
                #     label_v.append(io_handle._label[idx])
            blob['voxels']   = [np.vstack(voxel_v[i]) for i in range(io_handle.batch_size())]
            blob['feature'] = [np.vstack(feature_v[i]) for i in range(io_handle.batch_size())]
            new_idx_v = [np.array(x) for x in new_idx_v]
            # if len(label_v): label_v = np.hstack(label_v)
            for key in io_handle._flags.DATA_KEYS:
                blob[key] = [np.vstack(minibatch) for minibatch in blob[key]]
            blob[io_handle._flags.DATA_KEYS[0]] = [np.concatenate([blob['voxels'][i], blob['feature'][i]], axis=1) for i in range(io_handle.batch_size())]
            io_handle._buffs[thread_id] = (new_idx_v, blob)
            io_handle._locks[thread_id] = True
    return
