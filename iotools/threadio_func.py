import time
import numpy as np


def threadio_func(io_handle, thread_id):
    while 1:
        time.sleep(0.000001)
        while not io_handle._locks[thread_id]:
            idx_v     = []
            voxel_v   = []
            feature_v = []
            # label_v   = []
            blob = {}
            for key, val in io_handle._blob.iteritems():
                blob[key] = []
            if io_handle._flags.SHUFFLE:
                idx_v = np.random.random([io_handle.batch_size()])*io_handle.num_entries()
                idx_v = idx_v.astype(np.int32)
                # for key, val in io_handle._blob.iteritems():
                #     blob[key] = val  # fixme start, val?
            else:
                start = io_handle._start_idx[thread_id]
                end   = start + io_handle.batch_size()
                if end < io_handle.num_entries():
                    idx_v = np.arange(start,end)
                    # for key, val in io_handle._blob.iteritems():
                    #     blob[key] = val[start:end]
                else:
                    idx_v = np.arange(start, io_handle.num_entries())
                    idx_v = np.concatenate([idx_v,np.arange(0,end-io_handle.num_entries())])
                    # for key, val in io_handle._blob.iteritems():
                    #     blob[key] = val[start:] + val[0:end-io_handle.num_entries()]
                next_start = start + len(io_handle._threads) * io_handle.batch_size()
                if next_start >= io_handle.num_entries():
                    next_start -= io_handle.num_entries()
                io_handle._start_idx[thread_id] = next_start

            for data_id, idx in enumerate(idx_v):
                voxel   = io_handle._blob['voxels'][idx]
                voxel_v.append(np.pad(voxel, [(0,0),(0,1)],'constant',constant_values=data_id))
                feature_v.append(io_handle._blob['feature'][idx])
                for key in io_handle._flags.DATA_KEYS:
                    blob[key].append(io_handle._blob[key][idx])
                # if len(io_handle._label):
                #     label_v.append(io_handle._label[idx])
            blob['voxels']   = np.vstack(voxel_v)
            blob['feature'] = np.vstack(feature_v)
            # if len(label_v): label_v = np.hstack(label_v)
            for key in io_handle._flags.DATA_KEYS:
                blob[key] = np.vstack(blob[key])
            blob[io_handle._flags.DATA_KEYS[0]] = np.concatenate([blob['voxels'], blob['feature']], axis=1)
            for key in blob:
                blob[key] = [blob[key]]
            io_handle._buffs[thread_id] = (idx_v, blob)
            io_handle._locks[thread_id] = True
    return
