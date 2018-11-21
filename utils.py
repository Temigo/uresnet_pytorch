from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn.cluster import DBSCAN
import nu_net
import torch


class CSVData:

    def __init__(self,fout):
        self._fout = fout
        self._str  = None
        self._dict = {}

    def record(self, keys, vals):
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]

    def write(self):

        if self._str is None:
            self._fout=open(self._fout,'w')
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str+='{:f}'
            self._fout.write('\n')
            self._str+='\n'

        self._fout.write(self._str.format(*(self._dict.values())))

    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()


def compute_accuracy(io,idx_vv,label_vv,softmax_vv):
    """
    io: iotool.io_base_sparse inherit class
    idx_v: shape=(B*MB) ... where MB = mini-batch = array of points across events
    label_v: shape=(B,MB,1)
    softmax_v: shape=(B,MB,C)
    """
    acc_vv = []
    for batch, idx_v in enumerate(idx_vv):
        start, end = (0, 0)
        acc_v     = np.zeros(shape=[len(idx_v)], dtype=np.float32)
        label_v   = label_vv[batch]
        softmax_v = softmax_vv[batch]
        for i,idx in enumerate(idx_v):
            voxels  = io.blob()['voxels'][idx]
            end     = start + len(voxels)
            label   = label_v[start:end, :].reshape([-1])
            softmax = softmax_v[start:end, :]
            pred  = np.argmax(softmax, axis=1)
            acc_v[i] = (label == pred).astype(np.int32).sum() / float(len(label))
            start = end
        acc_vv.append(acc_v)
    return np.concatenate(acc_vv, axis=0)


def store_segmentation(io,idx_vv,label_vv,softmax_vv):
    res_acc_v=[]
    res_softmax_v=[]
    for batch,idx_v in enumerate(idx_vv):
        start,end = (0,0)
        res_acc = np.zeros(shape=[len(idx_v)],dtype=np.float32)
        res_softmax = []
        label_v = label_vv[batch]
        softmax_v = softmax_vv[batch]
        for i,idx in enumerate(idx_v):
            voxels = io.blob()['voxels'][idx]
            end    = start + len(voxels)
            label  = label_v[start:end,:].reshape([-1])
            softmax = softmax_v[start:end,:]
            pred   = np.argmax(softmax,axis=1)
            #pred   = np.argmin(softmax,axis=1)
            res_acc[i] = (label == pred).astype(np.int32).sum() / float(len(label))
            start = end
            io.store_segment(idx,softmax)
            res_softmax.append(softmax)
        res_softmax_v.append(res_softmax)
        res_acc_v.append(res_acc)
        start = end
    return res_acc_v, res_softmax_v


def compute_metrics(grp_pred, grp):
    grp_pred = np.squeeze(grp_pred)
    grp = np.squeeze(grp)
    labels = np.unique(grp)
    purity, efficiency, cluster_counts = {}, {}, {}
    for l in labels:
        cluster = np.where(grp == l)[0]
        values, counts = np.unique(grp_pred[cluster], return_counts=True)
        l_pred = values[np.argmax(counts)]
        l = int(l)
        efficiency[l] = np.amax(counts) / float(len(cluster))
        purity[l] = np.amax(counts) / float(len(np.where(grp_pred == l_pred)[0]))
        cluster_counts[l] = len(cluster)

    return purity, efficiency, cluster_counts


def DBScan3(dist_v, scores_v, debug=False):
    assert len(dist_v) == len(scores_v)
    threshold_dist = 0.5
    threshold_score = 0.2
    threshold_cardinal = 20
    threshold_iou = 0.05
    results = []
    for dist, scores in zip(dist_v, scores_v):
        if debug: print(scores.mean(), scores.std())
        # First create proposals based on similitude distance
        proposals = dist < threshold_dist

        # Discard proposals with small confidence score
        keep_index = scores > threshold_score
        proposals = proposals[keep_index, :]
        scores = scores[keep_index]
        if debug: print('score', proposals.shape, scores.shape)

        # Discard proposals with small cardinal
        keep_index2 = np.sum(proposals, axis=1) > threshold_cardinal
        proposals = proposals[keep_index2, :]
        scores = scores[keep_index2]
        if debug: print('cardinal', proposals.shape, scores.shape)
        # IoU NMS
        # Order by score
        index = np.argsort(scores)
        scores = scores[index]
        proposals = proposals[index, :]

        final_groups = []
        while proposals.shape[0] > 1:
            intersection = np.sum(np.logical_and(proposals[0], proposals), axis=-1)
            union = np.sum(np.logical_or(proposals[0], proposals), axis=-1) + 1e-6
            iou = intersection / union
            # print('iou', np.mean(iou))
            groups_to_merge = iou > threshold_iou
            new_group = np.logical_or.reduce(proposals[groups_to_merge])
            final_groups.append(new_group)
            proposals = proposals[iou <= threshold_iou]

        final_groups = np.array(final_groups)
        print('Found final groups:', final_groups.shape)
        if final_groups.shape[0]:
            final_groups = np.argmax(final_groups, axis=0)
        else:  # No group left
            final_groups = np.zeros((dist.shape[1],))
        results.append(final_groups.astype(np.float32))
    return results


def DBScan2(dist_v, threshold):
    """
    Using Scikit Learn DBSCAN implementation
    """
    labels = []
    # dbscan = DBSCAN(eps=3, min_samples=2, metric='precomputed')
    for dist in dist_v:
        db = DBSCAN(eps=threshold, min_samples=5, metric='precomputed').fit(dist)
        print('Clusters: ', np.unique(db.labels_))
        labels.append(db.labels_.astype(np.float32))
    return labels


def DBScan(dist_v, threshold, debug=False):
    """
    Performs clustering with handmade algorithm from Kazu
    """
    res = []
    for dist in dist_v:
        np.fill_diagonal(dist,threshold+1)
        cands   = np.where(np.min(dist,axis=0) < threshold)
        grp     = np.zeros(shape=[dist.shape[0]],dtype=np.float32)
        grp[:]  = -1
        latest_grp_id = 0
        # loop over candidates and group coordinates
        friends_v = [np.where(dist[c] < threshold) for c in cands[0]]
        if debug: print('Found',len(cands[0]),'candidates...')
        for i,cand in enumerate(cands[0]):
            if grp[cand]>=0: continue
            # Is any of friend in an existing group? Then use the closest one
            friends = friends_v[i][0]
            grouped_friends = [friend for friend in friends if grp[friend]>=0]
            if len(grouped_friends)>0:
                best_friend = np.argmin(dist[cand][grouped_friends])
                best_friend = grouped_friends[best_friend]
                grp[cand] = grp[best_friend]
                #print('found grouped friends:',grouped_friends)
                if debug: print('setting from best friend',cand,'(dist',dist[cand][best_friend],') grp',grp[cand])
            else:
                grp[friends] = latest_grp_id
                grp[cand] = latest_grp_id
                if debug: print('setting',cand,latest_grp_id)
                latest_grp_id +=1
                #print('setting friends',friends,latest_grp_id)

        res.append(grp)
    return res


def round_decimals(val, digits):
    factor = float(np.power(10, digits))
    return int(val * factor+0.5) / factor


def print_memory(msg):
    max_allocated = nu_net.utils.round_decimals(torch.cuda.max_memory_allocated()/1.e9, 3)
    allocated = nu_net.utils.round_decimals(torch.cuda.memory_allocated()/1.e9, 3)
    max_cached = nu_net.utils.round_decimals(torch.cuda.max_memory_cached()/1.e9, 3)
    cached = nu_net.utils.round_decimals(torch.cuda.memory_cached()/1.e9, 3)
    print(msg, max_allocated, allocated, max_cached, cached)


if __name__ == '__main__':
    d=CSVData('aho.csv')
    d.record('acc',1.000001)
    d.record('loss',0.1)
    d.write()
