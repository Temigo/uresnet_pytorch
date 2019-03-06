from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from sklearn.cluster import DBSCAN
import matplotlib
from sklearn.metrics import log_loss
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


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


def store_segmentation(io,idx_vv,softmax_vv):
    res_softmax_v=[]
    for batch,idx_v in enumerate(idx_vv):
        start,end = (0,0)
        res_softmax = []
        softmax_v = softmax_vv[batch]
        for i,idx in enumerate(idx_v):
            voxels = io.blob()['voxels'][idx]
            end    = start + len(voxels)
            softmax = softmax_v[start:end,:]
            start = end
            io.store_segment(idx,softmax)
            res_softmax.append(softmax)
        res_softmax_v.append(res_softmax)
        start = end
    return res_acc_v, res_softmax_v


def round_decimals(val, digits):
    factor = float(np.power(10, digits))
    return int(val * factor+0.5) / factor


def print_memory(msg=''):
    max_allocated = round_decimals(torch.cuda.max_memory_allocated()/1.e9, 3)
    allocated = round_decimals(torch.cuda.memory_allocated()/1.e9, 3)
    max_cached = round_decimals(torch.cuda.max_memory_cached()/1.e9, 3)
    cached = round_decimals(torch.cuda.memory_cached()/1.e9, 3)
    print(max_allocated, allocated, max_cached, cached, msg)


def compute_metrics_sparse(data_v, label_v, softmax_v, idx_v, N=192, particles=None):
    assert len(data_v) == len(label_v)
    assert len(data_v) == len(softmax_v)
    res = {
        'acc': [],
        'correct_softmax': [],
        'id': [],
        'nonzero_pixels': [],
        'class_acc': [],
        'class_pixel': [],
        'class_mean_softmax': [],
        # 'cluster_acc': [],
        # 'class_cluster_acc': [],
        'confusion_matrix': [],
        'energy_confusion_matrix': [],
        'loss_seg': [],
        'misclassified_pixels': [],
        'distances': [],
        # Michel energy analysis
        'michel_num': [],
        'michel_actual_num': [],
        'michel_deposited_energy': [],
        'michel_start_x': [],
        'michel_start_y': [],
        'michel_start_z': [],
        'michel_end_x': [],
        'michel_end_y': [],
        'michel_end_z': [],
        'michel_creation_x': [],
        'michel_creation_y': [],
        'michel_creation_z': [],
        'michel_npx': [],
        'michel_creation_energy': [],
        'michel_creation_momentum': [],
        # For each true michel electron
        'michel_appended': [],
        'michel_num_pix': [],
        'michel_sum_pix': [],
        'michel_sum_pix_pred': [],
        'michel_num_pix_pred': [],
        # For each predicted Michel cluster
        'michel_is_attached': [],  # Whether attached to MIP
        'michel_is_edge': [],  # Whether to the edge of MIP
        'michel_pred_num_pix': [],  # Num pix in predicted cluster
        'michel_pred_sum_pix': [],  # Sum of pix in predicted cluster
        'michel_pred_num_pix_true': [],  # Num pix in predicted cluster inter matched true cluster
        'michel_pred_sum_pix_true': [],  # Sum of pix in predicted cluster inter matched true cluster
        'michel_true_num_pix': [],  # Num of pix in matched true cluster
        'michel_true_sum_pix': [],  # Sum of pix in matched true cluster
        'michel_true_energy': [],
    }
    dbscan_vv = []
    for i, label in enumerate(label_v):
        data = data_v[i]
        softmax = softmax_v[i]
        # For each event
        for batch_id in np.unique(data[:, -2]):
            event_index = data[:, -2] == batch_id

            event_data = data[event_index]  # Shape (N, dim+2)
            event_softmax = softmax[event_index]
            event_label = label[event_index]
            # Non-zero Accuracy
            predictions = np.argmax(event_softmax, axis=1)[:, None]  # Shape (N, 1)
            acc = (event_label == predictions).astype(np.int32).sum() / float(len(event_label))
            res['acc'].append(acc)
            # Loss TODO add weighting
            loss = log_loss(event_label.astype(np.int32), event_softmax, labels=np.arange(event_softmax.shape[1]))
            res['loss_seg'].append(loss)

            # Softmax score of correct labels
            correct_softmax = event_softmax[np.arange(len(event_label)), event_label.reshape((-1,)).astype(np.int32)][:, None]
            res['correct_softmax'].append(np.mean(correct_softmax))
            res['id'].append(batch_id)
            res['nonzero_pixels'].append(event_label.shape[0])

            # Incorrect pixels and their distance to the boundary
            incorrect_pixels = event_data[(predictions != event_label).reshape((-1,)), ...]
            incorrect_pixels_labels = event_label[predictions != event_label][:, None]
            incorrect_pixels_predicted = predictions[predictions != event_label][:, None]
            incorrect_pixels_correct_softmax = correct_softmax[predictions != event_label][:, None]
            incorrect_pixels_predicted_softmax = event_softmax[np.arange(len(event_label)), predictions.reshape((-1,)).astype(np.int32)][:, None][predictions != event_label][:, None]
            res['misclassified_pixels'].append(np.concatenate([incorrect_pixels[:, 0:-2],
                                                               incorrect_pixels_correct_softmax,
                                                               incorrect_pixels_predicted_softmax,
                                                               incorrect_pixels_predicted,
                                                               incorrect_pixels[:, -1][:, None],  # Energy
                                                               incorrect_pixels_labels], axis=1))
            # Nonzero pixels and their distance to the boundary
            min_v = []
            for d in range(event_data.shape[1]-2):
                min_v.append(np.minimum(event_data[:, d], N-event_data[:, d]))
            distances = np.minimum.reduce(min_v)
            res['distances'].append(np.histogram(distances, bins=np.linspace(0, 50, 51))[0])

            # clusters_index = db > -1
            # # print(np.unique(db))
            # cluster_acc = (event_label[clusters_index] == predictions[clusters_index]).astype(np.int32).sum() / clusters_index.astype(np.int32).sum()
            # res['cluster_acc'].append(cluster_acc)

            classes, class_count = np.unique(event_label, return_counts=True)
            class_pixel = []
            class_acc = []
            class_mean_softmax = []
            # class_cluster_acc = []
            num_classes = event_softmax.shape[1]
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
            energy_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
            for c in range(num_classes):
                class_index = event_label == c
                class_acc.append((event_label[class_index] == predictions[class_index]).astype(np.int32).sum() / float(len(event_label[class_index])))
                # class_cluster_index = event_label[clusters_index] == c
                # class_cluster_acc.append((event_label[clusters_index][class_cluster_index] == predictions[clusters_index][class_cluster_index]).astype(np.int32).sum() / float(len(event_label[clusters_index][class_cluster_index])))
                class_mean_softmax.append(np.mean(correct_softmax[class_index]))
                if c in classes:
                    class_pixel.append(class_count[classes == c])
                else:
                    class_pixel.append(0)
                for c2 in range(num_classes):
                    confusion_index = predictions[class_index] == c2
                    confusion_matrix[c][c2] = confusion_index.astype(np.int32).sum()
                    energy_confusion_matrix[c][c2] = event_data[..., -1][..., None][class_index][confusion_index].sum()

            # Michel energy distribution
            if particles is not None:
                p = particles[i][int(batch_id)]  # All particles in current event
                michel_index = p['category'] == 4
                res['michel_num'].append(michel_index.astype(np.int32).sum())
                res['michel_actual_num'].append(np.count_nonzero(p['npx'][michel_index]))
                # res['michel_deposited_energy'].append(p['deposited_energy'][michel_index].sum())
                res['michel_start_x'].append(p['start_x'][michel_index].mean())
                res['michel_start_y'].append(p['start_y'][michel_index].mean())
                res['michel_start_z'].append(p['start_z'][michel_index].mean())
                res['michel_end_x'].append(p['end_x'][michel_index].mean())
                res['michel_end_y'].append(p['end_y'][michel_index].mean())
                res['michel_end_z'].append(p['end_z'][michel_index].mean())
                res['michel_creation_x'].append(p['creation_x'][michel_index].mean())
                res['michel_creation_y'].append(p['creation_y'][michel_index].mean())
                res['michel_creation_z'].append(p['creation_z'][michel_index].mean())
                res['michel_npx'].append(p['npx'][michel_index].sum())
                # res['michel_creation_energy'].append(p['creation_energy'][michel_index].mean())
                res['michel_creation_momentum'].append(p['creation_momentum'][michel_index].mean())
                # if res['michel_actual_num'][-1] == 1 and np.count_nonzero(event_label==4) > 0 and np.count_nonzero(event_label==1) > 0:
                #     MIP_index = event_label == 1
                #     MIP_coords = event_data[MIP_index.reshape((-1,)), ...][:, :-2]
                #     Michel_index = event_label == 4
                #     Michel_coords = event_data[Michel_index.reshape((-1,)), ...][:, :-2]
                #     distances = cdist(MIP_coords, Michel_coords)
                #     MIP_min, Michel_min = np.unravel_index(np.argmin(distances), distances.shape)
                #     is_attached = np.min(distances) < 2.8284271247461903
                #     # is_edge = Michel_coords[np.abs(Michel_coords-Michel_coords[Michel_min]) < 5] >= 5
                #     clusters = DBSCAN(eps=2, min_samples=10).fit(Michel_coords).labels_
                #     cluster_id = clusters[Michel_min]
                #     is_edge = cluster_id > -1 and np.count_nonzero(clusters == cluster_id) > 10
                #     res['michel_appended'].append(is_attached and is_edge)
                # else:
                #     res['michel_appended'].append(False)
                MIP_coords = event_data[(event_label==1).reshape((-1,)), ...][:, :-2]
                Michel_coords = event_data[(event_label==4).reshape((-1,)), ...][:, :-2]
                MIP_coords_pred = event_data[(predictions==1).reshape((-1,)), ...][:, :-2]
                Michel_coords_pred = event_data[(predictions==4).reshape((-1,)), ...][:, :-2]

                Michel_start = np.vstack([p['start_x'][michel_index], p['start_y'][michel_index], p['start_z'][michel_index]]).T

                michel_appended, michel_sum_pix, michel_num_pix = [], [], []
                michel_num_pix_pred, michel_sum_pix_pred = [], []
                michel_deposited_energy, michel_creation_energy = [], []
                if Michel_coords.shape[0] > 0:
                    # MIP_clusters = DBSCAN(eps=1, min_samples=10).fit(MIP_coords).labels_
                    Michel_true_clusters = DBSCAN(eps=2.8284271247461903, min_samples=5).fit(Michel_coords).labels_
                    # MIP_clusters_id = np.unique(MIP_clusters[MIP_clusters>-1])
                    Michel_clusters_id = np.unique(Michel_true_clusters[Michel_true_clusters>-1])
                    for Michel_id in Michel_clusters_id:
                        current_index = Michel_true_clusters == Michel_id
                        distances = cdist(Michel_coords[current_index], MIP_coords)
                        is_attached = np.min(distances) < 2.8284271247461903
                        # Match to MC Michel
                        distances2 = cdist(Michel_coords[current_index], Michel_start)
                        closest_mc = np.argmin(distances2, axis=1)
                        closest_mc_id = closest_mc[np.bincount(closest_mc).argmax()]
                        michel_deposited_energy.append(p['deposited_energy'][michel_index][closest_mc_id])
                        michel_creation_energy.append(p['creation_energy'][michel_index][closest_mc_id])

                        michel_appended.append(is_attached)
                        michel_sum_pix.append(event_data[(event_label==4).reshape((-1,)), ...][current_index][:, -1].sum())
                        michel_num_pix.append(np.count_nonzero(current_index))
                        michel_pred_index = predictions[event_label==4][current_index]==4
                        michel_num_pix_pred.append(np.count_nonzero(michel_pred_index))
                        michel_sum_pix_pred.append(event_data[(event_label==4).reshape((-1,)), ...][current_index][(michel_pred_index).reshape((-1,)), ...][:, -1].sum())
                michel_is_attached, michel_is_edge = [], []
                michel_pred_num_pix, michel_pred_sum_pix = [], []
                michel_pred_num_pix_true, michel_pred_sum_pix_true = [], []
                michel_true_num_pix, michel_true_sum_pix = [], []
                michel_true_energy = []
                if Michel_coords_pred.shape[0] > 0:
                    MIP_clusters = DBSCAN(eps=2.8284271247461903, min_samples=10).fit(MIP_coords_pred).labels_
                    Michel_pred_clusters = DBSCAN(eps=2.8284271247461903, min_samples=5).fit(Michel_coords_pred).labels_
                    Michel_pred_clusters_id = np.unique(Michel_pred_clusters[Michel_pred_clusters>-1])
                    for Michel_id in Michel_pred_clusters_id:
                        current_index = Michel_pred_clusters == Michel_id
                        distances = cdist(Michel_coords_pred[current_index], MIP_coords_pred[MIP_clusters>-1])
                        is_attached = np.min(distances) < 2.8284271247461903
                        is_edge = False  # default
                        if is_attached:
                            Michel_min, MIP_min = np.unravel_index(np.argmin(distances), distances.shape)
                            MIP_id = MIP_clusters[MIP_clusters>-1][MIP_min]
                            MIP_min_coords = MIP_coords_pred[MIP_clusters>-1][MIP_min]
                            MIP_cluster_coords = MIP_coords_pred[MIP_clusters==MIP_id]
                            ablated_cluster = MIP_cluster_coords[np.linalg.norm(MIP_cluster_coords-MIP_min_coords, axis=1)>15.0]
                            if ablated_cluster.shape[0] > 0:
                                new_cluster = DBSCAN(eps=2.8284271247461903, min_samples=5).fit(ablated_cluster).labels_
                                is_edge = len(np.unique(new_cluster[new_cluster>-1])) == 1
                            else:
                                is_edge = True
                        michel_is_attached.append(is_attached)
                        michel_is_edge.append(is_edge)
                        michel_pred_num_pix.append(np.count_nonzero(current_index))
                        michel_pred_sum_pix.append(event_data[(predictions==4).reshape((-1,)), ...][current_index][:, -1].sum())
                        michel_pred_num_pix_true.append(-1)
                        michel_pred_sum_pix_true.append(-1)
                        michel_true_num_pix.append(-1)
                        michel_true_sum_pix.append(-1)
                        michel_true_energy.append(-1)
                        # Match closest true Michel cluster
                        if is_attached and is_edge and Michel_coords.shape[0] > 0:
                            distances = cdist(Michel_coords_pred[current_index], Michel_coords)
                            closest_clusters = Michel_true_clusters[np.argmin(distances, axis=1)]
                            closest_clusters = closest_clusters[closest_clusters > -1]
                            if len(closest_clusters) > 0:
                                closest_true_id = closest_clusters[np.bincount(closest_clusters).argmax()]
                                if closest_true_id > -1:
                                    closest_true_index = event_label[predictions==4][current_index]==4
                                    michel_pred_num_pix_true[-1] = np.count_nonzero(closest_true_index)
                                    michel_pred_sum_pix_true[-1] = event_data[(predictions==4).reshape((-1,)), ...][current_index][(closest_true_index).reshape((-1,)), ...][:, -1].sum()
                                    michel_true_num_pix[-1] = np.count_nonzero(Michel_true_clusters == closest_true_id)
                                    michel_true_sum_pix[-1] = event_data[(event_label==4).reshape((-1,)), ...][Michel_true_clusters == closest_true_id][:, -1].sum()
                                    # Register true energy
                                    # Match to MC Michel
                                    distances2 = cdist(Michel_coords[Michel_true_clusters == closest_true_id], Michel_start)
                                    closest_mc = np.argmin(distances2, axis=1)
                                    closest_mc_id = closest_mc[np.bincount(closest_mc).argmax()]
                                    # closest_mc_id = closest_mc[np.bincount(closest_mc).argmax()]
                                    michel_true_energy[-1] = p['creation_energy'][michel_index][closest_mc_id]
                res['michel_appended'].append(michel_appended)
                res['michel_sum_pix'].append(michel_sum_pix)
                res['michel_num_pix'].append(michel_num_pix)
                res['michel_sum_pix_pred'].append(michel_sum_pix_pred)
                res['michel_num_pix_pred'].append(michel_num_pix_pred)
                res['michel_deposited_energy'].append(michel_deposited_energy)
                res['michel_creation_energy'].append(michel_creation_energy)

                res['michel_is_attached'].append(michel_is_attached)
                res['michel_is_edge'].append(michel_is_edge)
                res['michel_pred_num_pix'].append(michel_pred_num_pix)
                res['michel_pred_sum_pix'].append(michel_pred_sum_pix)
                res['michel_pred_num_pix_true'].append(michel_pred_num_pix_true)
                res['michel_pred_sum_pix_true'].append(michel_pred_sum_pix_true)
                res['michel_true_num_pix'].append(michel_true_num_pix)
                res['michel_true_sum_pix'].append(michel_true_sum_pix)
                res['michel_true_energy'].append(michel_true_energy)

            # Save event displays of softmax scores
            # if event_label.shape[0] > 500 or class_pixel[4] > 20:
            #     directory = 'figures_trash'
            #     idx = idx_v[i][int(batch_id)]
            #     print(event_data.shape, event_label.shape, predictions.shape, event_softmax.shape, idx)
            #     event_data = event_data.astype(np.int32)
            #     # x, y = event_data[:, 0], event_data[:, 1]
            #     # event_data[:, 0] = y
            #     # event_data[:, 1] = 511 - x
            #     data_dense = np.zeros((512, 512))
            #     data_dense[event_data[:, 0], event_data[:, 1]] = event_data[:, -1]
            #     data_dense = np.rot90(data_dense)
            #     label_dense = np.ones((512, 512))*5.0
            #     label_dense[event_data[:, 0], event_data[:, 1]] = event_label.reshape((-1,))
            #     label_dense = np.rot90(label_dense)
            #     predictions_dense = np.ones((512, 512))*5.0
            #     predictions_dense[event_data[:, 0], event_data[:, 1]] = predictions.reshape((-1,))
            #     predictions_dense = np.rot90(predictions_dense)
            #     correct_softmax_dense = np.zeros((512, 512))
            #     correct_softmax_dense[event_data[:, 0], event_data[:, 1]] = correct_softmax.reshape((-1,))
            #     correct_softmax_dense = np.rot90(correct_softmax_dense)
            #     matplotlib.image.imsave('%s/%d_%d_data.png' % (directory, idx, batch_id), data_dense, cmap="cividis", vmax=1.0, dpi=500, origin='lower')
            #     tab10 = matplotlib.cm.get_cmap('tab10', 6)
            #     matplotlib.image.imsave('%s/%d_%d_label.png' % (directory, idx, batch_id), label_dense, cmap=tab10, dpi=500, origin='lower')
            #     matplotlib.image.imsave('%s/%d_%d_predictions.png' % (directory, idx, batch_id), predictions_dense, cmap=tab10, dpi=500, origin='lower')
            #     matplotlib.image.imsave('%s/%d_%d_correct_softmax.png' % (directory, idx, batch_id), correct_softmax_dense, origin='lower', dpi=500, vmin=0.0, vmax=1.0)
            #     for c in range(num_classes):
            #         softmax_dense = np.zeros((512, 512))
            #         softmax_dense[event_data[:, 0], event_data[:, 1]] = event_softmax[:, c]
            #         softmax_dense = np.rot90(softmax_dense)
            #         matplotlib.image.imsave('%s/%d_%d_softmax_%d.png' % (directory, idx, batch_id, c), softmax_dense, dpi=500, origin='lower', vmin=0.0, vmax=1.0)

            res['class_acc'].append(class_acc)
            res['class_mean_softmax'].append(class_mean_softmax)
            res['class_pixel'].append(np.hstack(class_pixel))
            res['confusion_matrix'].append(confusion_matrix)
            res['energy_confusion_matrix'].append(energy_confusion_matrix)
            # res['class_cluster_acc'].append(class_cluster_acc)
    return res, dbscan_vv


def compute_metrics_dense(data_v, label_v, softmax_v, idx_v):
    assert len(data_v) == len(label_v)
    assert len(data_v) == len(softmax_v)
    res = {
        'acc': [],
        'correct_softmax': [],
        'id': [],
        'nonzero_pixels': [],
        'class_acc': [],
        'class_pixel': [],
        'class_mean_softmax': [],
        # 'cluster_acc': [],
        # 'class_cluster_acc': [],
        'confusion_matrix': [],
        'energy_confusion_matrix': [],
        'misclassified_pixels': [],
        'distances': []
    }
    # dbscan_vv = []
    for i, label in enumerate(label_v):
        nonzero_idx = data_v[i] > 0.000001
        event_data = data_v[i]
        event_softmax = softmax_v[i]
        event_label = label
        # Non-zero Accuracy
        predictions = np.argmax(event_softmax, axis=0)[None, ...]
        acc = (event_label == predictions)[nonzero_idx].astype(np.int32).sum() / float(np.sum(nonzero_idx.astype(np.int32)))
        res['acc'].append(acc)
        # Softmax score of correct labels
        reshaped_event_softmax = event_softmax.reshape((event_softmax.shape[0], -1)).transpose()
        correct_softmax = reshaped_event_softmax[np.arange(reshaped_event_softmax.shape[0]), event_label.reshape((-1,)).astype(np.int64)]
        correct_softmax_nonzero = correct_softmax.reshape(event_label.shape)[nonzero_idx]
        res['correct_softmax'].append(np.mean(correct_softmax_nonzero))
        res['id'].append(i)
        res['nonzero_pixels'].append(np.sum(nonzero_idx.astype(np.int32)))


        # Incorrect pixels and their distance to the boundary
        incorrect_pixels_coords = np.vstack(np.where((predictions != event_label) & nonzero_idx)).T
        incorrect_pixels = event_data[(predictions != event_label) & nonzero_idx].reshape((-1, 1))
        incorrect_pixels_labels = event_label[(predictions != event_label) & nonzero_idx].reshape((-1, 1))
        incorrect_pixels_predicted = predictions[(predictions != event_label) & nonzero_idx].reshape((-1, 1))
        incorrect_pixels_correct_softmax = correct_softmax.reshape(event_label.shape)[(predictions != event_label) & nonzero_idx].reshape((-1, 1))
        incorrect_pixels_predicted_softmax = event_softmax.max(axis=0)[None, ...][(predictions != event_label) & nonzero_idx].reshape((-1, 1))
        res['misclassified_pixels'].append(np.concatenate([incorrect_pixels_coords,
                                                           incorrect_pixels_correct_softmax,
                                                           incorrect_pixels_predicted_softmax,
                                                           incorrect_pixels_predicted,
                                                           incorrect_pixels,  # Energy
                                                           incorrect_pixels_labels], axis=1))
        # Nonzero pixels and their distance to the boundary
        min_v = []
        N = event_data.shape[-1]
        coords = np.vstack(np.where(nonzero_idx)).T[:, 1:]
        for d in range(len(event_data.shape)-1):
            min_v.append(np.minimum(coords[:, d], N-coords[:, d]))
        distances = np.minimum.reduce(min_v)
        res['distances'].append(np.histogram(distances, bins=np.linspace(0, 50, 51))[0])

        classes, class_count = np.unique(event_label, return_counts=True)
        class_pixel = []
        class_acc = []
        class_mean_softmax = []
        # class_cluster_acc = []
        num_classes = event_softmax.shape[0]
        # Ignore background = last index
        confusion_matrix = np.zeros((num_classes-1, num_classes-1), dtype=np.int32)
        energy_confusion_matrix = np.zeros((num_classes-1, num_classes-1), dtype=np.float32)
        for c in range(num_classes-1):
            class_index = event_label[nonzero_idx] == c
            class_acc.append((event_label[nonzero_idx][class_index] == predictions[nonzero_idx][class_index]).astype(np.int32).sum() / float(class_index.astype(np.int32).sum()))
            class_mean_softmax.append(np.mean(correct_softmax_nonzero[class_index]))
            if c in classes:
                class_pixel.append(class_count[classes == c])
            else:
                class_pixel.append(0)
            # TO BE TESTED
            for c2 in range(num_classes-1):
                confusion_index = predictions[nonzero_idx][class_index] == c2
                confusion_matrix[c][c2] = confusion_index.astype(np.int32).sum()
                energy_confusion_matrix[c][c2] = event_data[nonzero_idx][class_index][confusion_index].sum()

        # Save event displays of softmax scores
        # if np.sum(nonzero_idx.astype(np.int32)) > 500 or class_pixel[4] > 20:
        #     directory = 'figures_40'
        #     idx = idx_v[i]
        #     print(event_data.shape, event_label.shape, predictions.shape, event_softmax.shape, idx)
        #     matplotlib.image.imsave('%s/%d_%d_data.png' % (directory, idx, i), event_data[0, ...], cmap="cividis", vmax=1.0, dpi=500)
        #     tab10 = matplotlib.cm.get_cmap('tab10', num_classes)
        #     matplotlib.image.imsave('%s/%d_%d_label.png' % (directory, idx, i), event_label[0, ...], cmap=tab10, dpi=500)
        #     save_predictions = np.ones(predictions.shape)*5.0
        #     save_predictions[nonzero_idx] = predictions[nonzero_idx]
        #     matplotlib.image.imsave('%s/%d_%d_predictions.png' % (directory, idx, i), save_predictions[0, ...], cmap=tab10, dpi=500)
        #     matplotlib.image.imsave('%s/%d_%d_correct_softmax.png' % (directory, idx, i), correct_softmax.reshape(event_label.shape)[0, ...], dpi=500, vmin=0.0, vmax=1.0)
        #     for c in range(num_classes-1):
        #         save_softmax = np.zeros(event_softmax[c, ...][None, ...].shape)
        #         save_softmax[nonzero_idx] = event_softmax[c, ...][None, ...][nonzero_idx]
        #         matplotlib.image.imsave('%s/%d_%d_softmax_%d.png' % (directory, idx, i, c), save_softmax[0, ...], dpi=500, vmin=0.0, vmax=1.0)

        res['class_acc'].append(class_acc)
        res['class_pixel'].append(np.hstack(class_pixel))
        res['class_mean_softmax'].append(class_mean_softmax)
        res['confusion_matrix'].append(confusion_matrix)
        res['energy_confusion_matrix'].append(energy_confusion_matrix)
    return res
