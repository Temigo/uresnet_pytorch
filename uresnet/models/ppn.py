from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch


class PPN(torch.nn.Module):
    def __init__(self, flags):
        import sparseconvnet as scn
        super(PPN, self).__init__()
        self._flags = flags
        dimension = flags.DATA_DIM
        reps = 2  # Conv block repetition factor
        kernel_size = 2  # Use input_spatial_size method for other values?
        m = flags.URESNET_FILTERS  # Unet number of features
        nPlanes = [i*m for i in range(1, flags.URESNET_NUM_STRIDES+1)]  # UNet number of features per level
        # nPlanes = [(2**i) * m for i in range(1, num_strides+1)]  # UNet number of features per level
        nInputFeatures = 1

        leakiness = 0
        downsample = [kernel_size, 2]
        def block(m, a, b):
            # ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
             ).add(scn.AddTable())

        self.layers = scn.Sequential()
        for i in range(len(nPlanes)-1):
            module = scn.Sequential()
            for _ in range(reps):
                block(module, nPlanes[i], nPlanes[i])
            module.add(scn.Sequential().add(
                    scn.BatchNormLeakyReLU(nPlanes[i], leakiness=leakiness)).add(
                    scn.Convolution(dimension, nPlanes[i], nPlanes[i+1],
                        downsample[0], downsample[1], False))
                    )
            # self.layers.append(module)
            self.layers.add(module)
        print(len(self.layers), nPlanes)
        # ppn1_cls_prob = torch.nn.Softmax(dim=-1)
        self.ppn1_index, self.ppn2_index = len(self.layers)/2, len(self.layers)-1
        print('PPN1 index = ', self.ppn1_index, 'PPN2 index = ', self.ppn2_index)
        self.input_layer = scn.Sequential().add(
           scn.InputLayer(dimension, flags.SPATIAL_SIZE, mode=3)).add(
           scn.SubmanifoldConvolution(dimension, nInputFeatures, m, 3, False))
        # Sum of nPlanes values
        total_filters = int(m * flags.URESNET_NUM_STRIDES * (flags.URESNET_NUM_STRIDES + 1) / 2)
        self.ppn1_conv = scn.Sequential().add(scn.InputLayer(dimension, flags.SPATIAL_SIZE, mode=3)).add(
            scn.SubmanifoldConvolution(dimension, total_filters, total_filters, 3, False)
        )

        self.ppn1_pixel_pred = scn.SubmanifoldConvolution(dimension, total_filters, dimension, 3, False)
        self.ppn1_scores = scn.SubmanifoldConvolution(dimension, total_filters, 2, 3, False)
        # self.softmax = torch.nn.Softmax(dim=1)

    def expand_feature_map(self, x, coords, i):
        """
        x.features.shape = (N1, N_features)
        x.get_spatial_locations().size() = (N1, 4) (dim + batch_id)
        coords.size() = (N2, 4)
        Returns (N2, N_features)
        """
        # TODO deal with batch id
        # print('expand', i, x.features.shape, x.spatial_size, x.get_spatial_locations().size())
        feature_map = x.get_spatial_locations().cuda().float()
        # shape (N1, N2, 4) for next 2 lines
        feature_map_coords = (feature_map * (2**i))[:, None, :].expand(-1, coords.size(0), -1)
        coords_adapted = coords[None, ...].expand(x.features.shape[0], -1, -1)
        part1 = feature_map_coords <= coords_adapted
        part1 = part1.all(dim=-1)
        part2 = feature_map_coords + 2**i > coords_adapted
        part2 = part2.all(dim=-1)
        index = part1 & part2  # shape (N1, N2)
        # Make sure that all pixels from original belong to 1 only in feature
        # print((index.long().sum(dim=0)!=1).long().sum())
        final = torch.index_select(x.features, 0, torch.t(index).argmax(dim=1))
        # print(final.size())
        return final

    def forward(self, point_cloud):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        shape of point_cloud[0] = (N, 4)
        """
        coords = point_cloud[:, 0:-1].float()
        features = point_cloud[:, -1][:, None].float()
        x = self.input_layer((coords, features))
        feature_maps = [x.features]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            feature_maps.append(self.expand_feature_map(x, coords, i+1))
        final_feature_map = torch.cat(feature_maps, dim=1)
        # print(final_feature_map.size())
        # print(x.features.shape, x.spatial_size, x.get_spatial_locations().size())
        x = self.ppn1_conv((coords, final_feature_map))
        pixel_pred = self.ppn1_pixel_pred(x).features
        scores = self.ppn1_scores(x).features
        # print(pixel_pred.size(), scores.size())
        # print(torch.cat([pixel_pred, scores], dim=1).size())
        return [torch.cat([pixel_pred, scores], dim=1)]


class SegmentationLoss(torch.nn.modules.loss._Loss):
    def __init__(self, flags, reduction='sum'):
        super(SegmentationLoss, self).__init__(reduction=reduction)
        self._flags = flags
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def distances(self, v1, v2):
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))

    def forward(self, segmentation, data, label, weight):
        """
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has only 1 element because UResNet returns only 1 element.
        label[0] has shape (N, 1) where N is #pts across minibatch_size events.
        weight can be None.
        """
        assert len(segmentation) == len(data)
        assert len(data) == len(label)
        if weight is not None:
            assert len(data) == len(weight)
        batch_ids = [d[:, -2] for d in data]
        total_loss = 0
        total_acc = 0
        total_count = 0
        # Loop over ?
        for i in range(len(data)):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                # event_segmentation = segmentation[i][batch_index]
                event_data = data[i][batch_index][:, :-2]  # (N, 3)
                # print('data', event_data.shape, event_data.type())
                event_pixel_pred = segmentation[i][batch_index][:, :-2]  # (N, 3)
                event_scores = segmentation[i][batch_index][:, -2:]  # (N, 2)
                # print('preds', event_pixel_pred.shape, event_scores.shape)
                # Ground truth pixels
                event_label = label[i][label[i][:, -1] == b][:, :-2]  # (N_gt, 3)
                # print('label', event_label.shape)
                # class loss
                # distance loss
                acc = 0.0
                if event_pixel_pred.shape[0] > 0:
                    d = self.distances(event_label, event_pixel_pred)
                    d_true = self.distances(event_label, event_data)
                    positives = (d_true < 5).any(dim=0)
                    loss_seg = self.cross_entropy(event_scores.double(), positives.long())
                    distances_positives = d[:, positives]
                    if distances_positives.shape[1] > 0:
                        d2, _ = torch.min(distances_positives, dim=0)
                        loss_seg += d2.mean()

                    # Accuracy
                    predicted_labels = torch.argmax(event_scores, dim=-1)
                    acc = (predicted_labels == positives.long()).sum().item() / float(predicted_labels.nelement())

                else:
                    print('HELLO')
                    loss_seg = torch.tensor(100.0)
                # loss_seg = event_pixel_pred.mean()

                total_loss += torch.mean(loss_seg)
                total_count += 1
                total_acc += acc

        return total_loss, total_acc
