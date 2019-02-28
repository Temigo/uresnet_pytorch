from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from uresnet.models.extract_feature_map import Selection, Multiply


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
        total_filters = int(m * flags.URESNET_NUM_STRIDES * (flags.URESNET_NUM_STRIDES + 1) / 2)

        leakiness = 0
        downsample = [kernel_size, 2]
        print(nPlanes)
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

        # def U(nPlanes): #Recursive function
        #     m = scn.Sequential()
        #     if len(nPlanes) == 1:
        #         for _ in range(reps):
        #             block(m, nPlanes[0], nPlanes[0])
        #     else:
        #         m = scn.Sequential()
        #         for _ in range(reps):
        #             block(m, nPlanes[0], nPlanes[0])
        #         m.add(
        #             scn.ConcatTable().add(
        #                 scn.Identity()).add(
        #                 scn.Sequential().add(
        #                     scn.BatchNormReLU(nPlanes[0])).add(
        #                     scn.Convolution(dimension, nPlanes[0], nPlanes[1],
        #                         downsample[0], downsample[1], False)).add(
        #                     U(nPlanes[1:])).add(
        #                     scn.SubmanifoldConvolution(dimension, nPlanes[1], 2, 1, False)).add(
        #                     Selection(dimension, flags.SPATIAL_SIZE/(2**(len(nPlanes)-1)))).add(
        #                     scn.UnPooling(dimension, downsample[0], downsample[1]))))
        #         m.add(SelectionFeatures(dimension, flags.SPATIAL_SIZE/(2**(len(nPlanes)-1))))
        #         # m.add(scn.JoinTable())
        #     return m

        # self.conv = scn.SubmanifoldConvolution(dimension, nInputFeatures, m, 3, False)
        # self.u = U(nPlanes)
        # print(self.u)
        self.layers = scn.Sequential()#.add(scn.SubmanifoldConvolution(dimension, nInputFeatures, m, 3, False))
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
        # self.layers.add(scn.SubmanifoldConvolution(dimension, nPlanes[-1], nPlanes[-1], 3, False))
        # self.layers.add(scn.SubmanifoldConvolution(dimension, nPlanes[-1], 2, 1, False))
        # self.layers.add(Selection(dimension, flags.SPATIAL_SIZE/(2**(flags.URESNET_NUM_STRIDES-1))))
        # self.layers.add(scn.UnPooling(dimension, downsample[0], downsample[1]))

        self.input_layer = scn.Sequential().add(
           scn.InputLayer(dimension, flags.SPATIAL_SIZE, mode=3)).add(
           scn.SubmanifoldConvolution(dimension, nInputFeatures, m, 3, False))
        # Sum of nPlanes values
        self.half_stride = int(flags.URESNET_NUM_STRIDES/2)
        print(self.half_stride, flags.SPATIAL_SIZE/(2**flags.URESNET_NUM_STRIDES), flags.SPATIAL_SIZE/(2**self.half_stride))
        total_filters = int(m * flags.URESNET_NUM_STRIDES * (flags.URESNET_NUM_STRIDES + 1) / 2)
        self.ppn1_conv = scn.SubmanifoldConvolution(dimension, nPlanes[-1], nPlanes[-1], 3, False)
        # self.ppn1_conv.add(scn.UnPooling(dimension, downsample[0], downsample[1]))
        # self.ppn1_pixel_pred = scn.Sequential().add(scn.SubmanifoldConvolution(dimension, nPlanes[-1], dimension, 1, False))
        self.ppn1_scores = scn.SubmanifoldConvolution(dimension, nPlanes[-1], 2, 1, False)
        # self.ppn1_unpool = scn.Sequential()#.add(scn.InputLayer(dimension, flags.SPATIAL_SIZE/(2**(flags.URESNET_NUM_STRIDES-1)), mode=3)).add(scn.SubmanifoldConvolution(dimension, nPlanes[-1], nPlanes[-1], 1, False))
        self.selection1 = Selection()
        self.selection2 = Selection()
        self.unpool1 = scn.Sequential()
        for i in range(flags.URESNET_NUM_STRIDES-self.half_stride-1):
            self.unpool1.add(scn.UnPooling(dimension, downsample[0], downsample[1]))

        self.unpool2 = scn.Sequential()
        for i in range(self.half_stride):
            self.unpool2.add(scn.UnPooling(dimension, downsample[0], downsample[1]))

        # for i in range(flags.URESNET_NUM_STRIDES-half_stride-2):
        #     self.ppn1_unpool.add(scn.UnPooling(dimension, downsample[0], downsample[1]))
        # print(self.ppn1_unpool)
        # self.extract = ExtractFeatureMap(self._flags.URESNET_NUM_STRIDES-self.half_stride-1, dimension, flags.SPATIAL_SIZE/(2**self.half_stride))
        middle_filters = int(m * self.half_stride * (self.half_stride + 1) / 2)
        print('middle filters', middle_filters)
        self.ppn2_conv = scn.SubmanifoldConvolution(dimension, middle_filters, middle_filters, 3, False)
        self.ppn2_pixel_pred = scn.SubmanifoldConvolution(dimension, middle_filters, dimension, 1, False)
        self.ppn2_scores = scn.SubmanifoldConvolution(dimension, middle_filters, 2, 1, False)
        # self.ppn1_pixel_pred = torch.nn.Linear(total_filters, dimension)
        # self.ppn1_scores = torch.nn.Linear(total_filters, 2)
        # self.new_tensor = scn.InputBatch(self._flags.DATA_DIM,  flags.SPATIAL_SIZE/(2**(flags.URESNET_NUM_STRIDES-1)))
        # self.input_tensor = scn.InputBatch(self._flags.DATA_DIM, self._flags.SPATIAL_SIZE)
        self.multiply1 = Multiply()
        self.multiply2 = Multiply()

        self.ppn3_conv = scn.SubmanifoldConvolution(dimension, nPlanes[0], nPlanes[0], 3, False)
        self.ppn3_pixel_pred = scn.SubmanifoldConvolution(dimension, nPlanes[0], dimension, 1, False)
        self.ppn3_scores = scn.SubmanifoldConvolution(dimension, nPlanes[0], 2, 1, False)

    def forward(self, point_cloud):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        shape of point_cloud[0] = (N, 4)
        """
        # import sparseconvnet as scn
        coords = point_cloud[:, 0:-1].float()
        features = point_cloud[:, -1][:, None].float()
        x = self.input_layer((coords, features))
        feature_maps = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            feature_maps.append(x)

        # PPN1 conv and 1x1 predictions
        x = self.ppn1_conv(x)
        ppn1_scores = self.ppn1_scores(x)
        mask = self.selection1(ppn1_scores)
        attention = self.unpool1(mask)

        # PPN2 conv and 1x1 predictions
        y = feature_maps[self.half_stride]
        y = self.multiply1(y, attention)
        y = self.ppn2_conv(y)
        # ppn2_pixel_pred = self.ppn2_pixel_pred(y)
        ppn2_scores = self.ppn2_scores(y)
        mask2 = self.selection2(ppn2_scores)
        attention2 = self.unpool2(mask2)

        z = feature_maps[0]
        z = self.multiply2(z, attention2)
        # print('after multiply', z)
        z = self.ppn3_conv(z)
        ppn3_pixel_pred = self.ppn3_pixel_pred(z)
        ppn3_scores = self.ppn3_scores(z)

        # Select among PPN2 predictions
        # print(ppn2_pixel_pred.spatial_size, 2**self.half_stride)
        # positions = ppn2_pixel_pred.get_spatial_locations().cuda().float()
        # pixel_pred = (ppn2_pixel_pred.features + positions[:, :-1]) * (2**self.half_stride)
        pixel_pred = ppn3_pixel_pred.features
        scores = ppn3_scores.features
        # print(pixel_pred.shape, scores.shape)
        # Add batch id
        # return [torch.cat([pixel_pred, positions[:, -1][:, None], scores], dim=1)]
        return [torch.cat([pixel_pred, scores], dim=1),
                torch.cat([ppn1_scores.get_spatial_locations().cuda().float(), ppn1_scores.features], dim=1),
                torch.cat([ppn2_scores.get_spatial_locations().cuda().float(), ppn2_scores.features], dim=1)]


class PPN_FCN(torch.nn.Module):
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
        total_filters = int(m * flags.URESNET_NUM_STRIDES * (flags.URESNET_NUM_STRIDES + 1) / 2)

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, flags.SPATIAL_SIZE, mode=3)).add(
            scn.SubmanifoldConvolution(3, nInputFeatures, m, 3, False)).add(
            scn.FullyConvolutionalNet(dimension, reps, nPlanes, residual_blocks=True, downsample=[kernel_size, 2])).add(
            scn.OutputLayer(dimension))
        self.ppn1_pixel_pred = torch.nn.Linear(total_filters, dimension)
        self.ppn1_scores = torch.nn.Linear(total_filters, 2)

    def forward(self, point_cloud):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        shape of point_cloud[0] = (N, 4)
        """
        coords = point_cloud[:, 0:-1].float()
        features = point_cloud[:, -1][:, None].float()
        x = self.sparseModel((coords, features))
        pixel_pred = self.ppn1_pixel_pred(x)
        scores = self.ppn1_scores(x)
        return [torch.cat([pixel_pred, scores], dim=1)]


class PPN_old(torch.nn.Module):
    def __init__(self, flags):
        import sparseconvnet as scn
        super(PPN_old, self).__init__()
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

        self.ppn1_pixel_pred = scn.SubmanifoldConvolution(dimension, total_filters, dimension, 1, False)
        self.ppn1_scores = scn.SubmanifoldConvolution(dimension, total_filters, 2, 1, False)
        # self.softmax = torch.nn.Softmax(dim=1)

    def expand_feature_map(self, x, coords, i):
        """
        x.features.shape = (N1, N_features)
        x.get_spatial_locations().size() = (N1, 4) (dim + batch_id)
        coords.size() = (N2, 4) in original image size (bigger spatial size)
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


class SegmentationLoss0(torch.nn.modules.loss._Loss):
    def __init__(self, flags, reduction='sum'):
        super(SegmentationLoss, self).__init__(reduction=reduction)
        self._flags = flags
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.softmax = torch.nn.Softmax(dim=1)

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
        total_loss = 0.
        total_acc = 0.
        total_distance, total_class = 0., 0.
        # Loop over ?
        for i in range(len(data)):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                # event_segmentation = segmentation[i][batch_index]
                event_data = data[i][batch_ids[i] == b][:, :-2]  # (N, 3)
                # print('data', event_data.shape, event_data.type())
                print(segmentation[i][:, -1].type(), b.type())
                event_pixel_pred = segmentation[i][segmentation[i][:, -1] == b.float()][:, :-3] # (N, 3)
                event_scores = segmentation[i][segmentation[i][:, -1] == b.float()][:, -2:]  # (N, 2)
                # event_scores_softmax = self.softmax(event_scores)
                # print('preds', event_pixel_pred.shape, event_scores.shape)
                # Ground truth pixels
                event_label = label[i][label[i][:, -1] == b][:, :-2]  # (N_gt, 3)
                # print('label', event_label)
                # print('pred', event_pixel_pred[event_scores_softmax[:, 1]>0.9])
                # print('scores', event_scores_softmax[event_scores_softmax[:, 1]>0.9])
                # print('label', event_label.shape)
                # class loss
                # distance loss
                acc = 0.0
                if event_pixel_pred.shape[0] > 0:
                    d = self.distances(event_label, event_pixel_pred)
                    d_true = self.distances(event_label, event_data)
                    positives = (d_true < 5).any(dim=0)
                    loss_seg = torch.mean(self.cross_entropy(event_scores.double(), positives.long()))
                    total_class += loss_seg
                    distances_positives = d[:, positives]
                    if distances_positives.shape[1] > 0:
                        d2, _ = torch.min(distances_positives, dim=0)
                        loss_seg += d2.mean()
                        total_distance += d2.mean()

                    # Accuracy
                    predicted_labels = torch.argmax(event_scores, dim=-1)
                    acc = (predicted_labels == positives.long()).sum().item() / float(predicted_labels.nelement())

                else:
                    print('HELLO')
                    loss_seg = torch.tensor(100.0)
                # loss_seg = event_pixel_pred.mean()

                total_loss += loss_seg
                total_acc += acc

        # return total_loss, total_acc
        return {
            'accuracy': total_acc,
            'loss_seg': total_loss,
            'loss_class': total_class,
            'loss_distance': total_distance
        }



class SegmentationLoss(torch.nn.modules.loss._Loss):
    def __init__(self, flags, reduction='sum'):
        super(SegmentationLoss, self).__init__(reduction=reduction)
        self._flags = flags
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.softmax = torch.nn.Softmax(dim=1)

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
        assert len(segmentation[0]) == len(data)
        assert len(data) == len(label)
        if weight is not None:
            assert len(data) == len(weight)
        batch_ids = [d[:, -2] for d in data]
        total_loss = 0.
        total_acc = 0.
        total_distance, total_class = 0., 0.
        total_loss_ppn1, total_loss_ppn2 = 0., 0.
        # Loop over ?
        for i in range(len(data)):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                # event_segmentation = segmentation[i][batch_index]
                event_data = data[i][batch_index][:, :-2]  # (N, 3)
                event_ppn1_data = segmentation[1][i][segmentation[1][i][:, -3] == b.float()][:, :-3]
                event_ppn2_data = segmentation[2][i][segmentation[2][i][:, -3] == b.float()][:, :-3]
                anchors = (event_data + 0.5).float()
                # print('data', event_data.shape, event_data.type())
                event_pixel_pred = segmentation[0][i][batch_index][:, :-2] + anchors # (N, 3)
                event_scores = segmentation[0][i][batch_index][:, -2:]  # (N, 2)
                event_ppn1_scores = segmentation[1][i][segmentation[1][i][:, -3] == b.float()][:, -2:]
                event_ppn2_scores = segmentation[2][i][segmentation[2][i][:, -3] == b.float()][:, -2:]
                # event_scores_softmax = self.softmax(event_scores)
                # print('preds', event_pixel_pred.shape, event_scores.shape)
                # Ground truth pixels
                event_label = label[i][label[i][:, -1] == b][:, :-2]  # (N_gt, 3)
                # print('label', event_label)
                # print('pred', event_pixel_pred[event_scores_softmax[:, 1]>0.9])
                # print('scores', event_scores_softmax[event_scores_softmax[:, 1]>0.9])
                # print('label', event_label.shape)
                # class loss
                # distance loss
                acc = 0.0
                d = self.distances(event_label, event_pixel_pred)
                d_true = self.distances(event_label, event_data)
                positives = (d_true < 5).any(dim=0)
                loss_seg = torch.mean(self.cross_entropy(event_scores.double(), positives.long()))
                total_class += loss_seg
                distances_positives = d[:, positives]
                if distances_positives.shape[1] > 0:
                    d2, _ = torch.min(distances_positives, dim=0)
                    loss_seg += d2.mean()
                    total_distance += d2.mean()

                # Accuracy
                predicted_labels = torch.argmax(event_scores, dim=-1)
                acc = (predicted_labels == positives.long()).sum().item() / float(predicted_labels.nelement())

                # Loss ppn1
                d_true_ppn1 = self.distances(event_label/(2**(self._flags.URESNET_NUM_STRIDES-1)), event_ppn1_data)
                d_true_ppn2 = self.distances(event_label/(2**(int(self._flags.URESNET_NUM_STRIDES/2))), event_ppn2_data)
                positives_ppn1 = (d_true_ppn1 < 1).any(dim=0)
                positives_ppn2 = (d_true_ppn2 < 1).any(dim=0)
                loss_seg_ppn1 = torch.mean(self.cross_entropy(event_ppn1_scores.double(), positives_ppn1.long()))
                loss_seg_ppn2 = torch.mean(self.cross_entropy(event_ppn2_scores.double(), positives_ppn2.long()))

                # loss_seg = event_pixel_pred.mean()
                total_loss += loss_seg + loss_seg_ppn1 + loss_seg_ppn2
                total_acc += acc
                total_loss_ppn1 += loss_seg_ppn1
                total_loss_ppn2 += loss_seg_ppn2

        # return total_loss, total_acc
        return {
            'accuracy': total_acc,
            'loss_seg': total_loss,
            'loss_class': total_class,
            'loss_distance': total_distance,
            'loss_ppn1': total_loss_ppn1,
            'loss_ppn2': total_loss_ppn2
        }
