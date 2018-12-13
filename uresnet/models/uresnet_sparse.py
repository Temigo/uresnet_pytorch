from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import sparseconvnet as scn
import time


class UResNet(torch.nn.Module):
    def __init__(self, flags):
        super(UResNet, self).__init__()
        self._flags = flags
        dimension = flags.DATA_DIM
        reps = 2  # Conv block repetition factor
        kernel_size = 2  # Use input_spatial_size method for other values?
        m = flags.URESNET_FILTERS  # Unet number of features
        nPlanes = [i*m for i in range(1, flags.URESNET_NUM_STRIDES+1)]  # UNet number of features per level
        # nPlanes = [(2**i) * m for i in range(1, num_strides+1)]  # UNet number of features per level
        nInputFeatures = 1
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension, flags.SPATIAL_SIZE, mode=3)).add(
           scn.SubmanifoldConvolution(dimension, nInputFeatures, m, 3, False)).add( # Kernel size 3, no bias
           scn.UNet(dimension, reps, nPlanes, residual_blocks=True, downsample=[kernel_size, 2])).add(  # downsample = [filter size, filter stride]
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))
        self.linear = torch.nn.Linear(m, flags.NUM_CLASS)

    def forward(self, point_cloud):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        shape of point_cloud[0] = (N, 4)
        """
        # FIXME assumes (mini)batch size 1
        coords = [p[:, 0:-1].float() for p in point_cloud][0]
        features = [p[:, -1][:, None].float() for p in point_cloud][0]
        #return [self.linear(self.sparseModel((coords,features)))]
        x = self.sparseModel((coords, features))
        x = self.linear(x)
        return x


class SegmentationLoss(torch.nn.modules.loss._Loss):
    def __init__(self, flags, reduction='sum'):
        super(SegmentationLoss, self).__init__(reduction=reduction)
        self._flags = flags
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, segmentation, data, label, weight):
        """
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has only 1 element because UResNet returns only 1 element.
        label[0] has shape (N, 1) where N is #pts across minibatch_size events.
        """
        batch_ids = [d[:, 3] for d in data]
        # FIXME assumes batch size 1
        # seg = torch.argmax(segmentation, dim=-1)
        total_loss = 0
        total_acc = 0
        total_count = 0
        print(segmentation.size())
        for i in range(len(self._flags.GPUS)):
            segmentation_i = segmentation[]
            total_loss += torch.mean(self.cross_entropy(segmentation[i],torch.squeeze(label[i],dim=-1).long()))
            prediction = torch.argmax(segmentation[i],dim=-1)
            total_acc += (prediction == torch.squeeze(label[i],dim=-1).long()).sum().item() / float(prediction.nelement())
            total_count +=1
            continue
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                event_segmentation = segmentation[i][batch_index]
                event_label = label[i][batch_index]
                #if weight is not None:
                #    event_weight = weight[i][batch_index]

                event_label = torch.squeeze(event_label, dim=-1).long()
                loss_seg = self.cross_entropy(event_segmentation, event_label)
                #loss_seg = self.cross_entropy(event_segmentation, torch.squeeze(label[i][batch_index],dim=-1).long())
                if weight is not None:
                    total_loss += torch.mean(loss_seg * weight[i][batch_index])
                else:
                    total_loss += torch.mean(loss_seg)
                total_count += 1

                # Accuracy
                predicted_labels = torch.argmax(event_segmentation, dim=-1)
                acc = (predicted_labels == event_label).sum().item() / float(event_label.nelement())
                total_acc += acc
        total_loss = total_loss / total_count
        total_acc = total_acc / total_count

        return total_loss, total_acc
