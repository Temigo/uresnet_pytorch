import torch


class SegmentationLoss(torch.nn.modules.loss._Loss):
    def __init__(self, flags, size_average=False):
        super(SegmentationLoss, self).__init__(size_average)
        self._flags = flags
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduce=False)

    def forward(self, segmentation, label, weight):
        # FIXME assumes batch size 1
        # seg = torch.argmax(segmentation, dim=-1)
        label = torch.squeeze(label[0], dim=-1).long()
        loss_seg = self.cross_entropy(segmentation, label)
        if weight is not None:
            loss_seg *= weight
        #loss_seg = torch.mean(loss_seg).double()
        loss_seg = torch.mean(loss_seg)

        # Accuracy
        predicted_labels = torch.argmax(segmentation, dim=-1)
        acc = (predicted_labels == label).sum().item() / float(label.nelement())
        return loss_seg, acc