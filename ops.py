import torch
from torch.nn.parallel.scatter_gather import scatter, gather


class SegmentationLoss(torch.nn.modules.loss._Loss):
    def __init__(self, flags, size_average=False):
        super(SegmentationLoss, self).__init__(size_average)
        self._flags = flags
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduce=True)

    def forward(self, segmentation, label, batch_ids, weight):
        """
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has only 1 element because UResNet returns only 1 element.
        label[0] has shape (N, 1) where N is #pts across minibatch_size events.
        """
        # FIXME assumes batch size 1
        # seg = torch.argmax(segmentation, dim=-1)
        total_loss = 0
        total_acc = 0
        total_count = 0
        for i in range(self._flags.BATCH_SIZE):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                event_segmentation = segmentation[i][batch_index]
                event_label = label[i][batch_index]
                if weight is not None:
                    event_weight = weight[i][batch_index]

                event_label = torch.squeeze(event_label, dim=-1).long()
                loss_seg = self.cross_entropy(event_segmentation, event_label)
                if weight is not None:
                    loss_seg *= event_weight
                total_loss += loss_seg
                total_count += 1

                # Accuracy
                predicted_labels = torch.argmax(event_segmentation, dim=-1)
                acc = (predicted_labels == event_label).sum().item() / float(event_label.nelement())
                total_acc += acc
        total_loss = total_loss / total_count
        total_acc = total_acc / total_count

        return total_loss, total_acc

        
class GraphDataParallel(torch.nn.parallel.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, batch_size=1):
        """
        batch_size is per GPU.
        Nothing is guaranteed for batch_size > 1 - it should crash quickly.
        """
        super(GraphDataParallel, self).__init__(module,
                                                device_ids=device_ids,
                                                output_device=output_device,
                                                dim=dim)
        self.batch_size = batch_size

    def scatter(self, inputs, kwargs, device_ids):
        # TODO add a case for dict
        final_inputs = []
        for i, device in enumerate(device_ids):
            input_i = inputs[0][i*self.batch_size:(i+1)*self.batch_size]
            if self.batch_size == 1:
                input_i = [input_i[0][None, ...]]
            final_inputs += scatter(input_i, [device], self.dim) if inputs else []
        final_kwargs = scatter(kwargs, device_ids, self.moduledim) if kwargs else []

        if len(final_inputs) < len(final_kwargs):
            final_inputs.extend([() for _ in range(len(final_kwargs) - len(final_inputs))])
        elif len(final_kwargs) < len(final_inputs):
            final_kwargs.extend([{} for _ in range(len(final_inputs) - len(final_kwargs))])
        final_inputs = tuple(final_inputs)
        final_kwargs = tuple(final_kwargs)
        return final_inputs, final_kwargs

    def gather(self, outputs, output_device):
        """
        len(outputs) = number of gpus
        len(outputs[0]) = number of objects returned by network
        Returns a tuple of length the number of objects returned by network
        Length of tuple[0] = number of gpus
        """
        # TODO add a case for dict
        results = {}
        for output in outputs:  # Iterate over GPUs
            network_outputs = gather([output], output_device, dim=self.dim)
            # results.append(network_outputs)  # FIXME or should we concatenate?
            for i, x in enumerate(network_outputs):
                if i in results:
                    results[i].append(x)
                else:
                    results[i] = [x]
        return tuple(results.values())