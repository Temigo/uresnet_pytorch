from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch.nn.parallel.scatter_gather import scatter, gather
        
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
