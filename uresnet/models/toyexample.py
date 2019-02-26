from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import sparseconvnet as scn


class Selection1(torch.nn.Module):
    """
    Making a new SparseConvNetTensor from selection
    """
    def __init__(self, dimension, spatial_size):
        super(Selection1, self).__init__()
        self.dimension = dimension
        self.spatial_size = spatial_size

    def forward(self, scores):
        index = scores.features[:, 1] > 0.5
        new_scores = scn.SparseConvNetTensor()
        new_scores.metadata = scn.Metadata(3)
        new_scores.metadata.setInputSpatialSize(scn.toLongTensor(3, 16))
        new_scores.spatial_size = scores.spatial_size
        new_scores.features = torch.zeros((0, 2), dtype=torch.float)
        new_scores.metadata.setInputSpatialLocations(new_scores.features,
                                                     scores.get_spatial_locations()[index].contiguous(),
                                                     scores.features[index].contiguous(),
                                                     True)
        print('Selected from ', scores.features.shape, 'to', new_scores.features.shape)
        print(scores.features[index], new_scores.features)
        print(scores.get_spatial_locations()[index], new_scores.get_spatial_locations())
        return new_scores


class Selection2(torch.nn.Module):
    """
    Using InputLayer
    """
    def __init__(self, dimension, spatial_size):
        super(Selection2, self).__init__()
        self.dimension = dimension
        self.spatial_size = spatial_size
        self.input_layer = scn.InputLayer(dimension, spatial_size, mode=3)

    def forward(self, scores):
        index = scores.features[:, 1] > 0.5
        return self.input_layer((scores.get_spatial_locations()[index], scores.features[index]))


class Example(torch.nn.Module):
    def __init__(self, selection=None):
        super(Example, self).__init__()
        if selection is None:
            selection = scn.Identity()
        elif selection == 1:
            selection = Selection1(3, 32)
        elif selection == 2:
            selection = Selection2(3, 32)

        self.model = scn.Sequential().add(
            scn.Convolution(3, 1, 2, 2, 2, False)).add(
            selection).add(
            scn.UnPooling(3, 2, 2))

    def forward(self, input):
        return self.model(input)


if __name__=='__main__':
    # Random input data
    input = scn.InputBatch(3, 32)
    locations = torch.randint(low=0, high=31, size=(500, 4), dtype=torch.long)
    features = torch.rand((500, 1), dtype=torch.float)
    input.set_locations(locations, features, False)

    e0 = Example(selection=None)
    e1 = Example(selection=1)
    e2 = Example(selection=2)
    e0.train()
    e1.train()
    e2.train()
    output0 = e0(input)
    output1 = e1(input)
    output2 = e2(input)
    print("No selection : output.features.shape = ", output0.features.shape)
    print("Selection attempt 1 : output.features.shape = ", output1.features.shape)
    print("Selection attempt 2 : output.features.shape = ", output2.features.shape)
    # Output for selection=None is OK (output.features.shape[0] == 500).
    # Output for selection=1 or 2 is empty:
    # SparseConvNetTensor<<features=tensor([], size=(0, 2)),features.shape=torch.Size([0, 2]),
    # batch_locations=tensor([], size=(0, 4), dtype=torch.int64),batch_locations.shape=torch.Size([0, 4]),
    # spatial size=tensor([32, 32, 32])>>
