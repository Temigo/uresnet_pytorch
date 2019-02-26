from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import sparseconvnet as scn


class SelectionFeatures(torch.nn.Module):
    def __init__(self, dimension, spatial_size):
        import sparseconvnet as scn
        super(SelectionFeatures, self).__init__()
        self.input_layer = scn.InputLayer(dimension, spatial_size, mode=3)

    def forward(self, input):
        print('selection features', len(input), input[0], input[1])
        return input


class Selection(torch.nn.Module):
    def __init__(self, dimension, spatial_size):

        super(Selection, self).__init__()
        # self.input_layer = scn.InputLayer(dimension, spatial_size, mode=3)
        self.dimension = dimension
        self.spatial_size = spatial_size

    def forward(self, scores):
        import sparseconvnet as scn
        # # FIXME add softmax
        # index = scores.features[:, 0] > 0.5
        # print(index.shape, (scores.features[:, 0] != 0.0).long().sum())
        # # output = scn.InputBatch(self.dimension, self.spatial_size)
        # # return self.input_layer((scores.get_spatial_locations()[index], scores.features[index]))
        # # output.set_locations(scores.get_spatial_locations()[index], scores.features[index], True)
        # # scores.metadata.setInputSpatialLocations(
        # #     scores.features, scores.get_spatial_locations()[index].contiguous(), scores.features[index].contiguous(), True
        # # )
        # print(scores.features[index].shape)
        # # scores.features.index_fill(0, torch.where(index), 0.0)
        # print(index[:20])
        # print(scores.features[index][:, 0][:20])
        # scores.features[index][:, 0].fill_(0.0)
        # scores.features[index][:, 1].fill_(0.0)
        # print(scores.features[index][:, 0][:20])
        # print('selection', scores.features.shape,  (scores.features[:, 0] != 0.0).long().sum())
        # return scores
        output = scn.SparseConvNetTensor()
        i = scores.features
        # print(i)
        index = i[:, 1] > 0.5
        index_values = i[index]
        # m = i.new().resize_(1).expand_as(i[index]).fill_(0.0)
        # print(m.shape, m.sum())
        # m = index_values
        # output.features = m
        # print(m.shape, m.sum())
        # m = i.new_tensor(index_values)
        m = torch.rand_like(index_values).float()
        output.features = i.new().resize_(1)
        print(i.shape, m.shape)
        # output.metadata = scores.metadata
        output.metadata = scn.Metadata(self.dimension)
        output.metadata.setInputSpatialSize(scn.toLongTensor(self.dimension, self.spatial_size))
        # print(scores.get_spatial_locations()[index].contiguous())

        i2 = scores.get_spatial_locations()
        # locations = i2.new_tensor(i2[index])
        locations = torch.randint(high=128, size=i2[index].size()).long()
        print(i2.shape, locations.shape)
        output.metadata.setInputSpatialLocations(output.features, locations.contiguous(), m.contiguous(), True)
        output.spatial_size = scores.spatial_size
        print(output.features)
        return output


class ExtractFeatureMap(torch.nn.Module):
    def __init__(self, i, dimension, spatial_size):
        """
        i such that 2**i * small_spatial_size = big_spatial_size
        spatial size of output
        """
        super(ExtractFeatureMap, self).__init__()
        import sparseconvnet as scn
        self.i = i
        self.input_layer = scn.InputLayer(dimension, spatial_size, mode=3)

    def forward(self, x, y):
        """
        x is feature map with smallest spatial size
        y is output feature map with biggest spatial size
        x.features.shape = (N1, N_features)
        x.get_spatial_locations().size() = (N1, 4) (dim + batch_id)
        coords.size() = (N2, 4) in original image size (bigger spatial size)
        Returns (N2, N_features)
        """
        # FIXME no grad?
        # with torch.no_grad():
        # TODO deal with batch id
        # print('expand', i, x.features.shape, x.spatial_size, x.get_spatial_locations().size())
        feature_map = x.get_spatial_locations().cuda().float()
        coords = y.get_spatial_locations().cuda().float()
        N1 = feature_map.size(0)
        N2 = coords.size(0)

        print('N1 = %d, N2 = %d, 2**i = %d' % (N1, N2, 2**self.i))
        # shape (N1, N2, 4) for next 2 lines
        feature_map_coords = (feature_map * (2**self.i))[:, None, :].expand(-1, N2, -1)
        coords_adapted = coords[None, ...].expand(N1, -1, -1)
        part1 = feature_map_coords <= coords_adapted
        part1 = part1.all(dim=-1)
        part2 = feature_map_coords + 2**self.i > coords_adapted
        part2 = part2.all(dim=-1)
        index = part1 & part2  # shape (N1, N2)
        # Make sure that all pixels from original belong to 1 only in feature
        print((index.long().sum(dim=0)>=1).long().sum())
        print((index.long().sum(dim=0)!=1).long().sum())
        print((index.long().sum(dim=1)!=1).long().sum())
        final_features = torch.index_select(x.features, 0, torch.t(index).argmax(dim=1))
        print(final_features.size())
        final_coords = torch.index_select(feature_map, 0, torch.t(index).argmax(dim=1))
        print(final_coords.size())

        return self.input_layer((final_coords, final_features))
