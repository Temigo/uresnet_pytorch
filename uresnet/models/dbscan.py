from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch


class DBScan(torch.nn.Module):
    def __init__(self, epsilon=0.5, minPoints=10):
        super(DBScan, self).__init__()
        self.epsilon = epsilon
        self.minPoints = minPoints

    def forward(self, points):
        """
        points.shape = [N, dim]
        """
        num_points = points.size()[0]
        labels = torch.ones((num_points,)) * -2
        cluster_id = 0
        for p in range(num_points):
            if not (labels[p]) == -2:
                continue
            neighbors = self.region_query(points, points[p])
            if neighbors.size()[0] < self.minPoints:  # Noise
                labels[p] = -1
            else:
                self.grow_cluster(points, labels, p, neighbors, cluster_id)
                cluster_id += 1
        return labels.reshape((-1, 1))

    def grow_cluster(self, all_points, labels, point_id, neighbors, cluster_id):
        labels[point_id] = cluster_id
        i = 0
        while i < len(neighbors):
            point = neighbors[i]
            if labels[point] == -1:
                labels[point] = cluster_id
            elif labels[point] == -2:
                labels[point] = cluster_id
                new_neighbors = self.region_query(all_points, all_points[point])
                if len(new_neighbors) >= self.minPoints:
                    neighbors = neighbors + new_neighbors
            i += 1

    def distances(self, v1, v2):
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))

    def region_query(self, all_points, point):
        """
        Assumes all_points.shape = (N, dim) and point = (dim, )
        """
        d = self.distances(all_points, point[None, ...])
        return all_points[d.reshape((-1,)) < self.epsilon]


if __name__ == '__main__':
    module = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        DBScan(),
        torch.nn.Linear(1, 2)
    )
    MSELoss = torch.nn.MSELoss(reduction='none')
    module.train()
    optimizer = torch.optim.Adam(module.parameters())
    x = torch.Tensor([
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 1.0]
    ])
    y = module(x)
    labels = torch.Tensor([
        [-1, -1],
        [-1, -1]
    ])
    loss = MSELoss(labels, y).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
