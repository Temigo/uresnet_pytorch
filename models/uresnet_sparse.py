import torch
import sparseconvnet as scn


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

    def forward(self, coords, features):
        x = self.sparseModel((coords, features))
        x = self.linear(x)
        return x