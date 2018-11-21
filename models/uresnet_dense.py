import torch
import torch.nn as nn
import torch.nn.functional as F

# Accelerate *if all input sizes are same*
# torch.backends.cudnn.benchmark = True


def get_conv(is_3d):
    if is_3d:
        return nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d
    else:
        return nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d


def padding(kernel, stride, input_size):
    if input_size[-1] % stride == 0:
        p = max(kernel - stride, 0)
    else:
        p = max(kernel - (input_size[-1] % stride), 0)
    p1 = p // 2
    p2 = p - p1
    return (p1, p2,) * (len(input_size) - 2)


class ResNetModule(nn.Module):
    def __init__(self, is_3d, num_inputs, num_outputs, kernel=3, stride=1):
        super(ResNetModule, self).__init__()
        fn_conv, fn_conv_transpose, batch_norm = get_conv(is_3d)
        self.kernel, self.stride = kernel, stride

        # Shortcut path
        self.use_shortcut = (num_outputs != num_inputs or stride != 1)
        self.shortcut = torch.nn.Sequential(
            fn_conv(
                in_channels = num_inputs,
                out_channels = num_outputs,
                kernel_size = 1,
                stride      = stride,
                padding     = 0
            ),
            batch_norm(num_features = num_outputs)
        )

        # residual path
        self.residual1 = torch.nn.Sequential(
            fn_conv(
                in_channels = num_inputs,
                out_channels = num_outputs,
                kernel_size = kernel,
                stride      = stride,
                padding     = 0
            ),
            batch_norm(num_features = num_outputs)
        )

        self.residual2 = torch.nn.Sequential(
            fn_conv(
                in_channels = num_outputs,
                out_channels = num_outputs,
                kernel_size = kernel,
                stride      = 1,
                padding     = 0
            ),
            batch_norm(num_features = num_outputs)
        )

    def forward(self, input_tensor):
        if not self.use_shortcut:
            shortcut = input_tensor
        else:
            shortcut = F.pad(input_tensor, padding(self.shortcut[0].kernel_size[0], self.shortcut[0].stride[0], input_tensor.size()), mode='replicate')
            shortcut = self.shortcut(shortcut)
        # FIXME padding value
        residual = F.pad(input_tensor, padding(self.residual1[0].kernel_size[0], self.residual1[0].stride[0], input_tensor.size()), mode='replicate')
        residual = self.residual1(residual)
        residual = F.pad(residual, padding(self.residual2[0].kernel_size[0], self.residual2[0].stride[0], residual.size()), mode='replicate')
        residual = self.residual2(residual)
        return F.relu(shortcut + residual)


class DoubleResnet(nn.Module):
    def __init__(self, is_3d, num_inputs, num_outputs, kernel=3, stride=1):
        super(DoubleResnet, self).__init__()

        self.resnet1 = ResNetModule(
            is_3d = is_3d,
            num_inputs = num_inputs,
            num_outputs = num_outputs,
            kernel = kernel,
            stride = stride
        )
        self.resnet2 = ResNetModule(
            is_3d = is_3d,
            num_inputs = num_outputs,
            num_outputs = num_outputs,
            kernel = kernel,
            stride = 1
        )

    def forward(self, input_tensor):
        resnet = self.resnet1(input_tensor)
        resnet = self.resnet2(resnet)
        return resnet


class UResNet(nn.Module):
    def __init__(self, flags):
        super(UResNet, self).__init__()
        # Parameters
        self._flags = flags
        fn_conv, fn_conv_transpose, batch_norm = get_conv(is_3d)
        self.is_3d = flags.DATA_DIM == 3
        self.base_num_outputs = flags.URESNET_FILTERS
        self._num_strides = flags.URESNET_NUM_STRIDES
        self.num_inputs = 1  # number of channels of input image
        self.num_classes = flags.NUM_CLASS

        # Define layers
        self.conv1 = torch.nn.Sequential(
            fn_conv(
                in_channels = self.num_inputs,
                out_channels = self.base_num_outputs,
                kernel_size = 3,
                stride = 1,
                padding = 0 # FIXME 'same' in tensorflow
            ),
            batch_norm(num_features=self.base_num_outputs),
            torch.nn.ReLU()
        )
        # Encoding steps
        self.double_resnet = nn.ModuleList()
        current_num_outputs = self.base_num_outputs
        for step in xrange(self._num_strides):
            self.double_resnet.append(DoubleResnet(
                is_3d = self.is_3d,
                num_inputs = current_num_outputs,
                num_outputs = current_num_outputs * 2,
                kernel = 3,
                stride = 2
            ))
            current_num_outputs *= 2

        # Decoding steps
        self.decode_conv = nn.ModuleList()
        self.decode_double_resnet = nn.ModuleList()
        for step in xrange(self._num_strides):
            self.decode_double_resnet.append(DoubleResnet(
                is_3d = self.is_3d,
                num_inputs = current_num_outputs,
                num_outputs = current_num_outputs / 2,
                kernel = 3,
                stride = 1
            ))
            self.decode_conv.append(torch.nn.Sequential(
                fn_conv_transpose(
                    in_channels = current_num_outputs,
                    out_channels = current_num_outputs / 2,
                    kernel_size = 3,
                    stride = 2,
                    padding=1,
                    output_padding=1
                ),
                batch_norm(num_features=current_num_outputs / 2),
                torch.nn.ReLU()
            ))
            current_num_outputs /= 2

        self.conv2 = torch.nn.Sequential(
            fn_conv(
                in_channels = current_num_outputs,
                out_channels = self.base_num_outputs,
                padding = 0,
                kernel_size = 3,
                stride = 1
            ),
            batch_norm(num_features=current_num_outputs),
            torch.nn.ReLU()
        )

        self.conv3 = torch.nn.Sequential(
            fn_conv(
                in_channels = self.base_num_outputs,
                out_channels = self.num_classes,
                padding = 0,
                kernel_size = 3,
                stride = 1
            ),
            batch_norm(num_features=self.num_classes)
        )

    def forward(self, input):
        conv_feature_map = {}
        net = F.pad(input, padding(self.conv1[0].kernel_size[0], self.conv1[0].stride[0], input.size()), mode='replicate')
        net = self.conv1(net)
        conv_feature_map[net.size()[1]] = net
        # Encoding steps
        for step in xrange(self._num_strides):
            net = self.double_resnet[step](net)
            conv_feature_map[net.size()[1]] = net
        # Decoding steps
        for step in xrange(self._num_strides):
            # num_outputs = net.size()[1] / 2
            net = self.decode_conv[step](net)
            net = torch.cat((net, conv_feature_map[net.size()[1]]), dim=1)
            net = self.decode_double_resnet[step](net)
        # Final conv layers
        net = F.pad(net, padding(self.conv2[0].kernel_size[0], self.conv2[0].stride[0], net.size()), mode='replicate')
        net = self.conv2(net)
        net = F.pad(net, padding(self.conv3[0].kernel_size[0], self.conv3[0].stride[0], net.size()), mode='replicate')
        net = self.conv3(net)
        return net
