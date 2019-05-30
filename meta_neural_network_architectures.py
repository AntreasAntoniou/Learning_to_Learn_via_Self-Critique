import logging
import math
from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out


def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    return output_dict


def extract_params_and_check_for_missing_keys(current_dict, layer_dict):
    params_dict = extract_top_level_dict(current_dict=current_dict)
    for key in layer_dict.keys():
        if key not in params_dict:
            params_dict[key] = None

    return params_dict


class MetaConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, groups=1, dilation_rate=1):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

        self.groups = groups

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)
        return out


class MetaConv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, groups=1, dilation_rate=1):
        """
        A MetaConv1D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaConv1dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

        self.groups = groups

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv1d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)
        return out


class MetaLinearLayer(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()
        b, c = input_shape

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.empty(num_filters, c))
        nn.init.xavier_uniform_(self.weights)

        logging.debug("debug message", self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        # print(x.shape)
        out = F.linear(input=x, weight=weight, bias=bias)

        return out

    def reset_parameters(self):
        self.weights.data = self.weights.data * 0.
        fan_in, fan_out = _calculate_fan_in_and_fan_out(self.weights)
        std = 1. * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        a_array = torch.ones(self.weights.shape) * a
        a_array.to(self.weights.device)
        self.weights.data = self.weights.data + torch.distributions.Uniform(low=-a_array, high=a_array).rsample().to(
            self.weights.device)
        if self.use_bias:
            self.bias.data = self.bias.data * 0.


class MetaBatchNormLayer(nn.Module):
    def __init__(self, num_features, args, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, meta_batch_norm=True,
                 use_per_step_bn_statistics=False):
        """
        A MetaBatchNorm layer. Applies the same functionality of a standard BatchNorm layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting. Also has the additional functionality of being able to store per step running stats and per step beta and gamma.
        :param num_features:
        :param device:
        :param args:
        :param eps:
        :param momentum:
        :param affine:
        :param track_running_stats:
        :param meta_batch_norm:
        :param no_learnable_params:
        :param use_per_step_bn_statistics:
        """
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args
        self.learnable_gamma = self.args.learnable_bn_gamma
        self.learnable_beta = self.args.learnable_bn_beta

        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(
                torch.zeros(args.num_support_set_steps + args.num_target_set_steps + 1, num_features),
                requires_grad=False)
            self.running_var = nn.Parameter(
                torch.ones(args.num_support_set_steps + args.num_target_set_steps + 1, num_features),
                requires_grad=False)
            self.bias = nn.Parameter(
                torch.zeros(args.num_support_set_steps + args.num_target_set_steps + 1, num_features),
                requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(
                torch.ones(args.num_support_set_steps + args.num_target_set_steps + 1, num_features),
                requires_grad=self.learnable_gamma)
        else:
            self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        if self.args.enable_inner_loop_optimizable_bn_params:
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)

        self.momentum = momentum

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propagates by applying a bach norm function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
        else:
            weight, bias = self.weight, self.bias

        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            if params is None:
                if not self.args.enable_inner_loop_optimizable_bn_params:
                    bias = self.bias[num_step]
                    weight = self.weight[num_step]
        else:
            running_mean = None
            running_var = None

        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)

        momentum = self.momentum

        output = F.batch_norm(input, running_mean, running_var, weight, bias,
                              training=True, momentum=momentum, eps=self.eps)

        return output

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean, requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var, requires_grad=False)

        self.to(self.weight.device)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class MetaConvNormLayerLeakyReLU(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False, groups=1, conv_type=MetaConv2dLayer):
        """
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param args: A named tuple containing the system's hyperparameters.
           :param device: The device to run the layer on.
           :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
           :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
           meta-conv etc.
           :param input_shape: The image input shape in the form (b, c, h, w)
           :param num_filters: number of filters for convolutional layer
           :param kernel_size: the kernel size of the convolutional layer
           :param stride: the stride of the convolutional layer
           :param padding: the bias of the convolutional layer
           :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaConvNormLayerLeakyReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.conv_type = conv_type
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros(self.input_shape)

        out = x

        self.conv = self.conv_type(in_channels=out.shape[1], out_channels=self.num_filters,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride, padding=self.padding, use_bias=self.use_bias,
                                   groups=self.groups)

        out = self.conv(out)
        if type(out) == tuple:
            out, _ = out

        if self.normalization:
            if self.args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(out.shape[1], track_running_stats=True,
                                                     meta_batch_norm=self.meta_layer,
                                                     use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                     args=self.args)

            out = self.norm_layer(out, num_step=0)

        out = F.leaky_relu(out)

        print(out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        batch_norm_params = None
        conv_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']

            conv_params = params['conv']

        out = x

        out = self.conv(out, params=conv_params)

        if type(out) == tuple:
            out, attentio = out

        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          backup_running_statistics=backup_running_statistics)

        out = F.leaky_relu(out)
        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()


class VGGActivationNormNetwork(nn.Module):
    def __init__(self, im_shape, num_output_classes, args, device, num_stages=None, meta_classifier=True):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(VGGActivationNormNetwork, self).__init__()
        b, c, self.h, self.w = im_shape
        self.device = device
        self.total_layers = 0
        self.args = args
        self.upscale_shapes = []
        self.cnn_filters = args.cnn_num_filters
        self.input_shape = list(im_shape)
        if num_stages is not None:
            self.num_stages = num_stages
        else:
            self.num_stages = args.num_stages

        self.num_output_classes = num_output_classes

        if args.max_pooling:
            print("Using max pooling")
            self.conv_stride = 1
        else:
            print("Using strided convolutions")
            self.conv_stride = 2
        self.meta_classifier = meta_classifier

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        self.layer_dict = nn.ModuleDict()
        self.upscale_shapes.append(x.shape)
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                             num_filters=self.cnn_filters,
                                                                             kernel_size=3, stride=self.conv_stride,
                                                                             padding=self.args.conv_padding,
                                                                             use_bias=True, args=self.args,
                                                                             normalization=True,
                                                                             meta_layer=self.meta_classifier,
                                                                             no_bn_learnable_params=False)

            out = self.layer_dict['conv{}'.format(i)](out, training=True, num_step=0)

            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)

        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])

        self.encoder_features_shape = list(out.shape)
        out = out.view((out.shape[0], -1))

        self.layer_dict['linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
                                                    num_filters=self.num_output_classes, use_bias=True)

        out = self.layer_dict['linear'](out)
        print("VGGNetwork build", out.shape)

    def forward(self, x, num_step, dropout_training=None, params=None, training=False,
                backup_running_statistics=False, arg_max_softmax=False, output_features=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        for i in range(self.num_stages):
            out = self.layer_dict['conv{}'.format(i)](out, params=param_dict['conv{}'.format(i)], training=training,
                                                      backup_running_statistics=backup_running_statistics,
                                                      num_step=num_step)
            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)

        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])
        features = out
        out = out.view(out.size(0), -1)
        out = self.layer_dict['linear'](out, param_dict['linear'])
        if output_features:
            return out, features
        else:
            return out

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for name, module in self.named_modules():
            if type(module) == MetaBatchNormLayer:
                module.restore_backup_stats()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


class VGGActivationNormNetworkWithAttention(nn.Module):
    def __init__(self, im_shape, num_output_classes, args, device, use_spatial_attention, use_channel_wise_attention,
                 num_stages=None, meta_classifier=True):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(VGGActivationNormNetworkWithAttention, self).__init__()
        b, c, self.h, self.w = im_shape
        self.device = device
        self.total_layers = 0
        self.args = args
        self.upscale_shapes = []
        self.cnn_filters = args.cnn_num_filters
        self.input_shape = list(im_shape)
        self.use_spatial_attention = use_spatial_attention
        self.use_channel_wise_attention = use_channel_wise_attention
        if num_stages is not None:
            self.num_stages = num_stages
        else:
            self.num_stages = args.num_stages

        self.num_output_classes = num_output_classes

        if args.max_pooling:
            print("Using max pooling")
            self.conv_stride = 1
        else:
            print("Using strided convolutions")
            self.conv_stride = 2
        self.meta_classifier = meta_classifier

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        self.layer_dict = nn.ModuleDict()
        self.upscale_shapes.append(x.shape)

        for i in range(self.num_stages):

            if self.use_channel_wise_attention:
                out_channels = F.avg_pool2d(out, out.shape[-1]).squeeze()

                print(out_channels.shape)
                out_channels = out_channels.view(out_channels.shape[0], -1)

                self.layer_dict['channel_wise_attention_linear_pred_network_{}'.format(i)] = MetaLinearLayer(
                    input_shape=out_channels.shape, num_filters=out_channels.shape[-1], use_bias=True)
                channel_wise_attention_regions = self.layer_dict[
                    'channel_wise_attention_linear_pred_network_{}'.format(i)].forward(
                    out_channels, params=None)

                channel_wise_attention_regions = F.sigmoid(channel_wise_attention_regions)
                out = out * channel_wise_attention_regions.unsqueeze(2).unsqueeze(2)

            self.layer_dict['conv{}'.format(i)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                             num_filters=self.cnn_filters,
                                                                             kernel_size=3, stride=self.conv_stride,
                                                                             padding=self.args.conv_padding,
                                                                             use_bias=True, args=self.args,
                                                                             normalization=True,
                                                                             meta_layer=self.meta_classifier,
                                                                             no_bn_learnable_params=False,
                                                                             groups=1)
            out = self.layer_dict['conv{}'.format(i)](out, training=True, num_step=0)

            if self.args.max_pooling and not self.use_spatial_attention:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)

        if self.use_channel_wise_attention:
            out_channels = F.avg_pool2d(out, out.shape[-1]).squeeze()

            print(out_channels.shape)
            out_channels = out_channels.view(out_channels.shape[0], -1)


            self.layer_dict['channel_wise_attention_linear_pred_network_out'] = MetaLinearLayer(
                input_shape=out_channels.shape, num_filters=out_channels.shape[-1], use_bias=True)

            channel_wise_attention_regions = self.layer_dict[
                'channel_wise_attention_linear_pred_network_out'].forward(
                out_channels, params=None)

            channel_wise_attention_regions = F.sigmoid(channel_wise_attention_regions)
            out = out * channel_wise_attention_regions.unsqueeze(2).unsqueeze(2)

        if self.use_spatial_attention:
            b, c, h, w = out.shape

            out_channels_mean = F.avg_pool2d(out, out.shape[-1]).squeeze()
            out_spatial_mean = torch.mean(out.view(out.shape[0], out.shape[1], -1), dim=1).squeeze()
            out_flatten = torch.cat([out_channels_mean, out_spatial_mean], dim=1)

            self.layer_dict['spatial_attention_linear_network'] = MetaLinearLayer(
                input_shape=out_flatten.shape, num_filters=h * w, use_bias=True)
            spatial_region_selected = self.layer_dict['spatial_attention_linear_network'].forward(out_flatten,
                                                                                                  params=None)
            spatial_region_selected = F.softmax(spatial_region_selected, dim=1).unsqueeze(2)

            out = out.view(out.shape[0], -1, h * w)
            out = torch.bmm(out, spatial_region_selected)

            out = out.squeeze()
        else:
            out = F.avg_pool2d(out, out.shape[-1])

        if not self.args.max_pooling and not self.use_spatial_attention:
            out = F.avg_pool2d(out, out.shape[2])

        self.encoder_features_shape = list(out.shape)
        out = out.view((out.shape[0], -1))

        self.layer_dict['linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
                                                    num_filters=self.num_output_classes, use_bias=True)

        out = self.layer_dict['linear'](out)
        print("VGGNetwork build", out.shape)

    def forward(self, x, num_step, dropout_training=None, params=None, training=False,
                backup_running_statistics=False, arg_max_softmax=False, return_features=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        for i in range(self.num_stages):

            if self.use_channel_wise_attention:
                out_channels = F.avg_pool2d(out, out.shape[-1]).squeeze()

                b, c = out_channels.shape[:2]
                out_channels = out_channels.view(out_channels.shape[0], c)

                channel_wise_attention_regions = self.layer_dict[
                    'channel_wise_attention_linear_pred_network_{}'.format(i)].forward(
                    out_channels, params=param_dict['channel_wise_attention_linear_pred_network_{}'.format(i)])
                channel_wise_attention_regions = F.sigmoid(channel_wise_attention_regions)
                out = out * channel_wise_attention_regions.unsqueeze(2).unsqueeze(2)

            out = self.layer_dict['conv{}'.format(i)](out, params=param_dict['conv{}'.format(i)], training=training,
                                                      backup_running_statistics=backup_running_statistics,
                                                      num_step=num_step)

            if self.args.max_pooling and not self.use_spatial_attention:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)

        if self.use_channel_wise_attention:
            out_channels = F.avg_pool2d(out, out.shape[-1]).squeeze()
            out_channels = out_channels.view(out_channels.shape[0], -1)


            channel_wise_attention_regions = self.layer_dict[
                'channel_wise_attention_linear_pred_network_out'].forward(
                out_channels, params=param_dict['channel_wise_attention_linear_pred_network_out'])

            channel_wise_attention_regions = F.sigmoid(channel_wise_attention_regions)

            out = out * channel_wise_attention_regions.unsqueeze(2).unsqueeze(2)

        features = out
        if self.use_spatial_attention:
            b, c, h, w = out.shape
            out_channels_mean = F.avg_pool2d(out, out.shape[-1]).squeeze()

            out_spatial_mean = torch.mean(out.view(out.shape[0], out.shape[1], -1), dim=1).squeeze()
            out_flatten = torch.cat([out_channels_mean, out_spatial_mean], dim=1)
            spatial_region_selected = self.layer_dict['spatial_attention_linear_network'].forward(out_flatten,
                                                                                                  params=param_dict[
                                                                                                      'spatial_attention_linear_network'])
            spatial_region_selected = F.softmax(spatial_region_selected, dim=1).unsqueeze(2)
            out = out.view(out.shape[0], -1, h * w)
            out = torch.bmm(out, spatial_region_selected)
            # print(out.shape)
            # out = out.sum(2)
            out = out.squeeze()
        else:
            out = F.avg_pool2d(out, out.shape[-1])

        out = out.view(out.size(0), -1)
        out = self.layer_dict['linear'](out, param_dict['linear'])
        if return_features:
            return out, features
        else:
            return out

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for name, module in self.named_modules():
            if type(module) == MetaBatchNormLayer:
                module.restore_backup_stats()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


class FCCActivationNormNetwork(nn.Module):
    def __init__(self, im_shape, num_output_classes, args, device, use_bn, num_stages=None, use_bias=True,
                 meta_classifier=True):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(FCCActivationNormNetwork, self).__init__()
        self.device = device
        self.args = args
        self.input_shape = list(im_shape)
        self.num_output_classes = num_output_classes
        self.meta_classifier = meta_classifier
        self.num_stages = num_stages
        self.use_bias = use_bias
        self.use_bn = use_bn

        self.build_network()

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        out = out.view(out.size(0), -1)
        self.layer_dict = nn.ModuleDict()

        for i in range(self.num_stages):
            self.layer_dict['fcc_{}'.format(i)] = MetaLinearLayer(input_shape=out.shape, num_filters=40, use_bias=False)
            out = self.layer_dict['fcc_{}'.format(i)].forward(out)
            if self.use_bn:
                self.layer_dict['fcc_bn_{}'.format(i)] = MetaBatchNormLayer(num_features=out.shape[1], args=self.args,
                                                                            use_per_step_bn_statistics=True)
                out = self.layer_dict['fcc_bn_{}'.format(i)].forward(out, num_step=0)
            out = F.leaky_relu(out)

        out = out.view(out.shape[0], -1)

        self.layer_dict['preds_linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
                                                          num_filters=self.num_output_classes, use_bias=self.use_bias)

        out = self.layer_dict['preds_linear'](out)
        print("FCCActivationNormNetwork build", out.shape)

    def forward(self, x, num_step, params=None, training=False,
                backup_running_statistics=False, return_features=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x
        out = out.view(out.size(0), -1)
        for i in range(self.num_stages):
            out = self.layer_dict['fcc_{}'.format(i)](out, params=param_dict['fcc_{}'.format(i)])
            if self.use_bn:
                out = self.layer_dict['fcc_bn_{}'.format(i)].forward(out, num_step=num_step,
                                                                     params=None, training=training,
                                                                     backup_running_statistics=backup_running_statistics)
            out = F.leaky_relu(out)
            features = out

        out = out.view(out.size(0), -1)
        out = self.layer_dict['preds_linear'](out, param_dict['preds_linear'])

        if return_features:
            return out, features
        else:
            return out

    def reset_parameters(self):
        for name, module in self.named_modules():
            if type(module) == MetaLinearLayer:
                # print("reset", name)
                module.reset_parameters()

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for name, module in self.named_modules():
            if type(module) == MetaBatchNormLayer:
                module.restore_backup_stats()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


class DenseNetActivationNormNetwork(nn.Module):
    def __init__(self, im_shape, num_output_filters, num_stages, num_blocks_per_stage, num_filters, num_output_classes,
                 args):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(DenseNetActivationNormNetwork, self).__init__()
        self.args = args
        self.num_filters = num_filters
        self.num_output_filters = num_output_filters
        self.input_shape = list(im_shape)
        self.num_stages = num_stages
        self.num_output_classes = num_output_classes
        self.num_blocks_per_stage = num_blocks_per_stage
        self.layer_dict = nn.ModuleDict()
        self.build_network()

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['stem_conv'] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                  num_filters=self.num_filters,
                                                                  kernel_size=3, stride=1,
                                                                  padding=1,
                                                                  use_bias=True, args=self.args,
                                                                  normalization=True,
                                                                  meta_layer=True,
                                                                  no_bn_learnable_params=False)
        out = self.layer_dict['stem_conv'](out, training=True, num_step=0)
        out = F.avg_pool2d(out, 2)
        self.layer_dict['transition_init_conv'] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                             num_filters=out.shape[1],
                                                                             kernel_size=1, stride=1,
                                                                             padding=1,
                                                                             use_bias=False, args=self.args,
                                                                             normalization=True,
                                                                             meta_layer=True,
                                                                             no_bn_learnable_params=False)
        out = self.layer_dict['transition_init_conv'](out, training=True, num_step=0)

        for i in range(self.num_stages):
            for j in range(self.num_blocks_per_stage):
                self.layer_dict['conv_{}_{}'.format(i, j)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                                        num_filters=self.num_filters,
                                                                                        kernel_size=3, stride=1,
                                                                                        padding=1,
                                                                                        use_bias=False, args=self.args,
                                                                                        normalization=True,
                                                                                        meta_layer=True,
                                                                                        no_bn_learnable_params=False)
                cur = self.layer_dict['conv_{}_{}'.format(i, j)](out, training=True, num_step=0)
                out = torch.cat([out, cur], dim=1)

            out = F.avg_pool2d(out, 2)
            self.layer_dict['transition_conv_{}'.format(i)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                                         num_filters=out.shape[1],
                                                                                         kernel_size=1, stride=1,
                                                                                         padding=0,
                                                                                         use_bias=False, args=self.args,
                                                                                         normalization=True,
                                                                                         meta_layer=True,
                                                                                         no_bn_learnable_params=False)
            out = self.layer_dict['transition_conv_{}'.format(i)](out, training=True, num_step=0)


        out = F.avg_pool2d(out, out.shape[2])

        out = out.view(out.shape[0], -1)

        self.layer_dict['linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
                                                    num_filters=self.num_output_classes, use_bias=True)

        out = self.layer_dict['linear'](out)
        print("VGGNetwork build", out.shape)

    def forward(self, x, num_step, params=None, training=False,
                backup_running_statistics=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        out = self.layer_dict['stem_conv'](out, params=param_dict['stem_conv'], training=training,
                                           backup_running_statistics=backup_running_statistics,
                                           num_step=num_step)
        out = F.avg_pool2d(out, 2)
        out = self.layer_dict['transition_init_conv'](out, params=param_dict['transition_init_conv'], training=training,
                                                      backup_running_statistics=backup_running_statistics,
                                                      num_step=num_step)
        for i in range(self.num_stages):
            for j in range(self.num_blocks_per_stage):
                cur = self.layer_dict['conv_{}_{}'.format(i, j)](out, params=param_dict['conv_{}_{}'.format(i, j)],
                                                                 training=training,
                                                                 backup_running_statistics=backup_running_statistics,
                                                                 num_step=num_step)
                out = torch.cat([out, cur], dim=1)

            out = F.avg_pool2d(out, 2)
            out = self.layer_dict['transition_conv_{}'.format(i)](out, training=training,
                                                                  backup_running_statistics=backup_running_statistics,
                                                                  num_step=num_step,
                                                                  params=param_dict['transition_conv_{}'.format(i)])


        out = F.avg_pool2d(out, out.shape[2])

        out = out.view(out.size(0), -1)
        out = self.layer_dict['linear'](out, param_dict['linear'])

        return out

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for name, module in self.named_modules():
            if type(module) == MetaBatchNormLayer:
                module.restore_backup_stats()

        # if self.args.norm_layer == "batch_norm":
        #     self.layer_dict['fc_norm_layer'].restore_backup_stats()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None
