"""
Layer parsing

These functions parse a list of layer configurations into a list of PyTorch
modules. Parameters in the config for each layer follow the order in Pytorch's
documentation. Excluding any of them will use the default ones. We can also
pass kwargs in a dict:

    ('layer_name', <list_of_args>, <dict_of_kwargs>)

This is a list of the configuration values supported:

Layer                   Paramaeters
==============================================================================
Convolution           : n-channels, size, stride, padding
Transposed Convolution: same, output_padding when stride > 1! (use kwargs)
Pooling               : size, stride, padding, type
Linear                : output size, fit bias
Flatten               : start dim, (optional, defaults=-1) end dim
Unflatten             : unflatten shape (have to pass the full shape)
Batch-norm            : dimensionality (1-2-3d)
Upsample              : upsample_shape (hard to infer automatically). Bilinear
Non-linearity         : pass whatever arguments that non-linearity supports.
SpatialBroadcast      : height, width (optional, defaults to value of height)

There is a method called transpose_layer_defs which allows for automatically
transposing the layer definitions for a decoder in a generative model. This
will automatically convert convolutions into transposed convolutions and
flattening to unflattening. However it will produce weird (but functionally
equivalent) orders of layers for ReLU before flattening, which means
unflattening in the corresponding decoder will be done before the ReLU.
"""


import numpy as np
import torch.nn as nn
from .math import *

from src.layers.spatial import (
    PositionConcat,
    PositionEmbedding1D,
    PositionEmbedding2D,
    SpatialBroadcast,
)
from src.layers.scs import SharpenedCosineSimilarity


def create_sequential(input_size, layer_defs, constructor=nn.Sequential):
    if isinstance(layer_defs, dict):
        layer_defs = layer_defs.items()
    layer_defs = preprocess_defs(layer_defs)

    module_layers, output_size = [], input_size

    for layer_type, args, kwargs in layer_defs:
        if layer_type == 'linear':
            layer, output_size = create_linear(output_size, args, kwargs)
        elif layer_type == 'conv':
            layer, output_size = create_conv(output_size, args, kwargs)
        elif layer_type == 'scs':
            layer, output_size = create_scs(output_size, args, kwargs)
        elif layer_type == 'tconv':
            layer, output_size = create_tconv(output_size, args, kwargs)
        elif layer_type == 'batch_norm':
            layer = create_batch_norm(args[0], output_size, args[1:], kwargs)
        elif layer_type == 'layer_norm':
            layer = create_layer_norm(output_size, args, kwargs)
        elif layer_type == 'group_norm':
            layer = create_group_norm(output_size, args, kwargs)
        elif layer_type == 'pool':
            layer, output_size = create_pool(output_size, args, kwargs)
        elif layer_type == 'dropout':
            layer = nn.Dropout2d(*args, **kwargs)
        elif layer_type == 'flatten':
            layer, output_size = create_flatten(output_size, *args)
        elif layer_type == 'unflatten':
            layer, output_size = create_unflatten(output_size, *args)
        elif layer_type == 'upsample':
            layer, output_size = create_upsample(output_size, *args)
        elif layer_type == 'spatbroad':
            layer, output_size = create_spatbroad(output_size, args, kwargs)
        elif layer_type == 'posemb1d':
            layer = create_posemb1d(output_size, args, kwargs)
        elif layer_type == 'posemb2d':
            layer = create_posemb2d(output_size, args, kwargs)
        elif layer_type == 'posconcat':
            layer, output_size = create_posconcat(output_size, args, kwargs)
        elif layer_type == 'permute':
            layer, output_size = create_permute(output_size, args, kwargs)
        elif layer_type == 'pixel_shuffle':
            layer, output_size = create_pixel_shuffle(output_size, args, kwargs)
        elif layer_type == 'flatten':
            layer, output_size = create_flatten(output_size, *args)
        elif layer_type == 'unflatten':
            layer, output_size = create_unflatten(output_size, *args)
        else:
            layer = get_nonlinearity(layer_type)(*args, **kwargs)

        module_layers.append(layer)

    return constructor(*module_layers)


def preprocess_defs(layer_defs):
    def preprocess(definition):
        if len(definition) == 1:
            return definition[0], [], {}
        elif len(definition) == 2 and isinstance(definition[1], (tuple, list)):
            return (*definition, {})
        elif len(definition) == 2 and isinstance(definition[1], dict):
            return definition[0], [], definition[1]
        elif len(definition) == 3:
            return definition
        raise ValueError('Invalid layer definition')

    return list(map(preprocess, layer_defs))


def get_nonlinearity(nonlinearity):
    if nonlinearity == 'relu':
        return nn.ReLU
    elif nonlinearity == 'sigmoid':
        return nn.Sigmoid
    elif nonlinearity == 'tanh':
        return nn.Tanh
    elif nonlinearity == 'lrelu':
        return nn.LeakyReLU
    elif nonlinearity == 'elu':
        return nn.ELU
    raise ValueError('Unrecognized non linearity: {}'.format(nonlinearity))


def create_linear(input_size, args, kwargs, transposed=False):
    if isinstance(input_size, tuple):
        input_size = list(input_size)

    if isinstance(input_size, list):
        in_features = input_size[-1]
    else:
        in_features = input_size

    if transposed:
        layer = nn.Linear(args[0], in_features, *args[1:], **kwargs)
    else:
        layer = nn.Linear(in_features, *args, **kwargs)

    if isinstance(input_size, list):
        input_size[-1] = args[0]
    else:
        input_size = args[0]

    return layer, input_size


def create_pool(input_size, args, kwargs):
    kernel_size, stride, padding, mode = args
    if mode == 'avg':
        layer = nn.AvgPool2d(kernel_size, stride, padding, **kwargs)
        #TODO: implement layer output shape computation
    elif mode == 'max':
        layer = nn.MaxPool2d(kernel_size, stride, **kwargs)
        output_size = maxpool2d_out_shape(input_size, kernel_size,
                                          stride, padding)
    elif mode == 'adapt':
        layer = nn.AdaptiveAvgPool2d(kernel_size, **kwargs)
        #TODO: implement layer output shape computation
    else:
        raise ValueError('Unrecognised pooling mode {}'.format(mode))
    return layer, output_size


def create_batch_norm(ndims, input_size, args, kwargs):
    if ndims == 1:
        return nn.BatchNorm1d(input_size, *args, **kwargs)
    if ndims == 2:
        return nn.BatchNorm2d(input_size[0], *args, **kwargs)
    if ndims == 3:
        return nn.BatchNorm3d(input_size[0], *args, **kwargs)
    raise ValueError('Unknown number of dimension for Batch-norm')


def create_layer_norm(input_size, args, kwargs):
    if isinstance(input_size, tuple):
        normalized_shape = input_size[args[0]:]
    else:
        normalized_shape = input_size
    return nn.LayerNorm(normalized_shape, *args[1:], **kwargs)


def create_group_norm(input_size, args, kwargs):
    return nn.GroupNorm(args[0], input_size[0], **kwargs)


def create_conv(input_size, args, kwargs):
    layer = nn.Conv2d(input_size[0], *args, **kwargs)
    output_size = conv2d_out_shape(input_size, *args)
    return layer, output_size


def create_scs(input_size, args, kwargs):
    layer = SharpenedCosineSimilarity(input_size[0], *args, **kwargs)
    output_size = conv2d_out_shape(input_size, *args)
    return layer, output_size


def create_pixel_shuffle(input_size, args, kwargs):
    upscale_factor = args[0]
    n_channels, height, width = input_size[-3:]

    layer = nn.PixelShuffle(upscale_factor)
    output_size = [n_channels // upscale_factor ** 2,
                   height * upscale_factor,
                   width * upscale_factor]

    output_size = list(output_size[:-3]) + output_size
    return layer, output_size


def create_tconv(input_size, args, kwargs):
    layer = nn.ConvTranspose2d(input_size[0], *args, **kwargs)
    output_size = transp_conv2d_out_shape(input_size, *args)
    return layer, output_size


def create_spatbroad(input_size, args, kwargs):
    kwargs = kwargs.copy()

    input_last = kwargs['input_last']
    sbc_type = kwargs.pop('type', 'uniform')

    if sbc_type == 'uniform':
        layer = SpatialBroadcast(*args, **kwargs)
    # elif sbc_type == 'weighted':
    #     layer = WeightedSBC(*args, **kwargs)
    else:
        raise ValueError('Unrecognized spatial'
                         'broadcast type {}'.format(sbc_type))

    if input_last:
        output_size = (*args, input_size)
    else:
        output_size = (input_size, *args)

    return layer, output_size


def create_posconcat(input_size, args, kwargs):
    layer = PositionConcat(*args, **kwargs)
    output_size = [layer.height, layer.width]
    n_channels = layer.grid.shape[layer.dim]
    output_size.insert(layer.dim, input_size[layer.dim] + n_channels)
    return layer, output_size


def create_posemb1d(input_size, args, kwargs):
    max_len, d_model = input_size
    return PositionEmbedding1D(max_len, d_model)


def create_posemb2d(input_size, args, kwargs):
    height, width, dim_size = input_size
    return PositionEmbedding2D(dim_size, height, width, **kwargs)


class Permute(nn.Module):
    def __init__(self, new_order) -> None:
        super().__init__()
        self.new_order = new_order

    def forward(self, inputs):
        return inputs.permute(self.new_order)

    def __repr__(self):
        template = 'Permute({})'.format(','.join(['{}'] * len(self.new_order)))
        return template.format(*self.new_order)


def create_permute(input_size, args, kwargs):
    layer = Permute(args)
    output_size = list(np.array(input_size)[np.asarray(args[1:]) - 1])
    return layer, output_size


def create_upsample(output_size, *args):
    layer = nn.UpsamplingBilinear2d(*args)
    output_size = (output_size[0], *args[0])
    return layer, output_size


def create_flatten(input_size, start_dim, end_dim=-1):
    layer = nn.Flatten(start_dim, end_dim)
    output_size = compute_flattened_size(input_size, start_dim, end_dim)
    return layer, output_size


def create_unflatten(input_size, dim, unflattened_shape):
    if not isinstance(input_size, (tuple, list)):
        input_size = (input_size,)
    assert input_size[dim - 1] == np.prod(unflattened_shape)
    layer = nn.Unflatten(dim, unflattened_shape)
    output_size = (
        *input_size[: dim - 1],
        *unflattened_shape,
        *input_size[dim:],
    )
    return layer, output_size
