import tensorflow as tf
import functools
import sonnet as snt
import numpy as np

"""Implementation of miscellaneous DL operations.

Implementation of many of the layers can be found in the Sonnet library.
https://github.com/deepmind/sonnet
"""


def downsample_avg_pool(x):
    """Utility function for downsampling by 2x2 average pooling."""
    # return tf.layers.average_pooling2d(x, 2, 2, data_format='channels_last')
    return tf.nn.avg_pool2d(x, 2, 2, padding='VALID', data_format='NHWC')


def downsample_avg_pool3d(x):
    """Utility function for downsampling by 2x2 average pooling."""
    # return tf.layers.average_pooling3d(x, 2, 2, data_format='channels_last')
    return tf.nn.avg_pool3d(x, 2, 2, padding='VALID', data_format='NDHWC')


def upsample_nearest_neighbor(inputs, upsample_size):
    """Nearest neighbor upsampling.

    Args:
      inputs: inputs of size [b, h, w, c] where b is the batch size, h the height,
        w the width, and c the number of channels.
      upsample_size: upsample size S.
    Returns:
      outputs: nearest neighbor upsampled inputs of size [b, s * h, s * w, c].
    """
    shape = inputs.shape.as_list()
    h, w = shape[1], shape[2]
    results = tf.image.resize(images=inputs, \
                              size=[upsample_size * h, upsample_size * w], \
                              method='nearest', preserve_aspect_ratio=False, antialias=False)

    return results


class Conv2D(snt.Module):
    def __init__(self, output_channels, kernel_size, stride=1, rate=1,
                 padding='SAME', use_bias=True, name=None):
        super().__init__(name=name)
        self.conv_2D = snt.Conv2D(output_channels, kernel_shape=kernel_size, \
                                  stride=stride, rate=rate, padding=padding, \
                                  with_bias=use_bias)

    # not defined w_init=None, b_init=None, data_format='NHWC'
    def __call__(self, tensor):
        output = self.conv_2D(tensor)
        return output


# from https://github.com/deepmind/sonnet/blob/v2/examples/little_gan_on_mnist.ipynb
class SpectralNormalizer(snt.Module):

    def __init__(self, epsilon=1e-12, name=None):
        super().__init__(name=name)
        self.l2_normalize = functools.partial(tf.math.l2_normalize, epsilon=epsilon)

    @snt.once
    def _initialize(self, weights):
        init = self.l2_normalize(snt.initializers.TruncatedNormal()(
            shape=[1, weights.shape[-1]], dtype=weights.dtype))
        # 'u' tracks our estimate of the first spectral vector for the given weight.
        self.u = tf.Variable(init, name='u', trainable=False)

    def __call__(self, weights, is_training=True):
        self._initialize(weights)
        if is_training:
            # Do a power iteration and update u and weights.
            weights_matrix = tf.reshape(weights, [-1, weights.shape[-1]])
            v = self.l2_normalize(self.u @ tf.transpose(weights_matrix))
            v_w = v @ weights_matrix
            u = self.l2_normalize(v_w)
            sigma = tf.stop_gradient(tf.reshape(v_w @ tf.transpose(u), []))
            self.u.assign(u)
            weights.assign(weights / sigma)
        return weights


# class SNConv2D(Conv2D):
#  """2D convolution with spectral normalisation."""

#  def __init__(self, output_channels, kernel_size, stride=1, rate=1,
#               padding='SAME', sn_eps=0.0001, use_bias=True, name=None, w_init = snt.initializers.Orthogonal(), is_training=True):

#    super().__init__(output_channels,\
#                    kernel_size= kernel_size, stride = stride, rate = rate, \
#                    padding=padding, use_bias = use_bias)

#    self.spectral_normalizer = SpectralNormalizer(epsilon= sn_eps)


# so I need to add is_trainig to call function?

#  def __call__(self, tensor):


# call conv2d with tensor, so it initilaizes weights
#    super().conv_2D._initialize(tensor)

#    normed_weights= self.spectral_normalizer(super().conv_2D.w, is_training=is_training)
#    output = tf.matmul(tensor, normed_weights)
#    return output

class SNConv2D(snt.Module):
    """2D convolution with spectral normalisation."""

    def __init__(self, output_channels, kernel_size, stride=1, rate=1,
                 padding='SAME', sn_eps=0.0001, use_bias=True, name=None, \
                 data_format="NHWC", w_init=snt.initializers.Orthogonal()):
        super().__init__(name=name)
        self.conv_2D = snt.Conv2D(output_channels, kernel_shape=kernel_size, \
                                  stride=stride, rate=rate, padding=padding, \
                                  with_bias=use_bias)

        self.spectral_normalizer = SpectralNormalizer(epsilon=sn_eps)
        self.stride = stride
        self.padding = padding
        self.rate = rate
        self.data_format = data_format
        self.with_bias = use_bias

    def __call__(self, tensor, is_training=True):
        self.conv_2D._initialize(tensor)
        normed_weights = self.spectral_normalizer(self.conv_2D.w, is_training=is_training)

        # output = tf.matmul(tensor, normed_weights)
        # use confultional instead of matmul bc shapes don't match so it doesnt work
        # and other keras implementation also uses conv layer for this
        # change to use sonnet instead of tf conv, need to convert output then though
        #    output = snt.Conv2D(
        #          tensor,
        #          normed_weights,
        #          stride=self.stride,
        #          padding=self.padding,
        #          rate=self.rate,
        #          data_format=self.data_format)

        output = tf.nn.convolution(
            tensor,
            normed_weights,
            strides=self.stride,
            padding=self.padding,
            dilations=self.rate,
            data_format=self.data_format)

        if self.with_bias:
            # output = tf.add(output, self.conv_2D.b)
            output = tf.nn.bias_add(output, self.conv_2D.b, data_format=self.data_format)

        return output


class SNConv3D(snt.Module):
    """2D convolution with spectral regularisation."""

    def __init__(self, output_channels, kernel_size, stride=1, rate=1,
                 padding='SAME', sn_eps=0.0001, use_bias=True, name=None, \
                 data_format="NDHWC", w_init=snt.initializers.Orthogonal()):
        super().__init__(name=name)
        self.conv_3D = snt.Conv3D(output_channels, \
                                  kernel_shape=kernel_size, stride=stride, rate=rate, \
                                  padding=padding, with_bias=use_bias)

        self.spectral_normalizer = SpectralNormalizer(epsilon=sn_eps)
        self.stride = stride
        self.padding = padding
        self.rate = rate
        self.data_format = data_format
        self.with_bias = use_bias

    def __call__(self, tensor, is_training=True):
        self.conv_3D._initialize(tensor)
        normed_weights = self.spectral_normalizer(self.conv_3D.w, is_training=is_training)

        output = tf.nn.convolution(
            tensor,
            normed_weights,
            strides=self.stride,
            padding=self.padding,
            dilations=self.rate,
            data_format=self.data_format)

        if self.with_bias:
            output = tf.nn.bias_add(output, self.conv_3D.b, data_format=self.data_format)
        return output


class Linear(snt.Module):
    """Simple linear layer.

    Linear map from [batch_size, input_size] -> [batch_size, output_size].
    """

    def __init__(self, output_size, name=None):
        """Constructor."""
        super().__init__(name=name)
        self.lin = snt.Linear(output_size=output_size)
        # not defined with_bias=True, w_init=None, b_init=None

    def __call__(self, tensor):
        output = self.lin(tensor)
        return output


class BatchNorm(snt.Module):
    """Batch normalization."""

    def __init__(self, calc_sigma=True):
        """Constructor."""
        super().__init__()
        self.Batch_Norm = snt.BatchNorm(create_scale=calc_sigma, create_offset=True)
        # not defined  decay_rate=0.999, eps=1e-05, scale_init=None, offset_init=None, \
        # data_format='channels_last'

    def __call__(self, tensor, is_training=True):
        output = self.Batch_Norm(tensor, is_training=True)
        return output


class ApplyAlongAxis:
    """Layer for applying an operation on each element, along a specified axis."""

    def __init__(self, operation, axis=0):
        """Constructor."""
        self._operation = operation
        self._axis = axis

    def __call__(self, inputs):
        split_inputs = tf.unstack(inputs, axis=self._axis)
        res = [self._operation(x) for x in split_inputs]
        return tf.stack(res, axis=self._axis)


class ApplyAlongAxis_org:
    """Layer for applying an operation on each element, along a specified axis."""

    def __init__(self, operation, axis=0):
        self._operation = operation
        self._axis = axis

    def __call__(self, *args):
        """Apply the operation to each element of args along the specified axis."""
        split_inputs = [tf.unstack(arg, axis=self._axis) for arg in args]
        res = [self._operation(x) for x in zip(*split_inputs)]
        return tf.stack(res, axis=self._axis)


