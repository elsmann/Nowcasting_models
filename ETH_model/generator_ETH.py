"""Generator implementation."""

import functools
import tensorflow as tf
import sonnet as snt
import discriminator
import latent_stack
import layers


class Generator(snt.Module):
  """Generator for the proposed model."""

  def __init__(self, lead_time=90, time_delta=5,  strategy = None):
    """Constructor.

    Args:
      lead_time: last lead time for the generator to predict. Default: 90 min.
      time_delta: time step between predictions. Default: 5 min.
    """
    # for sonnet
    super().__init__()

    self._cond_stack = ConditioningStack()
    self._sampler = Sampler(lead_time, time_delta,  strategy=strategy)

  def __call__(self, radar_inputs, eth_inputs):
    """Connect to a graph.

    Args:
      inputs: a batch of inputs on the shape [batch_size, time, h, w, 1].
    Returns:
      predictions: a batch of predictions in the form
        [batch_size, num_lead_times, h, w, 1].
    """
    _, _, height, width, _ = radar_inputs.shape.as_list()
    initial_states = self._cond_stack(radar_inputs, eth_inputs)
    predictions = self._sampler(initial_states, [height, width])
    return predictions

  def get_variables(self):
    """Get all variables of the module."""
    pass



class ConditioningStack(snt.Module):
  """Conditioning Stack for the Generator."""

  def __init__(self):
    # for sonnet
    super().__init__()

    self._block1radar = discriminator.DBlock(output_channels=48, downsample=True)
    self._conv_mix1radar =  layers.SNConv2D(output_channels=48, kernel_size=3)
    self._block2radar = discriminator.DBlock(output_channels=96, downsample=True)
    self._conv_mix2radar =  layers.SNConv2D(output_channels=96, kernel_size=3)
    self._block3radar = discriminator.DBlock(output_channels=192, downsample=True)
    self._conv_mix3radar =  layers.SNConv2D(output_channels=192, kernel_size=3)
    self._block4radar = discriminator.DBlock(output_channels=384, downsample=True)
    self._conv_mix4radar =  layers.SNConv2D(output_channels=384, kernel_size=3)

    self._block1eth = discriminator.DBlock(output_channels=48, downsample=True)
    self._conv_mix1eth =  layers.SNConv2D(output_channels=48, kernel_size=3)
    self._block2eth = discriminator.DBlock(output_channels=96, downsample=True)
    self._conv_mix2eth =  layers.SNConv2D(output_channels=96, kernel_size=3)
    self._block3eth = discriminator.DBlock(output_channels=192, downsample=True)
    self._conv_mix3eth =  layers.SNConv2D(output_channels=192, kernel_size=3)
    self._block4eth = discriminator.DBlock(output_channels=384, downsample=True)
    self._conv_mix4eth =  layers.SNConv2D(output_channels=384, kernel_size=3)

  def __call__(self, inputs_radar, inputs_ETH):
    # Space to depth conversion of 256x256x1 radar to 128x128x4 hiddens.
    h0_radar = batch_apply(
        functools.partial(tf.nn.space_to_depth, block_size=2), inputs_radar)
    h0_ETH = batch_apply(
      functools.partial(tf.nn.space_to_depth, block_size=2), inputs_ETH)

    # Downsampling residual D Blocks.
    h1_radar = time_apply(self._block1radar, h0_radar)
    h2_radar = time_apply(self._block2radar, h1_radar)
    h3_radar = time_apply(self._block3radar, h2_radar)
    h4_radar = time_apply(self._block4radar, h3_radar)

    h1_ETH = time_apply(self._block1eth, h0_ETH)
    h2_ETH = time_apply(self._block2eth, h1_ETH)
    h3_ETH = time_apply(self._block3eth, h2_ETH)
    h4_ETH = time_apply(self._block4eth, h3_ETH)


    # Spectrally normalized convolutions, followed by rectified linear units.
    init_state_1_radar = self._mixing_layer(h1_radar, self._conv_mix1radar)
    init_state_2_radar = self._mixing_layer(h2_radar, self._conv_mix2radar)
    init_state_3_radar = self._mixing_layer(h3_radar, self._conv_mix3radar)
    init_state_4_radar = self._mixing_layer(h4_radar, self._conv_mix4radar)
    # Spectrally normalized convolutions, followed by rectified linear units.
    init_state_1_ETH = self._mixing_layer(h1_ETH, self._conv_mix1eth)
    init_state_2_ETH = self._mixing_layer(h2_ETH, self._conv_mix2eth)
    init_state_3_ETH = self._mixing_layer(h3_ETH, self._conv_mix3eth)
    init_state_4_ETH = self._mixing_layer(h4_ETH, self._conv_mix4eth)
    # in  this add eth
    init_state_1 = tf.concat([init_state_1_radar,init_state_1_ETH], -1)
    init_state_2 = tf.concat([init_state_2_radar,init_state_2_ETH], -1)
    init_state_3 = tf.concat([init_state_3_radar,init_state_3_ETH], -1)
    init_state_4 = tf.concat([init_state_4_radar,init_state_4_ETH], -1)

    # Return a stack of conditioning representations of size (64x64x48, 32x32x96,
    # 16x16x192 and 8x8x384.) last dimension times 2
    #print("inital states", init_state_1, init_state_2, init_state_3, init_state_4)
    return init_state_1, init_state_2, init_state_3, init_state_4

  def _mixing_layer(self, inputs, conv_block):
    # Convert from [batch_size, time, h, w, c] -> [batch_size, h, w, c * time]
    # then perform convolution on the output while preserving number of c.
    stacked_inputs = tf.concat(tf.unstack(inputs, axis=1), axis=-1)
    return tf.nn.relu(conv_block(stacked_inputs))


class Sampler(snt.Module):
  """Sampler for the Generator."""

  def __init__(self, lead_time=90, time_delta=5, strategy = None):
    # for sonnet
    super().__init__()

    self._num_predictions = lead_time // time_delta
    self._latent_stack = latent_stack.LatentCondStack(strategy=strategy)

    self._conv_gru4 = ConvGRU(num_channels=384*2)
    self._conv4 = layers.SNConv2D(kernel_size=1, output_channels=768)
    self._gblock4 = GBlock(output_channels=768)
    self._g_up_block4 = UpsampleGBlock(output_channels=384)

    self._conv_gru3 = ConvGRU(num_channels=192*2)
    self._conv3 = layers.SNConv2D(kernel_size=1, output_channels=384)
    self._gblock3 = GBlock(output_channels=384)
    self._g_up_block3 = UpsampleGBlock(output_channels=192)

    self._conv_gru2 = ConvGRU(num_channels=96*2)
    self._conv2 = layers.SNConv2D(kernel_size=1, output_channels=192)
    self._gblock2 = GBlock(output_channels=192)
    self._g_up_block2 = UpsampleGBlock(output_channels=96)

    self._conv_gru1 = ConvGRU(num_channels=48*2)
    self._conv1 = layers.SNConv2D(kernel_size=1, output_channels=96)
    self._gblock1 = GBlock(output_channels=96)
    self._g_up_block1 = UpsampleGBlock(output_channels=48)

    self._bn = layers.BatchNorm()
    self._output_conv = layers.SNConv2D(kernel_size=1, output_channels=4)

  def __call__(self, initial_states, resolution):
    # intial states are float64 for some reasom

    init_state_1, init_state_2, init_state_3, init_state_4 = initial_states
    batch_size = init_state_1.shape.as_list()[0]

    # Latent conditioning stack.
    z = self._latent_stack(batch_size, resolution)
    hs = [z] * self._num_predictions

    # Layer 4 (bottom-most).
    # why can we change this to this?
    # hs, _ = tf.nn.static_rnn(self._conv_gru4, hs, init_state_4)
    hs = tf.stack(hs)
    hs, _ = snt.static_unroll(self._conv_gru4, hs, init_state_4)
    hs = tf.unstack(hs)
    hs = [self._conv4(h) for h in hs]
    hs = [self._gblock4(h) for h in hs]
    hs = [self._g_up_block4(h) for h in hs]

    # Layer 3.
    # hs, _ = tf.nn.static_rnn(self._conv_gru3, hs, init_state_3)
    hs = tf.stack(hs)
    hs, _ = snt.static_unroll(self._conv_gru3, hs, init_state_3)
    hs = tf.unstack(hs)
    hs = [self._conv3(h) for h in hs]
    hs = [self._gblock3(h) for h in hs]
    hs = [self._g_up_block3(h) for h in hs]

    # Layer 2.
    # hs, _ = tf.nn.static_rnn(self._conv_gru2, hs, init_state_2)

    hs = tf.stack(hs)
    hs, _ = snt.static_unroll(self._conv_gru2, hs, init_state_2)
    hs = tf.unstack(hs)
    hs = [self._conv2(h) for h in hs]
    hs = [self._gblock2(h) for h in hs]
    hs = [self._g_up_block2(h) for h in hs]

    # Layer 1 (top-most).
    hs = tf.stack(hs)
    hs, _ = snt.static_unroll(self._conv_gru1, hs, init_state_1)
    hs = tf.unstack(hs)
    # hs, _ = tf.nn.static_rnn(self._conv_gru1, hs, init_state_1)
    hs = [self._conv1(h) for h in hs]
    hs = [self._gblock1(h) for h in hs]
    hs = [self._g_up_block1(h) for h in hs]

    # Output layer.
    hs = [tf.nn.relu(self._bn(h)) for h in hs]
    hs = [self._output_conv(h) for h in hs]
    hs = [tf.nn.depth_to_space(h, 2) for h in hs]

    return tf.stack(hs, axis=1)


class GBlock(snt.Module):
  """Residual generator block without upsampling."""

  def __init__(self, output_channels, sn_eps=0.0001):
    # for sonnet
    super().__init__()

    self._output_channels = output_channels
    self._sn_eps = sn_eps
    self._conv_1x1 = layers.SNConv2D(
          self._output_channels, kernel_size=1, sn_eps=self._sn_eps)
    self._conv1_3x3 = layers.SNConv2D(
        output_channels, kernel_size=3, sn_eps=sn_eps)
    self._bn1 = layers.BatchNorm()
    self._conv2_3x3 = layers.SNConv2D(
        output_channels, kernel_size=3, sn_eps=sn_eps)
    self._bn2 = layers.BatchNorm()


  def __call__(self, inputs):
    input_channels = inputs.shape[-1]

    # Optional spectrally normalized 1x1 convolution.
    if input_channels != self._output_channels:
      sc = self._conv_1x1(inputs)
    else:
      sc = inputs

    # Two-layer residual connection, with batch normalization, nonlinearity and
    # 3x3 spectrally normalized convolution in each layer.
    h = tf.nn.relu(self._bn1(inputs))
    h = self._conv1_3x3(h)
    h = tf.nn.relu(self._bn2(h))
    h = self._conv2_3x3(h)

    # Residual connection.
    return h + sc


class UpsampleGBlock(snt.Module):
  """Upsampling residual generator block."""


  def __init__(self, output_channels, sn_eps=0.0001):
    # for sonnet
    super().__init__()

    self._conv_1x1 = layers.SNConv2D(
        output_channels, kernel_size=1, sn_eps=sn_eps)
    self._conv1_3x3 = layers.SNConv2D(
        output_channels, kernel_size=3, sn_eps=sn_eps)
    self._bn1 = layers.BatchNorm()
    self._conv2_3x3 = layers.SNConv2D(
        output_channels, kernel_size=3, sn_eps=sn_eps)
    self._bn2 = layers.BatchNorm()
    self._output_channels = output_channels

  def __call__(self, inputs):
    # x2 upsampling and spectrally normalized 1x1 convolution.
    sc = layers.upsample_nearest_neighbor(inputs, upsample_size=2)
    sc = self._conv_1x1(sc)

    # Two-layer residual connection, with batch normalization, nonlinearity and
    # 3x3 spectrally normalized convolution in each layer, and x2 upsampling in
    # the first layer.
    h = tf.nn.relu(self._bn1(inputs))
    h = layers.upsample_nearest_neighbor(h, upsample_size=2)
    h = self._conv1_3x3(h)
    h = tf.nn.relu(self._bn2(h))
    h = self._conv2_3x3(h)

    # Residual connection.
    return h + sc

class ConvGRU(snt.Module):
  """A ConvGRU implementation."""

  def __init__(self,num_channels, kernel_size=3, sn_eps=0.0001):
    """Constructor.

    Args:
      kernel_size: kernel size of the convolutions. Default: 3.
      sn_eps: constant for spectral normalization. Default: 1e-4.
    """
    # for sonnet
    super().__init__()

    self._kernel_size = kernel_size
    self._sn_eps = sn_eps
    self._read_gate_conv = layers.SNConv2D(
        num_channels, self._kernel_size, sn_eps=self._sn_eps)
    self._update_gate_conv = layers.SNConv2D(
        num_channels, self._kernel_size, sn_eps=self._sn_eps)
    self._output_conv = layers.SNConv2D(
        num_channels, self._kernel_size, sn_eps=self._sn_eps)

  def __call__(self, inputs, prev_state):

    # Concatenate the inputs and previous state along the channel axis.
    num_channels = prev_state.shape[-1]
    xh = tf.concat([inputs, prev_state], axis=-1)

    # Read gate of the GRU.
    read_gate = tf.math.sigmoid(self._read_gate_conv(xh))

    # Update gate of the GRU.
    update_gate = tf.math.sigmoid(self._update_gate_conv(xh))

    # Gate the inputs.
    gated_input = tf.concat([inputs, read_gate * prev_state], axis=-1)

    # Gate the cell and state / outputs.
    c = tf.nn.relu(self._output_conv(gated_input))
    out = update_gate * prev_state + (1. - update_gate) * c
    new_state = out

    return out, new_state
2
def time_apply(func, inputs):
  """Apply function func on each element of inputs along the time axis."""
  return layers.ApplyAlongAxis(func, axis=1)(inputs)


def batch_apply(func, inputs):
  """Apply function func on each element of inputs along the batch axis."""
  return layers.ApplyAlongAxis(func, axis=0)(inputs)
