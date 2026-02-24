:title: conv
:type: module
:lang: python


The :obj:`conv` module provides classes for performing convolution operations in
various dimensions (1D, 2D, and 3D) on tensor inputs. These convolution
operations are core building blocks for neural networks, especially in computer
vision and sequence processing tasks.

Here's an example demonstrating how to use a 1D convolution:

.. code-block:: python

    import max.nn as nn
    from max.graph import Graph, ops, Weight, DeviceRef
    from max.dtype import DType
    import numpy as np

    with Graph(name="conv_example") as graph:
        # Define dimensions
        batch_size = 2
        seq_length = 10
        in_channels = 16
        out_channels = 32
        kernel_size = 3

        # Create input tensor [batch_size, sequence_length, channels]
        x_data = np.zeros((batch_size, seq_length, in_channels), dtype=np.float32)
        x = ops.constant(x_data, dtype=DType.float32, device=DeviceRef.CPU())

        # Create weights for convolution
        filter_1d = Weight(
            name="filter_weight",
            dtype=DType.float32,
            shape=[kernel_size, in_channels, out_channels]
            device=DeviceRef.CPU()
        )
        bias_1d = Weight(
            name="bias_weight",
            dtype=DType.float32,
            shape=[out_channels]
            device=DeviceRef.CPU()
        )

        # Create and apply Conv1D layer
        conv1d = nn.Conv1D(
            filter=filter_1d,
            bias=bias_1d,
            stride=1,
            padding=1
        )

        output_1d = conv1d(x)
        print(f"Conv1D output shape: {output_1d.shape}")
        # Output: Conv1D output shape: [Dim(2), Dim(10), Dim(32)]

.. automodule:: max.nn.conv
   :members:
   :undoc-members:
