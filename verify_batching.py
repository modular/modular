
import sys
import os
from unittest.mock import MagicMock

# Add the project root to sys.path
sys.path.append("/home/user0/Documents/gits/at_the_speed_of_light/modular/max/python")

# Mock the entire max.engine and max.serve modules to avoid protobuf issues
sys.modules['max.engine'] = MagicMock()
sys.modules['max.engine.api'] = MagicMock()
sys.modules['max.serve'] = MagicMock()
sys.modules['max.serve.kvcache_agent'] = MagicMock()
sys.modules['max._core'] = MagicMock()
sys.modules['max._core.engine'] = MagicMock()

import max.experimental.functional as F
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import Graph, TensorType
from max.pipelines.architectures.z_image_module_v3.nn.transformer_z_image import ZImageTransformer2DModel

def test_transformer_batching():
    print("Testing ZImageTransformer2DModel batching...")

    # Mock parameters
    dim = 3840
    n_heads = 30
    head_dim = dim // n_heads
    axes_dims = [32, 48, 48] # sum = 128 = head_dim
    axes_lens = [1024, 512, 512]

    # Initialize model
    model = ZImageTransformer2DModel(
        dim=dim,
        n_heads=n_heads,
        axes_dims=axes_dims,
        axes_lens=axes_lens,
        n_layers=1, # Small for testing
        n_refiner_layers=1
    )

    # Batch size 2
    B = 2
    C = 16
    F_dim = 1
    H = 128
    W = 128
    cap_seq_len = 75

    # Input types for compilation
    x_type = TensorType(DType.bfloat16, shape=(B, C, F_dim, H, W))
    t_type = TensorType(DType.float32, shape=(B,))
    cap_type = TensorType(DType.bfloat16, shape=(B, cap_seq_len, 2560))

    print(f"Building graph with batch_size={B}...")

    with Graph("test_transformer", input_types=(x_type, t_type, cap_type)) as graph:
        x, t, cap = graph.inputs
        out = model(x.tensor, t.tensor, cap.tensor)
        graph.output(out)

    print("Graph built successfully!")
    print(f"Output shape: {out.shape}")

    assert out.shape == (B, C, F_dim, H, W), f"Expected shape {(B, C, F_dim, H, W)}, got {out.shape}"
    print("Batching verification successful!")

if __name__ == "__main__":
    try:
        test_transformer_batching()
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
