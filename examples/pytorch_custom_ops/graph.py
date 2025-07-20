# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import max
import torch
from max.torch import graph_op


@graph_op
def max_matmul(a: max.graph.TensorValue, b: max.graph.TensorValue):
    """Custom PyTorch operation built using an internal MAX graph."""
    return a @ b  # Same as ops.matmul(a, b)


@torch.compile
def matmul_max(a: torch.Tensor, b: torch.Tensor):
    """Wrapper function that calls the MAX matmul operation."""
    # Create output tensor with appropriate shape
    output = a.new_empty(a.shape[0], b.shape[1])
    max_matmul(output, a, b)  # Call as destination-passing style
    return output


if __name__ == "__main__":
    # Test on both CPU and GPU if available
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        print(f"\n{'=' * 50}")
        print(f"Testing on device: {device}")
        print("=" * 50)

        # Create random input tensors
        M, K, N = 128, 256, 512
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)

        # Compute matmul using MAX
        result_max = matmul_max(a, b)

        # Compute matmul using PyTorch for comparison
        result_torch = torch.matmul(a, b)

        # Verify the results match
        if torch.allclose(result_max, result_torch, rtol=1e-1, atol=1e-1):
            print("✓ MAX matmul matches PyTorch matmul!")
            print(f"  Input shapes: A={a.shape}, B={b.shape}")
            print(f"  Output shape: {result_max.shape}")
        else:
            max_diff = torch.max(torch.abs(result_max - result_torch)).item()
            diff = torch.abs(result_max - result_torch)
            print("✗ Results do not match!")
            print(f"  Max absolute difference: {max_diff:.2e}")
            print(f"  Mean absolute difference: {diff.mean().item():.2e}")
            print(f"  Std of differences: {diff.std().item():.2e}")
