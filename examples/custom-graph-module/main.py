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

# DOC: max/tutorials/build-an-mlp-block.mdx

from max.graph import DeviceRef, ops
from mlp import MLPBlock

if __name__ == "__main__":
    print("--- Simple MLP Block ---")

    # Define device for all examples (not used with simplified MLPBlock)
    target_device = DeviceRef.CPU()
    # Uncomment to run on GPU
    # target_device = DeviceRef.GPU()

    # 1. Simple MLP (no hidden layers)
    simple_mlp = MLPBlock(
        in_features=10,
        out_features=20,
        hidden_features=[],
        activation=ops.relu,
    )
    print(simple_mlp)
    print("-" * 30)

    # 2. MLP with one hidden layer
    print("--- MLP Block (1 Hidden Layer) ---")
    mlp_one_hidden = MLPBlock(
        in_features=10,
        out_features=5,
        hidden_features=[32],
        activation=ops.relu,
    )
    print(mlp_one_hidden)
    print("-" * 30)

    # 3. Deeper MLP with multiple hidden layers and GELU
    print("--- Deeper MLP Block (3 Hidden Layers, GELU) ---")
    deep_mlp = MLPBlock(
        in_features=64,
        out_features=10,
        hidden_features=[128, 64, 32],
        activation=ops.gelu,
    )
    print(deep_mlp)
    print("-" * 30)
