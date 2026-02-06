# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Tests for InternVL vision embeddings multi-GPU sharding."""

import os

import pytest
import torch
from max.driver import accelerator_count
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.internvl.internvl import InternVisionEmbeddings
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import convert_hf_to_max_weights
from utils.weight_generator import get_weight_generator


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.INTERNVL_2B),
        pytest.param(
            ConfigNames.INTERNVL_8B,
            marks=[
                pytest.mark.skipif(
                    not os.environ.get("INTERNVL_8B_TESTS"),
                    reason="8B tests disabled (set INTERNVL_8B_TESTS env var to enable)",
                ),
            ],
        ),
        pytest.param(
            ConfigNames.INTERNVL_38B,
            marks=[
                pytest.mark.skipif(
                    not os.environ.get("INTERNVL_38B_TESTS"),
                    reason="38B tests disabled (set INTERNVL_38B_TESTS env var to enable)",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize("n_devices", [2, 4])
def test_vision_embeddings_multi_gpu_sharding(
    config_name: ConfigNames, n_devices: int
) -> None:
    """Test InternVisionEmbeddings sharding with multiple GPUs."""
    # Set seed for deterministic results
    torch.manual_seed(42)

    if n_devices > accelerator_count():
        pytest.skip(f"Not enough GPUs to run test with {n_devices} GPUs.")

    # Create test instance
    internvl_config = get_config_loader().create_internvl_config(config_name)
    embeddings_weights = get_weight_generator(
        config_name
    ).generate_vision_embeddings_weights()

    # Convert HuggingFace weights to MAX format using weight converter
    max_weights = convert_hf_to_max_weights(embeddings_weights)

    # Build graph to test sharding
    # Calculate expected patch dimensions
    patch_size = internvl_config.vision_config.patch_size
    height_patches = internvl_config.vision_config.image_size // patch_size
    width_patches = internvl_config.vision_config.image_size // patch_size
    channels = 3
    batch_size = 2  # Simulating 2 images

    input_type = TensorType(
        internvl_config.vision_config.dtype,
        shape=(
            batch_size,
            height_patches,
            width_patches,
            channels,
            patch_size,
            patch_size,
        ),
        device=DeviceRef.GPU(0),
    )

    with Graph(
        "test_sharding",
        input_types=(input_type,),
    ):
        # Create embeddings and set sharding strategy
        embeddings = InternVisionEmbeddings(internvl_config, DeviceRef.GPU(0))
        embeddings.load_state_dict(
            state_dict=max_weights,
        )

        # Set sharding strategy for replication across devices
        from max.graph import ShardingStrategy

        embeddings.sharding_strategy = ShardingStrategy.replicate(
            num_devices=n_devices
        )

        # Test that we can shard to each device
        devices = [DeviceRef.GPU(i) for i in range(n_devices)]
        shards = embeddings.shard(devices)
        for sharded in shards:
            # Verify sharded embeddings maintain configuration
            assert isinstance(sharded, InternVisionEmbeddings)
            assert (
                sharded.embed_dim == internvl_config.vision_config.hidden_size
            )
            assert (
                sharded.image_size == internvl_config.vision_config.image_size
            )
            assert (
                sharded.patch_size == internvl_config.vision_config.patch_size
            )

    # Test passes if sharding completes without errors
    print(f"Successfully tested sharding with {n_devices} devices")
