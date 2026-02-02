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

"""Tests for InternVL vision embeddings with square images."""

import os

import pytest
import torch
from max.driver import Accelerator
from max.dtype import DType
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.vision_embeddings_helpers import (
    generate_max_outputs,
    generate_torch_outputs,
)
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
@pytest.mark.parametrize(
    "target_size",
    [
        224,  # Downscale from 448 (16 patches)
        448,  # Same size (32 patches)
        672,  # 1.5x upscale from 448 (48 patches)
    ],
)
def test_vision_embeddings_square(
    config_name: ConfigNames, target_size: int
) -> None:
    """Test position embedding interpolation for different resolutions.
    Note: Image dimensions must be divisible by patch_size (14) for the
    reshape operations to work correctly.
    """
    torch.manual_seed(42)

    # Load HuggingFace config and generate weights
    hf_config = get_config_loader().load_hf_vision_config(config_name)
    internvl_config = get_config_loader().create_internvl_config(config_name)
    embeddings_weights = get_weight_generator(
        config_name
    ).generate_vision_embeddings_weights()

    # Create test inputs with target size
    batch_size = 1
    pixel_values = torch.randn(
        batch_size, 3, target_size, target_size, dtype=torch.bfloat16
    ).to("cuda")

    # Generate reference output
    torch_output = generate_torch_outputs(
        pixel_values, hf_config, embeddings_weights
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        pixel_values,
        internvl_config,
        embeddings_weights,
        DType.bfloat16,
        Accelerator(),
    )

    # Verify output shape
    patch_size = internvl_config.vision_config.patch_size
    expected_num_patches = (target_size // patch_size) ** 2
    expected_shape = (
        batch_size,
        expected_num_patches + 1,
        internvl_config.vision_config.hidden_size,
    )
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Compare outputs
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
        message="Vision embeddings outputs do not match",
    )
