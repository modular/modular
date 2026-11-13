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
"""The Mistral Large 4 vision graph builds, on the ragged attention ABI.

No GPU is involved: the graph is traced on a virtual B200 and its MLIR is
checked, the way ``test_dspark_paged_capture_safe`` pins the drafter's graph.
What is pinned here:

* the graph builds at the production geometry (30 layers, hidden 2048,
  head_dim 128, patch 588, text width 6144) -- the widths the kernels
  dispatch on, per the fixture-ladder rule that counts may shrink but
  widths may not;
* every block's attention lowers to the ragged flash kernel
  (``attention.mha.ragged.no_cache``), never to a dense-mask attention;
* the input ABI is the five ragged inputs, with no rank-4 tensor: the
  Small 4 ABI passed a ``[1, 1, P, P]`` f32 mask, which for the largest
  image is a 586 MB host-to-device copy per forward;
* the weights bind under the checkpoint's consolidated names, so the
  vision contract needs no renames beyond the conv patch embed;
* ``pack_images`` emits the buffers that ABI declares, for a multi-image
  batch, without a GPU.

The GPU parity gate against the S1 HF goldens is
``test_vision_parity.py``, manual like the text parity tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from max import functional as F
from max.driver import (
    CPU,
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.mistral_large4_modulev3.vision import (
    MistralLarge4Vision,
    MistralLarge4VisionConfig,
    vision_input_types,
)
from max.pipelines.architectures.mistral_large4_modulev3.vision_inputs import (
    pack_images,
)
from max.pipelines.context import ImageMetadata
from max.tensor import default_device, default_dtype
from transformers import PretrainedConfig

_TESTDATA = Path(__file__).parent / "testdata"


def _read_headers(
    path: Path,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Reads a headers fixture as ``name -> (dtype, shape)``."""
    raw = json.loads(path.read_text())
    return {
        name: (entry["dtype"], tuple(entry["shape"]))
        for name, entry in raw.items()
    }


# The checkpoint's vision geometry, which the committed headers fixture also
# pins (`vision_encoder.ln_pre.weight` is [2048], the merger input is
# 8192 = 4 x 2048); see the cross-check test at the bottom.
CONFIG = MistralLarge4VisionConfig(
    dtype=DType.bfloat16,
    num_channels=3,
    patch_size=14,
    image_size=1540,
    vision_hidden_size=2048,
    vision_num_attention_heads=16,
    vision_head_dim=128,
    vision_rope_theta=10000.0,
    vision_num_hidden_layers=30,
    vision_intermediate_size=6144,
    spatial_merge_size=2,
    hidden_size=6144,
    rms_norm_eps=1e-5,
)


def test_the_test_config_matches_the_committed_fixture_headers() -> None:
    """The traced geometry is the checkpoint's, not a hand-typed stand-in."""
    headers = _read_headers(_TESTDATA / "ml4_nvfp4_headers.json")
    ln_pre = headers["vision_encoder.ln_pre.weight"]
    assert ln_pre == ("BF16", (CONFIG.vision_hidden_size,))
    patch_conv = headers["vision_encoder.patch_conv.weight"]
    assert patch_conv == (
        "BF16",
        (
            CONFIG.vision_hidden_size,
            CONFIG.num_channels,
            CONFIG.patch_size,
            CONFIG.patch_size,
        ),
    )
    merging = headers["patch_merger.merging_layer.weight"]
    assert merging == (
        "BF16",
        (CONFIG.vision_hidden_size, CONFIG.vision_hidden_size * 4),
    )
    w_in = headers["vision_language_adapter.w_in.weight"]
    assert w_in == ("BF16", (CONFIG.hidden_size, CONFIG.vision_hidden_size))
    # The committed headers fixture carries the tower first block in full;
    # the layer count comes from the checkpoint config, which the parity
    # gate runs against.
    block = sorted(
        name
        for name in headers
        if name.startswith("vision_encoder.transformer.layers.0.")
    )
    assert len(block) == 9, "four attention, three MLP and two norm weights"


def test_from_hf_config_reads_the_generated_config() -> None:
    """``from_hf_config`` reads the shim's parsed ``config.json`` blocks."""
    vision = PretrainedConfig(
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=30,
        num_attention_heads=16,
        num_channels=3,
        patch_size=14,
        image_size=1540,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"},
    )
    text = PretrainedConfig(hidden_size=6144, rms_norm_eps=1e-5)
    hf_config = PretrainedConfig(
        model_type="mistral_large4",
        text_config=text,
        vision_config=vision,
        spatial_merge_size=2,
    )
    config = MistralLarge4VisionConfig.from_hf_config(hf_config)
    assert config == CONFIG
    assert config.patch_dim == 588
    assert config.max_patches_per_side == 110


def _trace() -> str:
    """Traces the production vision graph on one virtual B200."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)
    try:
        with (
            default_device(Accelerator(0)),
            default_dtype(CONFIG.dtype),
            F.lazy(),
        ):
            vision_nn = MistralLarge4Vision(CONFIG)
            return str(
                vision_nn.trace(*vision_input_types(CONFIG, DeviceRef.GPU(0)))
            )
    finally:
        set_virtual_device_count(0)


@pytest.fixture(scope="module")
def trace() -> str:
    return _trace()


def test_every_blocks_attention_lowers_to_the_ragged_flash_kernel(
    trace: str,
) -> None:
    """30 blocks, one ragged flash kernel each, and no dense-mask fallback."""
    assert trace.count("attention.mha.ragged.no_cache") == (
        CONFIG.vision_num_hidden_layers
    )
    assert "attention.mha.ragged.no_cache.segment" not in trace


def test_the_graph_binds_the_checkpoints_consolidated_names(
    trace: str,
) -> None:
    """The vision weights keep their names: no `.self_attn.` rewrites."""
    for name in (
        "vision_encoder.patch_conv",
        "vision_encoder.ln_pre.weight",
        "vision_encoder.transformer.layers.0.attention.wq.weight",
        "vision_encoder.transformer.layers.29.attention.wo.weight",
        "vision_encoder.transformer.layers.29.feed_forward.w2.weight",
        "pre_mm_projector_norm.weight",
        "patch_merger.merging_layer.weight",
        "vision_language_adapter.w_in.weight",
        "vision_language_adapter.w_out.weight",
    ):
        assert name in trace, name
    assert ".self_attn." not in trace
    assert "language_model." not in trace


def test_the_input_abi_stays_ragged() -> None:
    """Five inputs, none rank 4: no dense ``[1, 1, P, P]`` mask ever returns.

    The mask input was Small 4's ABI; the 586 MB worst case is why the
    ragged kernel reads image boundaries from ``cu_seqlens`` instead.
    """
    input_types = vision_input_types(CONFIG, DeviceRef.GPU(0))
    assert len(input_types) == 5
    patches, position_ids, cu_seqlens, merge_indices, max_seq_len = input_types
    assert patches.dtype == DType.float32
    assert list(patches.shape) == ["total_patches", 588]
    assert position_ids.dtype == DType.int64
    assert cu_seqlens.dtype == DType.uint32
    assert list(cu_seqlens.shape) == ["num_images_plus_1"]
    assert merge_indices.dtype == DType.int32
    assert list(merge_indices.shape) == ["merged_tokens", 4]
    assert max_seq_len.dtype == DType.uint32
    assert max_seq_len.device == DeviceRef.CPU()
    for input_type in input_types:
        assert len(input_type.shape) < 4


def test_pack_images_emits_the_ragged_abi() -> None:
    """Two images, different grids: the packer feeds the traced ABI.

    CPU only; the packing is host work, and the buffers' shapes, dtypes and
    devices are what the graph's input types declare.
    """
    height, width = 84, 84
    other = 56
    patch = CONFIG.patch_size
    images = [
        ImageMetadata(
            start_idx=0,
            end_idx=(height // patch // 2) * (width // patch // 2),
            pixel_values=np.zeros((3, height, width), dtype=np.float32),
        ),
        ImageMetadata(
            start_idx=0,
            end_idx=(other // patch // 2) ** 2,
            pixel_values=np.zeros((3, other, other), dtype=np.float32),
        ),
    ]
    packed = pack_images(images, CONFIG, CPU())
    assert len(packed) == 5
    patches, position_ids, cu_seqlens, merge_indices, max_seq_len = packed
    total = (height // patch) ** 2 + (other // patch) ** 2
    merged = ((height // patch) // 2) ** 2 + ((other // patch) // 2) ** 2
    assert patches.shape == (total, CONFIG.patch_dim)
    assert patches.dtype == DType.float32
    assert position_ids.shape == (total,)
    assert cu_seqlens.shape == (3,)
    assert np.array_equal(
        cu_seqlens.to_numpy(), np.array([0, 36, 52], np.uint32)
    )
    assert merge_indices.shape == (merged, 4)
    assert merge_indices.to_numpy().dtype == np.int32
    assert max_seq_len.shape == ()
    assert max_seq_len.to_numpy() == np.uint32(36)
    # The first square of the second image starts at its patch offset, nine
    # merged rows into the batch.
    assert merge_indices.to_numpy()[9][0] == 36
