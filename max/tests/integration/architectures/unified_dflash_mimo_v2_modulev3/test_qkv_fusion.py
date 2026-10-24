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
"""The target's fused QKV preparation reaches the ModuleV3 spec graph.

The fused graph is staged on virtual B200s at the published checkpoint's
per-device head layout and never compiled, so the test needs no GPU. The
drafter keeps its own preparation path.
"""

from __future__ import annotations

import pytest
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device, default_dtype
from max.graph import DeviceRef
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.unified_dflash_mimo_v2 import (
    GRAPH_NAME,
    UnifiedDflashMiMoV2ModuleV3,
)
from mimo_dflash_v3_fixtures import (
    _FakeAttrConfig,
    build_spec,
    fused_weights,
    mask_embedding,
)

FUSED_QKV = "mo.chunked_qkv_rope_store.ragged.paged"
SLIDING_LAYERS = sum(_FakeAttrConfig.hybrid_layer_pattern)
FULL_LAYERS = _FakeAttrConfig.num_hidden_layers - SLIDING_LAYERS


@pytest.mark.parametrize("n_devices", [2, 4])
@pytest.mark.parametrize(
    ("sliding", "full"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["off", "sliding", "full", "both"],
)
def test_qkv_fusion_reaches_the_spec_graph(
    n_devices: int, sliding: bool, full: bool
) -> None:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(n_devices)
    spec, config, draft, kv_tree = build_spec(
        [DeviceRef.GPU(i) for i in range(n_devices)]
    )
    config.fused_sliding_qkv = sliding
    config.fused_full_qkv = full
    mesh = DeviceMesh(
        tuple(Accelerator(i) for i in range(n_devices)), (n_devices,), ("tp",)
    )
    with default_device(mesh), default_dtype(DType.bfloat16):
        with F.lazy():
            module = UnifiedDflashMiMoV2ModuleV3(
                spec, mask_embedding(), kv_params=kv_tree
            )
        with F.lazy():
            staged, _registry = module.stage(
                *module.input_types(),
                weights=fused_weights(config, draft),
                name=GRAPH_NAME,
            )
            text = str(staged)
    assert text.count(FUSED_QKV) == n_devices * (
        SLIDING_LAYERS * sliding + FULL_LAYERS * full
    )
