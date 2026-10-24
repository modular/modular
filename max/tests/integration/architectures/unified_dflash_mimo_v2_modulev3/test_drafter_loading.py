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
"""The ModuleV3 DFlash drafter's checkpoint contract, on a CPU mesh.

The V3 drafter keeps the graph-API drafter's checkpoint names and files, so
its weights load through the shared adapter. What has to be pinned is the
part ``Module.compile`` makes silent: it drops a weights entry no parameter
matches, so the adapter's output names and the drafter module's parameters
must be asserted equal in both directions, plus the strict load and the
RoPE table's dtype (a review finding on the V2 drafter, where casting the
table to bf16 pre-rounded cos/sin and shifted every cached key).
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor, default_device, default_dtype
from max.graph import DeviceRef
from max.pipelines.architectures.dflash_mimo_v2 import DFlashMiMoV2Config
from max.pipelines.architectures.dflash_mimo_v2_modulev3 import DFlashMiMoV2
from max.pipelines.architectures.dflash_mimo_v2_modulev3.context_writer import (
    DFlashContextWriter,
)
from mimo_dflash_v3_fixtures import (
    BLOCK,
    adapted_draft,
    draft_kv_params,
    mask_embedding,
    tiny_draft_config,
)


def _devices() -> list[DeviceRef]:
    return [DeviceRef.CPU() for _ in range(2)]


def _config() -> DFlashMiMoV2Config:
    devices = _devices()
    return tiny_draft_config(devices, draft_kv_params(devices))


@pytest.fixture(scope="module")
def drafter() -> DFlashMiMoV2:
    """The drafter module traced on a 2-CPU mesh, lazily."""
    config = _config()
    with F.lazy(), default_dtype(DType.bfloat16):
        with default_device(config.devices[0].to_device()):
            return DFlashMiMoV2(config, mask_embedding())


def test_adapter_names_match_module_parameters(
    drafter: DFlashMiMoV2,
) -> None:
    """``Module.compile`` silently ignores a checkpoint entry no parameter
    matches and initializes the parameter to its default, so the name
    contract is the real gate. Both directions: a missing entry leaves a
    parameter at its default, and an extra one is dropped without a word."""
    config = _config()
    adapted = adapted_draft(config, mask_embedding())
    names = {name for name, _ in drafter.parameters}
    assert names == set(adapted), (
        f"module lacks {sorted(set(adapted) - names)[:8]}; no parameter "
        f"matches {sorted(names - set(adapted))[:8]}"
    )
    drafter.load_state_dict(adapted, strict=True)


def test_strict_load_pins_shapes_and_dtypes(
    drafter: DFlashMiMoV2,
) -> None:
    """The strict load rejects a missing or extra tensor and checks every
    shape and backing dtype, so a drafter checkpoint whose geometry drifted
    from its config fails at load, not mid-trace."""
    config = _config()
    adapted = adapted_draft(config, mask_embedding())
    drafter.load_state_dict(adapted, strict=True)
    # Every parameter ended up holding the checkpoint's buffer, not a
    # default-initialized one.
    loaded = dict(drafter.parameters)
    assert set(loaded) == set(adapted)
    for name, data in adapted.items():
        assert loaded[name].dtype == data.dtype, name


def test_the_rope_table_stays_float32() -> None:
    """The drafter's RoPE table must come back in float32, not the residual
    stream's bf16: ``fused_qk_ragged_rope`` rounds once per rotation, and a
    table pre-rounded to bf16 shifts every cached key the later block
    queries read. The V2 drafter's contract (``_freqs_cis`` is a device move
    only) is what this pins; a cast here was the review finding.
    """
    config = _config()
    with F.lazy(), default_dtype(DType.bfloat16):
        with default_device(config.devices[0].to_device()):
            writer = DFlashContextWriter(config)
            # The table is reached through the drafter's own freqs helper on
            # an arbitrary lazy tensor's mesh; any [rows, hidden] input does.
            h = Tensor.zeros([BLOCK, config.hidden_size], dtype=DType.bfloat16)
            freqs = writer._freqs_cis(h)
            assert freqs.dtype == DType.float32
