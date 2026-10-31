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
"""``act_quant`` on four real GPUs, against the same input on one.

``_scales_mapping`` claims the quantization kernel lays its scales out as
``[head_dim // block_size, M_padded]``, so a shard of ``M`` belongs on axis 1
rather than axis 0. ``rebind_mapping`` only asserts that; nothing checks it
against the kernel. If the claim is wrong, the slice and reshape downstream
read unrelated bytes and scores change with no placement error, so this
compares values rather than placements.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.functional import cast, transfer_to
from max.nn.legacy.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.layers.indexer import (
    act_quant,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh, Replicated, Sharded

_N = 4
_HEAD_DIM = 128
_ROWS = 32


def _quant_config() -> QuantConfig:
    return QuantConfig(
        format=QuantFormat.BLOCKSCALED_FP8,
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 128),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(128, 128),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
    )


@pytest.mark.skipif(accelerator_count() < _N, reason=f"needs {_N} GPUs")
def test_act_quant_sharded_rows_match_single_device() -> None:
    """Sharding the row axis must not change what ``act_quant`` returns."""
    cfg = _quant_config()
    x_np = (
        np.random.default_rng(0)
        .standard_normal((_ROWS, _HEAD_DIM))
        .astype(np.float32)
    )

    one = DeviceMesh(
        devices=(Accelerator(0),), mesh_shape=(1,), axis_names=("tp",)
    )
    many = DeviceMesh(
        devices=tuple(Accelerator(i) for i in range(_N)),
        mesh_shape=(_N,),
        axis_names=("tp",),
    )

    ref_q, ref_s = act_quant(
        transfer_to(Tensor(x_np), DeviceMapping(one, (Replicated(),))), cfg
    )
    got_q, got_s = act_quant(
        transfer_to(Tensor(x_np), DeviceMapping(many, (Sharded(0),))), cfg
    )

    # ``_scales_mapping`` moves the shard to axis 1 for the kernel's
    # ``[K_groups, M_padded]`` layout, but ``act_quant`` reshapes to
    # ``(*x_dims, 1)`` before returning, so the row extent lands back on
    # axis 0. Both outputs come back sharded on rows.
    assert got_q.placements == (Sharded(0),)
    assert got_s.placements == (Sharded(0),)

    # fp8 has no numpy dtype; widening to fp32 is exact for both formats.
    def _f32(t):  # noqa: ANN001, ANN202
        return cast(t, DType.float32).to_numpy()

    ref_q_np, got_q_np = _f32(ref_q), _f32(got_q)
    assert ref_q_np.shape == got_q_np.shape
    np.testing.assert_array_equal(
        got_q_np,
        ref_q_np,
        err_msg="quantized activations differ once the rows are sharded",
    )

    ref_s_np, got_s_np = _f32(ref_s), _f32(got_s)
    assert ref_s_np.shape == got_s_np.shape
    np.testing.assert_array_equal(
        got_s_np,
        ref_s_np,
        err_msg=(
            "scales differ once the rows are sharded: the layout "
            "_scales_mapping assumes does not match the kernel's"
        ),
    )
