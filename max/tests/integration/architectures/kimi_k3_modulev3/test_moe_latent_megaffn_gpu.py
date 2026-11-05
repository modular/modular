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
"""``KimiK3LatentMoE``'s W4A8 MegaFFN against its chain, through the layer.

The same MXFP4 layer is built twice, with the MegaFFN (the default) and
with ``amd_mega_ffn=False``, at the expert shape of one of 8 tensor-parallel GPUs (latent
3584, expert width 3072 / 8 = 384, top 16). Both run the class's own
``forward``, the MegaFFN on the weights its init preshuffles from the
per-expert host buffers; the outputs must match byte for byte, inside the
MegaFFN's decode band and above it, where it runs its prefill kernel.
"""

from __future__ import annotations

import numpy as np
import pytest
from _moe_test_utils import assign_experts
from max.driver import Accelerator, Buffer, Device, accelerator_api
from max.dtype import DType
from max.nn.legacy.quant_config import QuantConfig
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    KimiK3LatentMoE,
)
from max.tensor import Tensor, default_device

_HIDDEN = 512
_LATENT = 3584
_MOE_INTERMEDIATE = 384
_NUM_EXPERTS = 64
_TOP_K = 16


def _scales(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return (127 - 7 + rng.integers(0, 3, size=shape)).astype(np.uint8)


def _tensor(array: np.ndarray, dtype: DType, device: Device) -> Tensor:
    buf = (
        Buffer.from_dlpack(array)
        .to(device)
        .view(dtype=dtype, shape=tuple(int(d) for d in array.shape))
    )
    return Tensor(storage=buf)


def _build(mega: bool) -> KimiK3LatentMoE:
    layer = KimiK3LatentMoE(
        hidden_size=_HIDDEN,
        latent_size=_LATENT,
        moe_intermediate_size=_MOE_INTERMEDIATE,
        shared_intermediate_size=2 * _MOE_INTERMEDIATE,
        num_experts=_NUM_EXPERTS,
        num_experts_per_token=_TOP_K,
        beta=4.0,
        linear_beta=25.0,
        rms_norm_eps=1e-5,
        routed_scaling_factor=2.5,
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        dtype=DType.bfloat16,
        quant_config=QuantConfig.mxfp4(mlp_quantized_layers=[0]),
        amd_w4a8=True,
        amd_mega_ffn=mega,
    )
    assert layer.amd_mega_ffn == mega
    rng = np.random.default_rng(5)

    def normal(*shape: int, scale: float) -> Tensor:
        array = (rng.standard_normal(shape) * scale).astype(np.float32)
        return Tensor(array, dtype=DType.float32)

    layer.gate.gate_score.weight = normal(_NUM_EXPERTS, _HIDDEN, scale=0.1)
    layer.gate.e_score_correction_bias = normal(_NUM_EXPERTS, scale=0.1)
    layer.routed_expert_down_proj.weight = normal(
        _LATENT, _HIDDEN, scale=_HIDDEN**-0.5
    ).cast(DType.bfloat16)
    layer.routed_expert_up_proj.weight = normal(
        _HIDDEN, _LATENT, scale=_LATENT**-0.5
    ).cast(DType.bfloat16)
    layer.routed_expert_norm.weight = Tensor(
        np.ones(_LATENT, dtype=np.float32)
    ).cast(DType.bfloat16)
    layer.shared_experts.gate_up_proj.weight = normal(
        4 * _MOE_INTERMEDIATE, _HIDDEN, scale=_HIDDEN**-0.5
    ).cast(DType.bfloat16)
    layer.shared_experts.down_proj.weight = normal(
        _HIDDEN, 2 * _MOE_INTERMEDIATE, scale=(2 * _MOE_INTERMEDIATE) ** -0.5
    ).cast(DType.bfloat16)
    n1 = 2 * _MOE_INTERMEDIATE
    assign_experts(
        layer,
        rng.integers(0, 256, (_NUM_EXPERTS, n1, _LATENT // 2), np.uint8),
        rng.integers(
            0, 256, (_NUM_EXPERTS, _LATENT, _MOE_INTERMEDIATE // 2), np.uint8
        ),
        lambda a, device: _tensor(a, DType.uint8, device),
    )
    assign_experts(
        layer,
        _scales(rng, _NUM_EXPERTS, n1, _LATENT // 32),
        _scales(rng, _NUM_EXPERTS, _LATENT, _MOE_INTERMEDIATE // 32),
        lambda a, device: _tensor(a, DType.float8_e8m0fnu, device),
        leaf="weight_scale",
    )
    return layer


@pytest.mark.skipif(
    accelerator_api() != "hip", reason="the W4A8 MegaFFN is CDNA4-only"
)
@pytest.mark.compile_on_gpu(
    "mo.moe.single.group.router is staged as an undeclared custom op the"
    " interpreter can't bind"
)
@pytest.mark.parametrize("tokens", [1, 9, 64, 160, 1024])
def test_megaffn_matches_chain(tokens: int) -> None:
    """160 tokens is 2560 routed rows, above the MegaFFN's 2304-row band."""
    x = np.random.default_rng(tokens).standard_normal((tokens, _HIDDEN))
    device = Accelerator()
    with default_device(device):
        x_bf16 = Tensor(x.astype(np.float32)).cast(DType.bfloat16)
        outs = [
            _build(mega)(x_bf16).cast(DType.float32).to_numpy()
            for mega in (False, True)
        ]
    assert np.all(np.isfinite(outs[0]))
    assert np.count_nonzero(outs[0]) > outs[0].size // 2
    bad = int(np.count_nonzero(outs[0] != outs[1]))
    assert bad == 0, f"{bad} of {outs[0].size} outputs differ from the chain"
