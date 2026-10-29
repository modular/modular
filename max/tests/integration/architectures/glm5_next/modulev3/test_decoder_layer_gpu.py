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
"""The GLM-5.3-Flash ModuleV3 decoder layer folds its input norms into mHC.

Where ``mo.mhc.pre`` runs, both sites of a layer hand their sublayer's input
RMSNorm to the fused op, so the graph has no separate ``rms_norm``; elsewhere
the layer builds the op chain. The sublayers are pass-throughs, so the layer's
output is the two sites and their write-backs alone.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext

import pytest
import torch
from max.driver import CPU, Accelerator, accelerator_api
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Module
from max.nn import hyper_connection as hyper_connection_module
from max.nn.legacy.kernels import mhc_pre_supports
from max.nn.legacy.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.glm5_next_modulev3.layers.decoder import (
    Glm5NextDecoderLayer,
)
from max.pipelines.architectures.glm5_next_modulev3.model_config import (
    Glm5NextConfig,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    TensorLayout,
)

_HC_MULT = 4
_HIDDEN = 4096
_MIX_WIDTH = 2 * _HC_MULT + _HC_MULT**2
_ROWS = [1, 8, 128, 1024]
# The output streams are bf16, and each site's normalized row may round
# differently fused and unfused: 2**-6 of the value plus 1% of the RMS.
_OUTPUT_RTOL = 2.0**-6
_OUTPUT_ATOL_FRAC = 1e-2
_RMS_NORM_OP = re.compile(r"\bmo\.[\w.]*rms_norm\b")

_Layer = CompiledCallable[[Tensor], Tensor]


class _Passthrough(Module[..., Tensor]):
    def forward(self, x: Tensor, *args: object) -> Tensor:
        return x


class _Layer1(Module[[Tensor], Tensor]):
    """One decoder layer with pass-through attention and feed-forward."""

    def __init__(self, config: Glm5NextConfig) -> None:
        self.layer = Glm5NextDecoderLayer(
            config, self_attn=_Passthrough(), mlp=_Passthrough()
        )

    def forward(self, streams: Tensor) -> Tensor:
        return self.layer(streams, (), None)


def _config(fn_dtype: DType) -> Glm5NextConfig:
    devices = [DeviceRef.GPU()]
    leaf = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        page_size=128,
        devices=devices,
        num_q_heads=8,
    )
    return Glm5NextConfig(
        dtype=DType.bfloat16,
        kv_params=MultiKVCacheParams.from_params(
            {"mla": leaf, "indexer": leaf}
        ),
        devices=devices,
        max_seq_len=1024,
        hidden_size=_HIDDEN,
        num_hidden_layers=1,
        qk_rope_head_dim=0,
        rms_norm_eps=1e-5,
        hc_fn_dtype=fn_dtype,
    )


def _mesh() -> DeviceMesh:
    return DeviceMesh((Accelerator(),), (1,), ("tp",))


def _input_type(mesh: DeviceMesh) -> TensorLayout:
    return TensorLayout(
        DType.bfloat16,
        ["total_seq_len", _HC_MULT, _HIDDEN],
        DeviceMapping(mesh, (Replicated(),)),
    )


def _layer(fn_dtype: DType, mesh: DeviceMesh) -> _Layer1:
    with default_device(mesh), default_dtype(DType.bfloat16):
        return _Layer1(_config(fn_dtype))


def _weights(fn_dtype: DType) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(2)
    fan_in = _HC_MULT * _HIDDEN
    torch_fn_dtype = (
        torch.bfloat16 if fn_dtype == DType.bfloat16 else torch.float32
    )
    weights = {}
    for site in ("hc_attn", "hc_ffn"):
        hc_fn = torch.randn(_MIX_WIDTH, fan_in, generator=gen)
        weights[f"layer.{site}.hc_fn"] = (hc_fn / math.sqrt(fan_in)).to(
            torch_fn_dtype
        )
        weights[f"layer.{site}.hc_base"] = 0.85 * (
            2 * torch.rand(_MIX_WIDTH, generator=gen) - 1
        )
        weights[f"layer.{site}.hc_scale"] = torch.tensor([0.5, 0.5, 1.0])
    for norm in ("input_layernorm", "post_attention_layernorm"):
        gamma = 1 + 0.085 * (2 * torch.rand(_HIDDEN, generator=gen) - 1)
        weights[f"layer.{norm}.weight"] = gamma.to(torch.bfloat16)
    return weights


def _uses_mhc_pre(fn_dtype: DType) -> bool:
    return accelerator_api() == "hip" and mhc_pre_supports(
        _HC_MULT, _HIDDEN, DType.bfloat16, fn_dtype, Accelerator()
    )


@contextmanager
def _op_chain_only() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            hyper_connection_module, "mhc_pre_supports", lambda *args: False
        )
        yield


@pytest.fixture(scope="module")
def compiled() -> Callable[[DType, bool], _Layer]:
    """Compiles each projection dtype once, by default and as the op chain."""
    mesh = _mesh()
    cache: dict[tuple[DType, bool], _Layer] = {}

    def get(fn_dtype: DType, op_chain: bool) -> _Layer:
        if (fn_dtype, op_chain) not in cache:
            layer = _layer(fn_dtype, mesh)
            with _op_chain_only() if op_chain else nullcontext():
                cache[fn_dtype, op_chain] = layer.compile(
                    _input_type(mesh), weights=_weights(fn_dtype)
                )
        return cache[fn_dtype, op_chain]

    return get


@pytest.mark.parametrize("fn_dtype", [DType.float32, DType.bfloat16])
def test_input_norms_fold_into_mhc_pre(fn_dtype: DType) -> None:
    mesh = _mesh()
    layer = _layer(fn_dtype, mesh)
    graph = str(layer.trace(_input_type(mesh)))
    uses_kernel = _uses_mhc_pre(fn_dtype)
    assert graph.count("mo.mhc.pre") == (2 if uses_kernel else 0)
    assert (_RMS_NORM_OP.search(graph) is None) == uses_kernel

    with _op_chain_only():
        chain = str(layer.trace(_input_type(mesh)))
    assert "mo.mhc.pre" not in chain
    assert len(_RMS_NORM_OP.findall(chain)) == 2


@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize("fn_dtype", [DType.float32, DType.bfloat16])
def test_fused_layer_matches_op_chain(
    compiled: Callable[[DType, bool], _Layer], fn_dtype: DType, rows: int
) -> None:
    gen = torch.Generator().manual_seed(rows)
    streams = torch.randn(rows, _HC_MULT, _HIDDEN, generator=gen).to(
        torch.bfloat16
    )
    device = Accelerator()

    def run(op_chain: bool) -> torch.Tensor:
        out = compiled(fn_dtype, op_chain)(Tensor(streams, device=device))
        return torch.from_dlpack(out.to(CPU())).double()

    chain = run(op_chain=True)
    assert tuple(chain.shape) == (rows, _HC_MULT, _HIDDEN)
    assert torch.isfinite(chain).all()
    if not _uses_mhc_pre(fn_dtype):
        return

    fused = run(op_chain=False)
    atol = _OUTPUT_ATOL_FRAC * chain.square().mean().sqrt().item()
    diff = (fused - chain).abs()
    excess = diff - _OUTPUT_RTOL * chain.abs()
    worst = excess.max().item() / atol
    print(
        f"{fn_dtype} T={rows}: max abs {diff.max().item():.3e}"
        f" excess/atol {worst:.3e}"
    )
    assert torch.isfinite(fused).all()
    assert worst <= 1.0
