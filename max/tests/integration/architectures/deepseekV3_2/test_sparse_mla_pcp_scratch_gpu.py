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
"""Prefill context parallelism against DCP over sparse MLA layers, TP2 x DCP2.

The step's plan builds one scratch cache pair that every layer's PCP path
refills with its own prefix: each layer's attention must read the scratch
after its own prefix unpack and before the next layer's. The layers run as
GLM-5.x schedules them, a full layer (its own indexer), a shared one (the
full layer's selection) and a full one, each kind one subgraph as in the
model. A partly cached batch of two requests is prefilled once on the DCP
path, then its next chunk runs through the layers on each path, and each
layer's PCP output must match its DCP output.

The latent cache is BF16: under DCP every step runs the sparse decode
kernel and PCP the sparse prefill kernel, and over an FP8 cache the two
differ by FP8 rounding, about as much as a misread prefix would at these
sizes. Over BF16 they agree to accumulation order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from max import functional as F
from max import tree
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, TensorType
from max.nn import Module, as_subgraph
from max.nn.common_layers.kv_cache import PagedCacheValues, kv_input_values
from max.nn.common_layers.mesh_axis import TP
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheQuantizationConfig,
    MLAKVCacheParams,
    MultiKVCacheParams,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.sequential import ModuleList
from max.pipelines.architectures.deepseekV3_2_modulev3.layers.prefill_context_parallel import (
    PrefillContextParallelStep,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.layers.sparse_mla import (
    QuantizedSparseLatentAttentionWithRope,
    Selection,
    tensor_parallel_sparse_latent_attention_with_rope,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMapping, DeviceMesh, Partial, Replicated
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack

_N = 2
# The model's indexer schedule: shared layers reuse the last full layer's
# selection and own no indexer cache slot.
_LAYER_TYPES = ("full", "shared", "full")
_NUM_LAYERS = len(_LAYER_TYPES)
_NUM_FULL = _LAYER_TYPES.count("full")
_N_HEADS = 16
_HIDDEN = 256
_Q_LORA_RANK = 128
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 192
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 256
_PAGE_SIZE = 128
_TOTAL_NUM_PAGES = 64
_MAX_SEQ_LEN = 4096
# Past every context here, so each query selects its whole causal context
# and the comparison does not hinge on top-k ties.
_TOPK = 2048

# Each request's cached prefix and the chunk the step under test prefills.
# Neither is page aligned, and the chunk starts mid-page.
_PREFIX = (300, 517)
_CHUNK = (260, 141)


def _fp8_config() -> QuantConfig:
    return QuantConfig(
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
        attn_quantized_layers=set(range(_NUM_LAYERS)),
        format=QuantFormat.BLOCKSCALED_FP8,
    )


def _kv_params(
    devices: list[DeviceRef],
) -> tuple[MultiKVCacheParams, MLAKVCacheParams]:
    """Returns the latent and indexer caches, and the latent cache alone."""
    mla = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=_KV_LORA_RANK + _QK_ROPE_HEAD_DIM,
        num_layers=_NUM_LAYERS,
        devices=devices,
        num_q_heads=_N_HEADS,
        page_size=_PAGE_SIZE,
    )
    indexer = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=128,
        num_layers=_NUM_FULL,
        devices=devices,
        num_q_heads=_N_HEADS,
        page_size=_PAGE_SIZE,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=128
        ),
    )
    return (
        MultiKVCacheParams.from_params({"mla": mla, "indexer": indexer}),
        mla,
    )


def _make_layer(
    layer_idx: int, mla: MLAKVCacheParams
) -> QuantizedSparseLatentAttentionWithRope:
    # BF16 MLA weights and an FP8 indexer, as PCP serves GLM-5.x.
    return QuantizedSparseLatentAttentionWithRope(
        num_attention_heads=_N_HEADS,
        num_key_value_heads=1,
        hidden_size=_HIDDEN,
        kv_params=mla,
        layer_idx=layer_idx,
        q_lora_rank=_Q_LORA_RANK,
        kv_lora_rank=_KV_LORA_RANK,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        graph_mode="auto",
        quant_config=None,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=_TOPK,
        indexer_quant_config=_fp8_config(),
        skip_topk=_LAYER_TYPES[layer_idx] == "shared",
        dcp_size=_N,
    )


class _Layers(Module[..., tuple[Tensor, ...]]):
    """The sparse MLA layers of one step, each reading the last's output."""

    def __init__(
        self,
        layers: list[QuantizedSparseLatentAttentionWithRope],
        mesh: DeviceMesh,
        kv_params: MultiKVCacheParams,
    ) -> None:
        super().__init__()
        self.layers = ModuleList[QuantizedSparseLatentAttentionWithRope](layers)
        self.mesh = mesh
        self.kv_params = kv_params

    def forward(
        self,
        x: Tensor,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
        active: Tensor,
        batch_context_length: Tensor,
        kv: Any,
    ) -> tuple[Tensor, ...]:
        mesh = self.mesh
        replicated = DeviceMapping(mesh, (Replicated(),))
        h = F.distributed_broadcast(x, mesh)
        offsets = F.distributed_broadcast(input_row_offsets, mesh)
        freqs = F.distributed_broadcast(freqs_cis, mesh)

        kv_tree = self.kv_params.unflatten_kv_inputs(iter(kv_input_values(kv)))
        assert isinstance(kv_tree, dict)

        def collection(name: str) -> PagedCacheValues:
            leaves = tree.leaves(kv_tree[name], leaf=KVCacheInputsPerDevice)
            return PagedCacheValues.from_upstream(leaves, replicated)

        latent, indexer = collection("mla"), collection("indexer")
        metadata = self.layers[0].create_mla_prefill_metadata(offsets, latent)
        metadata.buffer_lengths = batch_context_length
        # As the model does: one predicated step, its plan built once over
        # the PCP path's rows. ``active`` gives every row to PCP or to DCP;
        # both paths run in every layer.
        pcp = PrefillContextParallelStep(
            dcp=_N,
            interleave=1,
            page_size=_PAGE_SIZE,
            namespace="pcp",
            active=active,
            batch_context_length=batch_context_length,
        ).predicated(h)
        pcp = pcp.with_plan(offsets, pcp.rows_split(h)[0], latent, indexer)

        outs = []
        topk: Selection | None = None
        slot = 0
        for idx, (kind, layer) in enumerate(
            zip(_LAYER_TYPES, self.layers, strict=True)
        ):
            layer_idx = F.constant(idx, DType.uint32, device=CPU())
            indexer_slot = F.constant(slot, DType.uint32, device=CPU())
            slot += kind == "full"
            out, topk = as_subgraph(layer, name=f"sparse_attn_{kind}")(
                h,
                latent,
                indexer,
                freqs,
                layer_idx,
                offsets,
                metadata,
                topk,
                False,
                indexer_slot,
                pcp=pcp,
            )
            if any(isinstance(p, Partial) for p in out.placements):
                out = F.allreduce_sum(out)
            outs.append(out.to(mesh.devices[0]))
            h = h + out
        return tuple(outs)


def _input_types(kv_params: MultiKVCacheParams) -> list[Any]:
    gpu = DeviceRef.GPU(0)
    return [
        TensorType(DType.bfloat16, ["total_seq_len", _HIDDEN], gpu),
        TensorType(DType.uint32, ["input_row_offsets_len"], gpu),
        TensorType(DType.float32, [_MAX_SEQ_LEN, _QK_ROPE_HEAD_DIM], gpu),
        TensorType(DType.bool, [], DeviceRef.CPU()),
        TensorType(DType.int32, [1], DeviceRef.CPU()),
        kv_params.get_symbolic_inputs(),
    ]


def _random_weights(model: Module[..., Any]) -> dict[str, Buffer]:
    """Seeded weights for every parameter: unit norms and FP8 scales."""
    generator = torch.Generator().manual_seed(0)
    weights: dict[str, Buffer] = {}
    for name, param in model.parameters:
        shape = tuple(int(d) for d in param.shape)
        if "norm" in name or "scale" in name:
            values = torch.ones(shape, dtype=torch.float32)
        else:
            values = torch.randn(shape, generator=generator) / np.sqrt(
                shape[-1]
            )
        if param.dtype == DType.float8_e4m3fn:
            raw = values.to(torch.float8_e4m3fn).view(torch.uint8)
            weights[name] = Buffer.from_dlpack(raw).view(DType.float8_e4m3fn)
        elif param.dtype == DType.bfloat16:
            weights[name] = Buffer.from_dlpack(values.to(torch.bfloat16))
        else:
            assert param.dtype == DType.float32, (name, param.dtype)
            weights[name] = Buffer.from_dlpack(values)
    return weights


def _freqs_cis() -> np.ndarray:
    """Interleaved cos/sin rope rows, one per position."""
    half = _QK_ROPE_HEAD_DIM // 2
    inv = 10000.0 ** (-np.arange(half, dtype=np.float64) / half)
    angles = np.arange(_MAX_SEQ_LEN)[:, None] * inv[None, :]
    out = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return out.reshape(_MAX_SEQ_LEN, _QK_ROPE_HEAD_DIM).astype(np.float32)


def _batch_context_length(positions: list[int]) -> Buffer:
    """The page-aligned summed context, as the batch processor computes it."""
    length = sum(-(-p // _PAGE_SIZE) * _PAGE_SIZE for p in positions)
    return Buffer.from_numpy(np.array([length], dtype=np.int32))


def test_pcp_matches_dcp_with_a_shared_scratch() -> None:
    # Fails rather than skips without two GPUs, so a green run proves TP2.
    assert accelerator_count() >= _N, f"needs {_N} GPUs"
    devices = [Accelerator(i) for i in range(_N)]
    mesh = DeviceMesh(tuple(devices), (_N,), (TP,))
    kv_params, mla = _kv_params([DeviceRef.from_device(d) for d in devices])

    with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
        layers = [
            tensor_parallel_sparse_latent_attention_with_rope(
                _make_layer(i, mla), _N
            )
            for i in range(_NUM_LAYERS)
        ]
        model = _Layers(layers, mesh, kv_params)
    session = InferenceSession(devices=devices)
    run = model.compile(
        *_input_types(kv_params),
        weights=_random_weights(model),
        session=session,
    )

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=_TOTAL_NUM_PAGES,
        session=session,
        max_batch_size=len(_PREFIX),
    )
    batch = [
        create_text_context(np.arange(p + c), max_length=p + c + 8)
        for p, c in zip(_PREFIX, _CHUNK, strict=True)
    ]
    for ctx, prefix in zip(batch, _PREFIX, strict=True):
        kv_manager.claim(ctx)
        ctx.tokens.chunk(prefix)

    generator = torch.Generator().manual_seed(1)
    gpu = devices[0]
    freqs = Buffer.from_numpy(_freqs_cis()).to(gpu)

    def hidden(rows: int) -> Buffer:
        values = torch.randn((rows, _HIDDEN), generator=generator)
        return Buffer.from_dlpack(values.to(torch.bfloat16)).to(gpu)

    def offsets(lengths: tuple[int, ...]) -> Buffer:
        values = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)
        return Buffer.from_numpy(values).to(gpu)

    def step(x: Buffer, lengths: tuple[int, ...], pcp: bool) -> list[Any]:
        for ctx in batch:
            kv_manager.alloc(ctx)
        kv = kv_manager.runtime_inputs([batch])
        outs = run.execute_raw(
            x,
            offsets(lengths),
            freqs,
            Buffer.from_numpy(np.array(pcp)),
            _batch_context_length(
                [ctx.tokens.current_position for ctx in batch]
            ),
            *tree.leaves(kv),
        )
        return [from_dlpack(o).cpu().float().numpy() for o in outs]

    # The prefix, stored by ownership on the DCP path.
    step(hidden(sum(_PREFIX)), _PREFIX, pcp=False)
    for ctx in batch:
        ctx.tokens.advance_chunk()

    # The chunk on each path. Both store the same rows into the DCP caches,
    # so the second run starts from the cache the first did.
    x = hidden(sum(_CHUNK))
    want = step(x, _CHUNK, pcp=False)
    got = step(x, _CHUNK, pcp=True)

    # Every layer's error first, so a failure shows which kinds diverged.
    errors = []
    for layer, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g.shape == (sum(_CHUNK), _HIDDEN)
        assert np.isfinite(g).all() and np.isfinite(w).all()
        assert np.any(w != 0)
        rel = float(np.linalg.norm(g - w) / np.linalg.norm(w))
        peak = float(np.max(np.abs(g - w)) / np.max(np.abs(w)))
        kind = _LAYER_TYPES[layer]
        print(
            f"layer {layer} ({kind}): PCP vs DCP rel {rel:.3e}, max {peak:.3e}"
        )
        errors.append((layer, kind, rel))
    # Measured on B200: about 6e-3 when every layer reads its own prefix,
    # 0.17 and up when one reads another's. The bound sits between.
    bad = [e for e in errors if e[2] >= 3e-2]
    assert not bad, (
        f"PCP differs from DCP in {bad} (layer, kind, rel); did a layer read"
        " another layer's prefix from the shared scratch?"
    )
