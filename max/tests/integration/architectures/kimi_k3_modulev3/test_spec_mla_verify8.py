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
"""Eight-rows-per-request MLA verify over an FP8 latent cache.

A DSpark verify step feeds each request 8 rows -- the anchor plus 7 drafts --
through the target's MLA layers over the same FP8 latent cache production
serves with. Checks the properties the spec pipeline leans on:

* one 8-row verify must match feeding the same rows one decode step at a
  time, with requests whose windows cross a 128-token page boundary in the
  batch;
* the verify must agree between TP1 and TP2;
* the decode fold (production TP8's arm) must not read the prefill plan: a
  truncating plan cap changes nothing on that route;
* sabotaged sizing must fail the parity check rather than pass it: pages
  sized for the decode assumption (one row per request) and a plan cap
  below the batch's true window both produce visibly wrong output (RED
  controls proving the test can catch the defect it exists for).

Head counts pick the kernel arm the way production's TP split does: 12
heads is K3 at TP8's per-rank count, which on gfx950 routes S=8 through the
AMD decode fold (``num_heads <= 16``); 96 heads is the full model, which
routes S=8 through the one-shot decompression prefill branch.
"""

from __future__ import annotations

import os

os.environ.setdefault("MODULAR_DEBUG", "source-tracebacks")

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
from _tp_model_test_utils import rel_error, spec_decode_context
from max import functional as F
from max import tree
from max.driver import CPU, Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kernels import flare_mla_prefill_plan
from max.nn.kv_cache import (
    MIXED_STEP_SPLIT_SIZE,
    PACKED_PAGE_STRIDE,
    KVCacheInputsPerDevice,
    MLAAttnKey,
    MLAKVCacheParams,
    padded_lut_cols,
)
from max.nn.kv_cache.input_types import _mixed_step_decode_view
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
    kimi_k3_mla_prefill_plan,
    kimi_k3_nope_freqs_cis,
    tensor_parallel_mla_attention,
)
from max.pipelines.architectures.kimi_k3_modulev3.spec_mla import (
    kimi_k3_mla_buffer_size,
    kimi_k3_mla_table_rows,
)
from max.pipelines.kv_cache.paged_kv_cache._common import (
    cache_valid_length_for_context,
)
from max.pipelines.kv_cache.paged_kv_cache._mixed_step_split import (
    mixed_step_decode_width,
    mixed_step_split_record,
)
from max.support.math import ceildiv
from max.tensor import Tensor, default_device
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import DeviceMesh

_PAGE_SIZE = 128
_VERIFY_ROWS = 8
_NUM_DRAFT_TOKENS = _VERIFY_ROWS - 1
# Two of the four requests' verify windows cross a 128-token page boundary
# (125 + 8 = 133 and 253 + 8 = 261); the other two stay inside their pages.
_CACHE_LENGTHS = (125, 253, 3, 200)
# The longest committed context (253) plus its verify rows: the test
# "model"'s context limit.
_MAX_SEQ_LEN = 261
# The batch's page-aligned verify windows: 256 + 384 + 128 + 256.
_TRUE_PADDED_WINDOW = 1024

# Production head dims -- the latent width must be exactly 576 for the fused
# MLA kernels -- over a shrunken hidden size.
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 128
_Q_LORA_RANK = 1536
_KV_LORA_RANK = 512
_HIDDEN = 512
_KV_CACHE_DTYPE = DType.float8_e4m3fn

_HEADS_FOLD_ARM = 12
_HEADS_PREFILL_ARM = 96


def _plan_cap(heads_per_device: int) -> int:
    """The plan cap the shipped sizing helper computes for this test's model."""
    return kimi_k3_mla_buffer_size(
        max_batch_context_length=_TRUE_PADDED_WINDOW,
        max_seq_len=_MAX_SEQ_LEN,
        page_size=_PAGE_SIZE,
        num_draft_tokens=_NUM_DRAFT_TOKENS,
        max_batch_size=len(_CACHE_LENGTHS),
        num_q_heads_per_device=heads_per_device,
        cache_dtype=_KV_CACHE_DTYPE,
        device_api=accelerator_api(),
    )


# Both arms compare an 8-row verify against S=1 decode steps over an FP8
# cache, so the bound is the cross-width FP8 noise of the pair of kernels
# each arm exercises, not exact equality. Measured on MI355: the fold arm
# (same kernel family, S=8 vs S=1) sits at ~3e-3 relative; the prefill arm
# crosses BRANCHES (one-shot decompression vs absorbed decode), whose FP8
# quantization points differ, and sits at ~5.5e-2 relative. The TP2-vs-TP1
# comparisons run the SAME kernel arm on both sides, so they get the
# tighter bound a sharding defect would otherwise hide inside.
_TOL_FOLD_ARM = 2e-2
_TOL_PREFILL_ARM = 0.12
_TOL_TP_PARITY = 2e-2
# The bound for "nothing changed": an output that must be bit-identical
# still carries float dust in the relative-error report.
_TOL_BITEXACT = 1e-6

_RNG = np.random.default_rng(20261117)


def _parity_report(label: str, got: np.ndarray, want: np.ndarray) -> float:
    """Prints one comparison's relative error, max abs diff and scale."""
    err = rel_error(got, want)
    print(
        f"{label}: rel={err}"
        f" max_abs={np.max(np.abs(got - want))}"
        f" scale={np.max(np.abs(want))}"
    )
    return err


def _zeros(shape: tuple[int, ...], dtype: DType, device: Accelerator) -> Tensor:
    """A real zero-filled device tensor, the way ``_paged_cache_test_utils``
    builds cache blocks (``Tensor.zeros`` traces a broadcast the interpreter
    caps below the blocks buffer's rank)."""
    raw = np.zeros((*shape[:-1], shape[-1] * dtype.size_in_bytes), np.uint8)
    buf = Buffer.from_dlpack(raw).to(device).view(dtype=dtype, shape=shape)
    return Tensor(storage=buf)


def _random_pages(
    shape: tuple[int, ...],
    dtype: DType,
    lut: np.ndarray,
    blocks_per_request: Sequence[int],
    fill_ids: Sequence[int],
) -> Buffer:
    """FP8 blocks whose pages hold small random latents seeded by request
    id, so a request's cached prefix is the same in every pool."""
    assert dtype == DType.float8_e4m3fn
    page_bytes = math.prod(shape[1:])
    raw = np.zeros((shape[0], page_bytes), np.uint8)
    for req, (count, fill_id) in enumerate(
        zip(blocks_per_request, fill_ids, strict=True)
    ):
        rng = np.random.default_rng(1000 + fill_id)
        # Exponent at most 7 and a random sign: |x| < 2, never NaN.
        raw[lut[req, :count]] = rng.integers(
            0, 0x40, (count, page_bytes), np.uint8
        ) | (rng.integers(0, 2, (count, page_bytes), np.uint8) << 7)
    return Buffer.from_numpy(raw).view(dtype=dtype, shape=shape)


class _LatentPool:
    """One FP8 MLA latent pool whose pages survive across steps.

    Production's cache manager owns this state; the test keeps one blocks
    buffer and rebuilds only the per-step metadata (cache lengths, lookup
    table, dispatch scalars), so a later step reads what an earlier step
    wrote -- what a stepwise oracle needs.
    """

    def __init__(
        self,
        params: MLAKVCacheParams,
        blocks_per_request: Sequence[int],
        device: Accelerator,
        fill_ids: Sequence[int] | None = None,
    ) -> None:
        self._params = params
        self._device = device
        total = sum(blocks_per_request)
        shape = (total + 1, *params.shape_per_block)
        self._lut = np.full(
            (len(blocks_per_request), padded_lut_cols(max(blocks_per_request))),
            total,
            dtype=np.uint32,
        )
        next_block = 0
        for req, count in enumerate(blocks_per_request):
            self._lut[req, :count] = np.arange(
                next_block, next_block + count, dtype=np.uint32
            )
            next_block += count
        self._kv_blocks = (
            _zeros(shape, params.dtype, device)
            if fill_ids is None
            else Tensor(
                storage=_random_pages(
                    shape, params.dtype, self._lut, blocks_per_request, fill_ids
                ).to(device)
            )
        )

    def values(
        self,
        cache_lengths: Sequence[int],
        prompt_lens: Sequence[int],
        record: np.ndarray | None = None,
    ) -> PagedCacheValues:
        """Builds one step's cache view over the persistent pool; a split
        ``record`` fills the dispatch key's decode view as the cache manager
        does."""
        max_prompt = max(prompt_lens)
        max_cache = max(
            cache + prompt
            for cache, prompt in zip(cache_lengths, prompt_lens, strict=True)
        )
        key = self._params.resolve_attn_key(
            len(prompt_lens), max_prompt, max_cache
        )
        assert isinstance(key, MLAAttnKey)
        view = (
            None
            if record is None
            else _mixed_step_decode_view(record, len(prompt_lens))
        )
        if view is not None:
            decode_key = self._params.resolve_attn_key(*view)
            assert isinstance(decode_key, MLAAttnKey)
            key = replace(
                key,
                decode_batch_size=decode_key.batch_size,
                decode_max_prompt_length=decode_key.max_prompt_length,
                decode_num_partitions=decode_key.num_partitions,
            )
        device = self._device
        # The host-resident fields match the cache's declared symbolic input
        # types (``MLAKVCacheParams._get_symbolic_inputs_for_replica``): the
        # kernels read page stride, prompt/cache length and the MLA partition
        # count by value on the host, so a compiled graph binds them from
        # host memory, as production's staging does.
        with ensure_context():
            return PagedCacheValues(
                kv_blocks=self._kv_blocks,
                cache_lengths=Tensor(
                    storage=Buffer.from_numpy(
                        np.array(cache_lengths, np.uint32)
                    ).to(device)
                ),
                lookup_table=Tensor(
                    storage=Buffer.from_numpy(self._lut.copy()).to(device)
                ),
                max_prompt_length=Tensor(
                    storage=Buffer.from_numpy(np.array([max_prompt], np.uint32))
                ),
                max_cache_length=Tensor(
                    storage=Buffer.from_numpy(np.array([max_cache], np.uint32))
                ),
                page_stride=Tensor(
                    storage=Buffer.from_numpy(
                        np.array([PACKED_PAGE_STRIDE], np.int64)
                    )
                ),
                attention_dispatch_metadata=Tensor(
                    storage=key.pack_into_buffer(device, max_cache)
                ),
                mla_num_partitions=Tensor(
                    storage=Buffer.from_numpy(
                        np.array(
                            [key.num_partitions, key.decode_num_partitions],
                            np.int64,
                        )
                    )
                ),
                # Every paged leaf declares the record; zeros mean no split.
                mixed_step_split=Tensor(
                    storage=Buffer.from_numpy(
                        np.zeros(MIXED_STEP_SPLIT_SIZE, np.uint32)
                        if record is None
                        else record.astype(np.uint32)
                    )
                ),
            )


def _kv_params(heads: int, devices: Sequence[Accelerator]) -> MLAKVCacheParams:
    return MLAKVCacheParams(
        dtype=_KV_CACHE_DTYPE,
        head_dim=_KV_LORA_RANK + _QK_ROPE_HEAD_DIM,
        num_layers=1,
        devices=[DeviceRef.from_device(d) for d in devices],
        page_size=_PAGE_SIZE,
        num_q_heads=heads,
    )


def _random_weights(heads: int) -> dict[str, np.ndarray]:
    """One weight draw, shared by every layer a test compares."""
    shapes = {
        "q_a_proj": (_Q_LORA_RANK, _HIDDEN),
        "q_b_proj": (
            heads * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM),
            _Q_LORA_RANK,
        ),
        "kv_a_proj_with_mqa": (_KV_LORA_RANK + _QK_ROPE_HEAD_DIM, _HIDDEN),
        "kv_b_proj": (
            heads * (_QK_NOPE_HEAD_DIM + _V_HEAD_DIM),
            _KV_LORA_RANK,
        ),
        "g_proj": (heads * _V_HEAD_DIM, _HIDDEN),
        "o_proj": (_HIDDEN, heads * _V_HEAD_DIM),
    }
    return {
        name: (_RNG.standard_normal(shape) * 0.05).astype(np.float32)
        for name, shape in shapes.items()
    }


def _make_layer(
    heads: int,
    kv_params: MLAKVCacheParams,
    weights: dict[str, np.ndarray],
) -> KimiK3MLAAttention:
    """Builds a random-weight MLA layer with production head geometry."""
    layer = KimiK3MLAAttention(
        hidden_size=_HIDDEN,
        num_heads=heads,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        q_lora_rank=_Q_LORA_RANK,
        kv_lora_rank=_KV_LORA_RANK,
        kv_params=kv_params,
        layer_idx=0,
    )
    for name, value in weights.items():
        weight = getattr(layer, name).weight
        getattr(layer, name).weight = Tensor(value, dtype=DType.float32).cast(
            weight.dtype
        )
    layer.q_a_layernorm.weight = Tensor(
        np.ones(_Q_LORA_RANK, np.float32), dtype=DType.float32
    ).cast(layer.q_a_layernorm.weight.dtype)
    layer.kv_a_layernorm.weight = Tensor(
        np.ones(_KV_LORA_RANK, np.float32), dtype=DType.float32
    ).cast(layer.kv_a_layernorm.weight.dtype)
    return layer


def _step_input() -> np.ndarray:
    """The hidden states the verify step feeds: ``[requests, rows, hidden]``.

    Drawn once per test so the batched verify and the stepwise oracle see
    identical rows.
    """
    return (
        _RNG.standard_normal((len(_CACHE_LENGTHS), _VERIFY_ROWS, _HIDDEN))
        * 0.05
    ).astype(np.float32)


def _plan_extent(cache_lengths: Sequence[int], rows_per_request: int) -> Tensor:
    """The host plan extent that production threads through the spec adapter.

    Each request's window -- its cached rows plus this call's rows --
    page-aligned and summed. The plan reads it host-side; passing it keeps
    the test on the capture-safe path instead of reading the device plan
    back.
    """
    # The kernel reads the extent as a host scalar, so the tensor must not
    # ride the caller's default device.
    return Tensor(
        np.array(
            [
                sum(
                    ceildiv(cache + rows_per_request, _PAGE_SIZE) * _PAGE_SIZE
                    for cache in cache_lengths
                )
            ],
            np.int32,
        ),
        dtype=DType.int32,
        device=CPU(),
    )


def _run_rows(
    layer: KimiK3MLAAttention,
    pool: _LatentPool,
    cache_lengths: Sequence[int],
    rows_per_request: int,
    hidden_rows: np.ndarray,
    buffer_size: int,
    device: Accelerator,
) -> np.ndarray:
    """Runs one forward over ``rows_per_request`` rows per request.

    ``hidden_rows`` is laid out request-major: ``[num_requests, rows,
    hidden]`` flattened to the ragged batch's row order.
    """
    num_requests = len(cache_lengths)
    total_rows = num_requests * rows_per_request
    assert isinstance(layer.kv_params, MLAKVCacheParams)
    x = Tensor(
        hidden_rows.reshape(total_rows, _HIDDEN), dtype=DType.float32
    ).cast(DType.bfloat16)
    input_row_offsets = Tensor(
        np.arange(0, total_rows + 1, rows_per_request, dtype=np.uint32),
        dtype=DType.uint32,
    )
    kv_collection = pool.values(
        cache_lengths, [rows_per_request] * num_requests
    )
    plan = kimi_k3_mla_prefill_plan(
        kv_params=layer.kv_params,
        kv_collection=kv_collection,
        input_row_offsets=input_row_offsets,
        buffer_size=buffer_size,
        buffer_length=_plan_extent(cache_lengths, rows_per_request),
    )
    freqs_cis = kimi_k3_nope_freqs_cis(
        max_seq_len=buffer_size,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        dtype=layer.kv_params.dtype,
        device=device,
    )
    out = layer.forward_latent_ragged(
        x, kv_collection, input_row_offsets, plan, freqs_cis
    )
    return out.cast(DType.float32).to_numpy()


def _verify_once(
    layer: KimiK3MLAAttention,
    pool: _LatentPool,
    hidden_rows: np.ndarray,
    device: Accelerator,
    buffer_size: int | None = None,
) -> np.ndarray:
    """One 8-rows-per-request verify step at the helper-sized plan cap."""
    if buffer_size is None:
        buffer_size = _plan_cap(layer.num_heads)
    return _run_rows(
        layer,
        pool,
        _CACHE_LENGTHS,
        _VERIFY_ROWS,
        hidden_rows,
        buffer_size,
        device,
    )


def _stepwise(
    layer: KimiK3MLAAttention,
    pool: _LatentPool,
    hidden_rows: np.ndarray,
    device: Accelerator,
) -> np.ndarray:
    """The oracle: the same 8 rows one decode step at a time."""
    buffer_size = _plan_cap(layer.num_heads)
    per_step = []
    for row in range(_VERIFY_ROWS):
        per_step.append(
            _run_rows(
                layer,
                pool,
                [cache + row for cache in _CACHE_LENGTHS],
                1,
                hidden_rows[:, row, :],
                buffer_size,
                device,
            )
        )
    # per_step is [rows][num_requests, hidden]; the verify output is
    # request-major, so interleave before flattening.
    return np.stack(per_step, axis=1).reshape(-1, _HIDDEN)


def _healthy_pool(params: MLAKVCacheParams, device: Accelerator) -> _LatentPool:
    """A pool with a page for every row the verify step writes."""
    return _LatentPool(
        params,
        [ceildiv(cache + _VERIFY_ROWS, _PAGE_SIZE) for cache in _CACHE_LENGTHS],
        device,
    )


def _decode_assumed_pool(
    params: MLAKVCacheParams, device: Accelerator
) -> _LatentPool:
    """A pool sized from the decode assumption: one new row per request.

    The block manager allocates exactly this way when the MLA cache leaf's
    ``num_draft_tokens`` is 0 -- pages for the committed context plus one
    row, none for the drafts.
    """
    return _LatentPool(
        params,
        [ceildiv(cache + 1, _PAGE_SIZE) for cache in _CACHE_LENGTHS],
        device,
    )


class _TPVerifyStep(Module[..., Tensor]):
    """One 8-row verify through the compiled path a TP-served model takes.

    Mirrors ``KimiK3Model.forward``'s MLA block: broadcast the batch inputs,
    rebuild the distributed cache view from the per-device inputs, plan
    in-graph, run the sharded layer, and allreduce the ``o_proj`` partials.
    The eager path cannot stage the cache's mutable blocks this way, so a
    TP-served verify is only provable compiled.
    """

    def __init__(
        self,
        layer: KimiK3MLAAttention,
        attn_params: MLAKVCacheParams,
        qk_rope_head_dim: int,
        buffer_size: int,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.attn_params = attn_params
        self.qk_rope_head_dim = qk_rope_head_dim
        self.buffer_size = buffer_size

    def forward(
        self, x: Tensor, input_row_offsets: Tensor, *kv_inputs: Tensor
    ) -> Tensor:
        mesh = self.layer.o_proj.weight.mesh
        if mesh.num_devices > 1:
            x = F.distributed_broadcast(x, mesh)
            input_row_offsets = F.distributed_broadcast(input_row_offsets, mesh)
        attn_tree = self.attn_params.unflatten_kv_inputs(
            iter(value._graph_value for value in kv_inputs)
        )
        kv_collection = PagedCacheValues.from_upstream(
            tree.leaves(attn_tree, leaf=KVCacheInputsPerDevice), x.mapping
        )
        plan = kimi_k3_mla_prefill_plan(
            kv_params=self.attn_params,
            kv_collection=kv_collection,
            input_row_offsets=input_row_offsets,
            buffer_size=self.buffer_size,
            buffer_length=_plan_extent(_CACHE_LENGTHS, _VERIFY_ROWS),
        )
        freqs_cis = kimi_k3_nope_freqs_cis(
            max_seq_len=self.buffer_size,
            qk_rope_head_dim=self.qk_rope_head_dim,
            dtype=self.attn_params.dtype,
            device=CPU(),
        ).to(mesh)
        partial = self.layer.forward_latent_ragged(
            x, kv_collection, input_row_offsets, plan, freqs_cis
        )
        full = F.allreduce_sum(partial)
        return full.to(mesh.devices[0])


def _run_tp2(
    layer: KimiK3MLAAttention,
    heads: int,
    params: MLAKVCacheParams,
    hidden_rows: np.ndarray,
    devices: Sequence[Accelerator],
) -> np.ndarray:
    """Compiles and runs one 8-row verify sharded across two devices."""
    num_requests = len(_CACHE_LENGTHS)
    total_rows = num_requests * _VERIFY_ROWS
    per_device = tuple(
        _healthy_pool(params, device).values(
            list(_CACHE_LENGTHS), [_VERIFY_ROWS] * num_requests
        )
        for device in devices
    )
    cap = _plan_cap(heads // len(devices))
    step = _TPVerifyStep(layer, params, _QK_ROPE_HEAD_DIM, cap)
    x_type = TensorType(
        DType.bfloat16,
        shape=["total_seq_len", _HIDDEN],
        device=DeviceRef.from_device(devices[0]),
    )
    row_offsets_type = TensorType(
        DType.uint32,
        shape=["input_row_offsets_len"],
        device=DeviceRef.from_device(devices[0]),
    )
    run = step.compile(
        x_type, row_offsets_type, *tree.leaves(params.get_symbolic_inputs())
    )
    x = Tensor(
        hidden_rows.reshape(total_rows, _HIDDEN),
        dtype=DType.float32,
        device=devices[0],
    ).cast(DType.bfloat16)
    input_row_offsets = Tensor(
        np.arange(0, total_rows + 1, _VERIFY_ROWS, dtype=np.uint32),
        dtype=DType.uint32,
        device=devices[0],
    )
    out = run(x, input_row_offsets, *tree.leaves(per_device))
    return out.cast(DType.float32).to_numpy()


def _build_tp2(
    heads: int,
    devices: Sequence[Accelerator],
    mesh: DeviceMesh,
    weights: dict[str, np.ndarray],
) -> tuple[KimiK3MLAAttention, MLAKVCacheParams]:
    params = _kv_params(heads, devices)
    layer = _make_layer(heads, params, weights)
    return tensor_parallel_mla_attention(layer, mesh), params


@pytest.mark.compile_on_gpu(
    "the fused MLA prefill/decode graph op is not interpreter-bindable"
)
def test_verify8_fold_arm_matches_stepwise() -> None:
    """Twelve heads (K3 at TP8's per-rank count): the 8-row verify takes the
    AMD decode fold, must match the rows fed one decode step at a time,
    and -- because the fold reads the cache directly, not the plan --
    must be unable to see a truncating plan cap, while pages sized for
    the decode assumption corrupt it past the parity bound (the RED
    control: the ModuleV3 form of the one-page-short defect). One healthy
    verify feeds every check.
    """
    device = Accelerator()
    with default_device(device):
        params = _kv_params(_HEADS_FOLD_ARM, [device])
        layer = _make_layer(
            _HEADS_FOLD_ARM, params, _random_weights(_HEADS_FOLD_ARM)
        )
        hidden_rows = _step_input()
        # One healthy pool serves the healthy, stepwise and truncated runs:
        # every run stores its own rows before attending, so sharing the
        # pool's pages and lookup table changes nothing but the rebuild
        # cost.
        healthy_pool = _healthy_pool(params, device)

        healthy = _verify_once(layer, healthy_pool, hidden_rows, device)
        stepwise = _stepwise(layer, healthy_pool, hidden_rows, device)
        sabotaged = _verify_once(
            layer,
            _decode_assumed_pool(params, device),
            hidden_rows,
            device,
        )
        truncated = _verify_once(
            layer,
            healthy_pool,
            hidden_rows,
            device,
            buffer_size=_TRUE_PADDED_WINDOW - _PAGE_SIZE,
        )

    err = _parity_report("fold arm verify8 vs stepwise", healthy, stepwise)
    assert err < _TOL_FOLD_ARM, (
        f"8-row verify (decode fold) diverges from stepwise decode: {err}"
    )

    err = _parity_report("RED pages-sized-for-decode", sabotaged, healthy)
    assert err > _TOL_FOLD_ARM, (
        "an undersized page pool produced output that passes the verify8"
        f" parity check (rel={err}); the test cannot catch the defect"
    )

    # The production TP8 shape serves verify steps through the fold, which
    # takes none of the plan's outputs (``mla_decode_branch_bf16``/``_fp8``
    # receive no ``buffer_row_offsets``/``cache_offsets``/``buffer_length``;
    # only the prefill branches read them), so the cap the plan was built
    # with is dead data on this route. The same cap that NaN's the prefill
    # arm's truncated request below must leave the fold's output
    # bit-identical.
    err = rel_error(truncated, healthy)
    print(f"fold arm verify8 with a truncating plan cap: rel={err}")
    assert err < _TOL_BITEXACT, (
        "the decode fold read the prefill plan: a truncating cap changed"
        f" its output (rel={err})"
    )


@pytest.mark.compile_on_gpu(
    "the fused MLA prefill/decode graph op is not interpreter-bindable"
)
def test_verify8_prefill_arm_matches_stepwise() -> None:
    """Ninety-six heads (the full model, as TP1/TP2 serve it): the 8-row
    verify takes the one-shot decompression prefill branch on gfx9, must
    match stepwise decode bound by the cross-branch FP8 difference, and
    must be corrupted by a plan cap one page short of the batch's true
    window -- the undersizing the TG-uncapped scheduler regime makes
    possible -- only on the request it truncates (the RED control). One
    healthy verify feeds every check.

    The plan-cap RED control only exists where S > 1 leaves the decode
    fold: ``mla_decode_max_seq_len`` (mla.mojo) folds an 8-row verify at
    any head count off AMD, and a fold-served verify never reads the
    prefill plan -- a truncating cap changes nothing there by design, so
    off gfx9 this test checks the healthy parity alone.
    """
    device = Accelerator()
    with default_device(device):
        params = _kv_params(_HEADS_PREFILL_ARM, [device])
        layer = _make_layer(
            _HEADS_PREFILL_ARM, params, _random_weights(_HEADS_PREFILL_ARM)
        )
        hidden_rows = _step_input()
        healthy_pool = _healthy_pool(params, device)

        healthy = _verify_once(layer, healthy_pool, hidden_rows, device)
        stepwise = _stepwise(layer, healthy_pool, hidden_rows, device)

    err = _parity_report("prefill arm verify8 vs stepwise", healthy, stepwise)
    assert err < _TOL_PREFILL_ARM, (
        f"8-row verify (prefill branch) diverges from stepwise decode: {err}"
    )

    if accelerator_api() != "hip":
        # The decode fold served this verify whole, so the plan went unread
        # and no cap can truncate it.
        return

    with default_device(device):
        short = _verify_once(
            layer,
            healthy_pool,
            hidden_rows,
            device,
            buffer_size=_TRUE_PADDED_WINDOW - _PAGE_SIZE,
        )

    # The truncated cap still covers the first three requests' windows;
    # only the fourth's is clipped.
    fitted_rows = 3 * _VERIFY_ROWS
    untouched = healthy[:fitted_rows]
    untouched_short = short[:fitted_rows]
    assert rel_error(untouched_short, untouched) < _TOL_BITEXACT, (
        "shrinking the plan cap changed requests whose windows fit"
    )
    err = _parity_report(
        "RED plan-cap-one-page-short (truncated request)",
        short[fitted_rows:],
        healthy[fitted_rows:],
    )
    # The truncated window reads past the decompression buffer, so the wrong
    # rows are garbage -- NaN here -- which is the defect being caught, not
    # a numerical near-miss; `isnan` treats it as the failure it is.
    assert math.isnan(err) or err > _TOL_PREFILL_ARM, (
        "a plan cap one page short produced output that passes the verify8"
        f" parity check (rel={err}); the test cannot catch the defect"
    )


@pytest.mark.compile_on_gpu(
    "the fused MLA prefill/decode graph op is not interpreter-bindable"
)
def test_verify8_fold_arm_tp2_matches_tp1() -> None:
    """The 8-row verify at TP2 (6 heads per rank) must match TP1 (12 heads).

    Twelve heads is K3 at TP8's per-rank count, so the TP1 leg reproduces
    the production fold geometry (M = 12*8 = 96 rows); the TP2 leg checks
    the fold survives a head split. Production TP2 geometry (48 heads per
    rank, prefill arm) is covered by the prefill-arm TP2 test below.
    """
    hidden_rows = _step_input()
    weights = _random_weights(_HEADS_FOLD_ARM)

    with default_device(Accelerator(0)):
        device = Accelerator(0)
        params = _kv_params(_HEADS_FOLD_ARM, [device])
        layer = _make_layer(_HEADS_FOLD_ARM, params, weights)
        tp1 = _verify_once(
            layer, _healthy_pool(params, device), hidden_rows, device
        )

    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        layer2, params2 = _build_tp2(_HEADS_FOLD_ARM, devices, mesh, weights)
        tp2 = _run_tp2(layer2, _HEADS_FOLD_ARM, params2, hidden_rows, devices)

    err = _parity_report("fold arm verify8 TP2 vs TP1", tp2, tp1)
    assert err < _TOL_TP_PARITY, f"verify8 TP mismatch (fold arm): {err}"


@pytest.mark.compile_on_gpu(
    "the fused MLA prefill/decode graph op is not interpreter-bindable"
)
def test_verify8_prefill_arm_tp2_matches_tp1() -> None:
    """The 8-row verify at TP2 (48 heads per rank) must match TP1 (96)."""
    hidden_rows = _step_input()
    weights = _random_weights(_HEADS_PREFILL_ARM)

    with default_device(Accelerator(0)):
        device = Accelerator(0)
        params = _kv_params(_HEADS_PREFILL_ARM, [device])
        layer = _make_layer(_HEADS_PREFILL_ARM, params, weights)
        tp1 = _verify_once(
            layer, _healthy_pool(params, device), hidden_rows, device
        )

    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        layer2, params2 = _build_tp2(_HEADS_PREFILL_ARM, devices, mesh, weights)
        tp2 = _run_tp2(
            layer2, _HEADS_PREFILL_ARM, params2, hidden_rows, devices
        )

    err = _parity_report("prefill arm verify8 TP2 vs TP1", tp2, tp1)
    assert err < _TOL_TP_PARITY, f"verify8 TP mismatch (prefill arm): {err}"


@pytest.mark.compile_on_gpu(
    "the fused MLA prefill/decode graph op is not interpreter-bindable"
)
def test_plan_extent_matches_the_host_helper_on_random_batches() -> None:
    """The plan op's own extent is the host's per-request window sum.

    ``k3_spec_mla_extent``'s arms are upper bounds on
    ``sum(align_up(cache_valid_length_for_context(ctx, 0), page))``
    (``test_spec_mla_verify_helpers.py``), so that sum must be what the
    plan kernel computes on device from the runtime cache lengths and row
    offsets. Random batches -- drafts padded to the verify's width, as
    production pads them, windows crossing 128-token pages -- run the real
    plan op and compare its ``buffer_lengths`` output against the sum.
    """
    rng = np.random.default_rng(20261118)
    device = Accelerator()
    with default_device(device):
        params = _kv_params(_HEADS_PREFILL_ARM, [device])
        for _ in range(8):
            contexts = [
                spec_decode_context(
                    cached=int(rng.integers(1, 600)),
                    active=int(rng.integers(1, 3)),
                    # Production pads every row's drafts to the width.
                    drafts=_NUM_DRAFT_TOKENS,
                    maybe_accepted=int(rng.integers(0, _NUM_DRAFT_TOKENS + 1)),
                )
                for _ in range(int(rng.integers(1, 6)))
            ]
            cache_lengths = [
                ctx.tokens.processed_length
                + len(ctx.spec_decoding_state.maybe_accepted_draft_tokens)
                for ctx in contexts
            ]
            rows = [
                ctx.tokens.active_length
                + len(ctx.spec_decoding_state.draft_tokens_to_verify)
                for ctx in contexts
            ]
            pool = _LatentPool(
                params,
                [
                    ceildiv(cache + row, _PAGE_SIZE)
                    for cache, row in zip(cache_lengths, rows, strict=True)
                ],
                device,
            )
            kv_collection = pool.values(cache_lengths, rows)
            offsets = np.zeros(len(contexts) + 1, np.uint32)
            offsets[1:] = np.cumsum(rows)
            input_row_offsets = Tensor(offsets, dtype=DType.uint32)
            # The plan cap must not clamp the chunk (a clamped cap truncates
            # the extent itself), so charge the capture-style bound: one
            # batch of pages at the widest window -- an upper bound on the
            # per-request sum by construction.
            widest = max(
                cache + row
                for cache, row in zip(cache_lengths, rows, strict=True)
            )
            cap = len(contexts) * ceildiv(widest, _PAGE_SIZE) * _PAGE_SIZE
            _, _, device_length = flare_mla_prefill_plan(
                params,
                input_row_offsets,
                kv_collection,
                F.constant(0, DType.uint32, device=CPU()),
                cap,
                max_chunks=1,
            )
            got = int(device_length.to_numpy()[0])
            want = sum(
                ceildiv(cache_valid_length_for_context(ctx, 0), _PAGE_SIZE)
                * _PAGE_SIZE
                for ctx in contexts
            )
            assert got == want, (
                f"the plan op's extent {got} differs from the host sum"
                f" {want} on caches {cache_lengths} with rows {rows}"
            )


# Two decode rows beside a prompt row whose window spans three pages.
_MIXED_CACHE_LENGTHS = (125, 253, 70)
_MIXED_WIDTHS = (_VERIFY_ROWS, _VERIFY_ROWS, 200)
# The table production sizes for a model whose context is this batch's.
_MIXED_FREQS_ROWS = kimi_k3_mla_table_rows(
    max_seq_len=max(
        c + w for c, w in zip(_MIXED_CACHE_LENGTHS, _MIXED_WIDTHS, strict=True)
    ),
    page_size=_PAGE_SIZE,
    num_draft_tokens=_NUM_DRAFT_TOKENS,
)
# Only the projections' GEMMs, picked by row count, round differently:
# measured 4e-3 to 6e-3 on MI355, against ~7e-2 unsplit.
_TOL_MIXED_SPLIT = 1.5e-2


def _run_ragged(
    layer: KimiK3MLAAttention,
    pool: _LatentPool,
    cache_lengths: Sequence[int],
    widths: Sequence[int],
    hidden: np.ndarray,
    device: Accelerator,
    record: np.ndarray | None = None,
) -> np.ndarray:
    """One forward over rows of the given widths, request-major."""
    assert isinstance(layer.kv_params, MLAKVCacheParams)
    x = Tensor(hidden, dtype=DType.float32).cast(DType.bfloat16)
    input_row_offsets = Tensor(
        np.concatenate([[0], np.cumsum(widths)]).astype(np.uint32),
        dtype=DType.uint32,
    )
    kv_collection = pool.values(cache_lengths, widths, record=record)
    extent = sum(
        ceildiv(cache + width, _PAGE_SIZE) * _PAGE_SIZE
        for cache, width in zip(cache_lengths, widths, strict=True)
    )
    plan = kimi_k3_mla_prefill_plan(
        kv_params=layer.kv_params,
        kv_collection=kv_collection,
        input_row_offsets=input_row_offsets,
        buffer_size=extent,
        buffer_length=Tensor(
            np.array([extent], np.int32), dtype=DType.int32, device=CPU()
        ),
    )
    freqs_cis = kimi_k3_nope_freqs_cis(
        max_seq_len=_MIXED_FREQS_ROWS,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        dtype=layer.kv_params.dtype,
        device=device,
    )
    out = layer.forward_latent_ragged(
        x, kv_collection, input_row_offsets, plan, freqs_cis
    )
    return out.cast(DType.float32).to_numpy()


def _pool_for(
    params: MLAKVCacheParams,
    request_ids: Sequence[int],
    device: Accelerator,
) -> _LatentPool:
    """A pool whose requests' cached prefixes are random, the same for a
    request id in every pool."""
    return _LatentPool(
        params,
        [
            ceildiv(_MIXED_CACHE_LENGTHS[i] + _MIXED_WIDTHS[i], _PAGE_SIZE)
            for i in request_ids
        ],
        device,
        fill_ids=request_ids,
    )


@pytest.mark.skipif(
    accelerator_api() != "hip",
    reason="checks the AMD route; NVIDIA's split has its own tests",
)
@pytest.mark.compile_on_gpu(
    "the fused MLA prefill/decode graph op is not interpreter-bindable"
)
@pytest.mark.parametrize("decode_first", [True, False])
def test_mixed_split_matches_its_views_run_apart(decode_first: bool) -> None:
    """A split mixed verify gives each row what its view gives run alone, at
    the FP8 fold arm, the one AMD configuration that splits."""
    heads = _HEADS_FOLD_ARM
    order = [0, 1, 2] if decode_first else [2, 0, 1]
    caches = [_MIXED_CACHE_LENGTHS[i] for i in order]
    widths = [_MIXED_WIDTHS[i] for i in order]
    total = sum(widths)
    hidden = (_RNG.standard_normal((total, _HIDDEN)) * 0.05).astype(np.float32)
    starts = np.concatenate([[0], np.cumsum(widths)])
    rows = {i: hidden[starts[k] : starts[k + 1]] for k, i in enumerate(order)}
    record = mixed_step_split_record(
        widths,
        [c + w for c, w in zip(caches, widths, strict=True)],
        mixed_step_decode_width(_NUM_DRAFT_TOKENS),
    )
    assert record[0] != 0, "the batch must split"

    device = Accelerator()
    with default_device(device):
        params = _kv_params(heads, [device])
        layer = _make_layer(heads, params, _random_weights(heads))
        split = _run_ragged(
            layer,
            _pool_for(params, order, device),
            caches,
            widths,
            hidden,
            device,
            record=record,
        )
        unsplit = _run_ragged(
            layer,
            _pool_for(params, order, device),
            caches,
            widths,
            hidden,
            device,
        )
        decode_alone = _run_ragged(
            layer,
            _pool_for(params, [0, 1], device),
            _MIXED_CACHE_LENGTHS[:2],
            _MIXED_WIDTHS[:2],
            np.concatenate([rows[0], rows[1]]),
            device,
        )
        prompt_alone = _run_ragged(
            layer,
            _pool_for(params, [2], device),
            _MIXED_CACHE_LENGTHS[2:],
            _MIXED_WIDTHS[2:],
            rows[2],
            device,
        )

    by_request = {
        i: split[starts[k] : starts[k + 1]] for k, i in enumerate(order)
    }
    tag = f"decode_first={decode_first}"
    split_err = _parity_report(
        f"{tag} split decode rows vs decode alone",
        np.concatenate([by_request[0], by_request[1]]),
        decode_alone,
    )
    assert split_err < _TOL_MIXED_SPLIT, (
        f"{tag}: split decode rows diverge ({split_err})"
    )
    err = _parity_report(
        f"{tag} split prompt row vs prompt alone", by_request[2], prompt_alone
    )
    assert err < _TOL_MIXED_SPLIT, f"{tag}: split prompt row diverges ({err})"
    unsplit_by_request = {
        i: unsplit[starts[k] : starts[k + 1]] for k, i in enumerate(order)
    }
    unsplit_err = _parity_report(
        f"{tag} unsplit decode rows vs decode alone",
        np.concatenate([unsplit_by_request[0], unsplit_by_request[1]]),
        decode_alone,
    )
    assert unsplit_err < _TOL_PREFILL_ARM, (
        f"{tag}: unsplit decode rows ({unsplit_err})"
    )
    # Split, the decode rows stay on the fold instead of crossing to the
    # prefill branch.
    assert split_err * 3 < unsplit_err, (
        f"{tag}: the split ({split_err}) is not clearly closer than the"
        f" unsplit batch ({unsplit_err})"
    )


def _cache_bytes(pool: _LatentPool) -> np.ndarray:
    buf = pool._kv_blocks.driver_tensor.to(CPU())
    return np.asarray(buf.view(dtype=DType.uint8).to_numpy()).reshape(
        buf.shape[0], -1
    )


@pytest.mark.skipif(
    accelerator_api() != "hip",
    reason="checks the AMD route; NVIDIA's split has its own tests",
)
@pytest.mark.compile_on_gpu(
    "the fused MLA prefill/decode graph op is not interpreter-bindable"
)
@pytest.mark.parametrize("decode_first", [True, False])
def test_mixed_split_writes_the_cache_the_unsplit_batch_writes(
    decode_first: bool,
) -> None:
    """A split mixed step stores every row's new latent where the unsplit
    batch stores it, with the same bytes."""
    heads = _HEADS_FOLD_ARM
    order = [0, 1, 2] if decode_first else [2, 0, 1]
    caches = [_MIXED_CACHE_LENGTHS[i] for i in order]
    widths = [_MIXED_WIDTHS[i] for i in order]
    hidden = (_RNG.standard_normal((sum(widths), _HIDDEN)) * 0.05).astype(
        np.float32
    )
    record = mixed_step_split_record(
        widths,
        [c + w for c, w in zip(caches, widths, strict=True)],
        mixed_step_decode_width(_NUM_DRAFT_TOKENS),
    )
    device = Accelerator()
    with default_device(device):
        params = _kv_params(heads, [device])
        layer = _make_layer(heads, params, _random_weights(heads))
        split_pool = _pool_for(params, order, device)
        unsplit_pool = _pool_for(params, order, device)
        _run_ragged(
            layer, split_pool, caches, widths, hidden, device, record=record
        )
        _run_ragged(layer, unsplit_pool, caches, widths, hidden, device)
        got = _cache_bytes(split_pool)
        want = _cache_bytes(unsplit_pool)
    np.testing.assert_array_equal(
        got, want, err_msg=f"decode_first={decode_first}: cache bytes differ"
    )
