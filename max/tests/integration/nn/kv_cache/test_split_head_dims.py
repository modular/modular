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
"""KV cache params whose key and value heads have different dims.

Such a cache packs each KV head's key and value into one row of a single
plane. These tests pin the layout arithmetic, that every existing cache is
unchanged, that the ops which read the layout name it, and that every other
cache op refuses it rather than mistaking its one plane for an MLA latent.
"""

from __future__ import annotations

import csv
import dataclasses
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from max import tree
from max._artifact.value_types import treedef_from_json, treedef_to_json
from max._kv_cache_ops import mha_split_kv_decode_num_partitions
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.legacy import kernels
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.nn.legacy.kv_cache import (
    KVCacheQuantizationConfig,
    MHAAttnKey,
    MHAKVCacheParams,
    MLAKVCacheParams,
    PagedCacheValues,
    SplitKVHeadDims,
)
from max.nn.legacy.kv_cache import cache_params as cache_params_module
from max.pipelines.lib import KVCacheConfig


def _params(
    *,
    head_dim: int = 192,
    v_head_dim: int | None = None,
    dtype: DType = DType.bfloat16,
    kvcache_quant_config: KVCacheQuantizationConfig | None = None,
    window_size: int | None = None,
    stride_alignment: int | None = None,
) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=4,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        num_layers=2,
        devices=[DeviceRef.GPU()],
        page_size=128,
        kvcache_quant_config=kvcache_quant_config,
        window_size=window_size,
        stride_alignment=stride_alignment,
    )


def test_v_head_dim_defaults_to_head_dim() -> None:
    for params in (
        _params(head_dim=128),
        _params(head_dim=128, v_head_dim=128),
    ):
        assert params.v_head_dim is None
        assert not params.has_split_head_dims
        assert params.kv_dim == 2
        assert params.shape_per_block == [2, 2, 128, 4, 128]


def test_replace_head_dim_keeps_an_unsplit_cache_unsplit() -> None:
    """An unset value width follows ``head_dim``, so replacing ``head_dim``
    cannot turn a cache into a split one."""
    for params in (
        _params(head_dim=128),
        _params(head_dim=128, v_head_dim=128),
    ):
        wider = dataclasses.replace(params, head_dim=256)
        assert not wider.has_split_head_dims
        assert wider.kv_dim == 2
        assert wider.storage_head_dim == 256


def test_split_head_dims_pack_one_unpadded_row() -> None:
    params = _params(head_dim=192, v_head_dim=128)
    assert params.has_split_head_dims
    assert params.kv_dim == 1
    assert params.storage_head_dim == 320
    assert params.shape_per_block == [1, 2, 128, 4, 320]
    assert params.row_bytes == 4 * 320 * 2
    # Padding both to 256 stores 512 values per head and token.
    padded = _params(head_dim=256)
    assert params.bytes_per_block * 512 == padded.bytes_per_block * 320


def test_to_params_forwards_v_head_dim() -> None:
    params = KVCacheConfig().to_params(
        dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=192,
        v_head_dim=128,
        num_layers=2,
        devices=[DeviceRef.GPU()],
    )
    assert isinstance(params, MHAKVCacheParams)
    assert params.shape_per_block[0] == 1
    assert params.shape_per_block[-1] == 320


def test_v_head_dim_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _params(v_head_dim=0)


@pytest.mark.parametrize(
    ("dtype", "quant"),
    [
        (
            DType.float8_e4m3fn,
            KVCacheQuantizationConfig(quantization_granularity=64),
        ),
        (DType.float8_e4m3fn, None),
        (
            DType.uint8,
            KVCacheQuantizationConfig(
                scale_dtype=DType.float8_e4m3fn, quantization_granularity=16
            ),
        ),
    ],
    ids=["fp8-scaled", "fp8", "nvfp4"],
)
def test_split_head_dims_must_be_unquantized(
    dtype: DType, quant: KVCacheQuantizationConfig | None
) -> None:
    with pytest.raises(ValueError, match="must be unquantized"):
        _params(
            head_dim=192,
            v_head_dim=128,
            dtype=dtype,
            kvcache_quant_config=quant,
        )


def test_split_head_dims_stride_alignment_must_hold_whole_rows() -> None:
    # One row is 4 heads x (192 + 128) bf16 values = 2560 bytes.
    with pytest.raises(ValueError, match="multiple of a split KV cache row"):
        _params(head_dim=192, v_head_dim=128, stride_alignment=4096)
    params = _params(head_dim=192, v_head_dim=128, stride_alignment=5120)
    assert params.row_bytes == 2560
    assert _params(head_dim=128, stride_alignment=4096).stride_alignment == 4096


def test_mla_rejects_split_head_dims() -> None:
    with pytest.raises(ValueError, match="no separate value head"):
        MLAKVCacheParams(
            dtype=DType.bfloat16,
            head_dim=576,
            v_head_dim=512,
            num_layers=1,
            devices=[DeviceRef.GPU()],
            num_q_heads=128,
        )


def test_split_head_dims_graph_inputs_carry_the_layout() -> None:
    """A split cache's own graph inputs are one plane of rows as wide as the
    key and value together, and the values built from them are marked."""
    params = _params(head_dim=192, v_head_dim=128)
    (per_device,) = params.get_symbolic_inputs()
    assert per_device.split_head_dims == SplitKVHeadDims()
    assert [int(d) for d in per_device.kv_blocks.shape[1:]] == [
        1,
        2,
        128,
        4,
        320,
    ]
    with _cache_graph(params) as graph:
        assert _graph_cache(params, graph).split_head_dims == SplitKVHeadDims()


def _cache_graph(params: MHAKVCacheParams) -> Graph:
    """A graph whose inputs are ``params``' cache inputs."""
    return Graph(
        "split_head_dims", input_types=tree.leaves(params.get_symbolic_inputs())
    )


def _graph_cache(params: MHAKVCacheParams, graph: Graph) -> PagedCacheValues:
    """One device's cache inputs as graph values."""
    return params.unflatten_kv_inputs(iter(graph.inputs))[0]


def test_unsplit_cache_inputs_carry_no_marker() -> None:
    """An existing cache's input tree, and so its artifact form, is as it was."""
    params = _params(head_dim=128)
    for per_device in params.get_symbolic_inputs():
        assert per_device.split_head_dims is None
    _, treedef = tree.flatten(params.get_symbolic_inputs())
    assert "split_kv_head_dims" not in repr(treedef_to_json(treedef))


def test_split_marker_adds_no_leaves_and_round_trips() -> None:
    (unsplit,) = _params(head_dim=128).get_symbolic_inputs()
    split = dataclasses.replace(unsplit, split_head_dims=SplitKVHeadDims())

    leaves, treedef = tree.flatten(split)
    assert leaves == tree.leaves(unsplit)
    assert (
        split.flatten_without_attention_dispatch_metadata()
        == unsplit.flatten_without_attention_dispatch_metadata()
    )
    rebuilt = tree.unflatten(
        treedef_from_json(treedef_to_json(treedef)), leaves
    )
    assert rebuilt.split_head_dims == SplitKVHeadDims()
    assert tree.unflatten(tree.flatten(unsplit)[1], leaves) == unsplit


def _cache_params_of(fn: Callable[..., Any]) -> list[str]:
    return [
        name
        for name, param in inspect.signature(fn).parameters.items()
        if "PagedCacheValues" in str(param.annotation)
    ]


# Ops that read a split cache correctly.
_SPLIT_AWARE = frozenset(
    {
        "_validate_kv_cache_store_common",
        "flash_attention_ragged",
        "fused_chunked_qkv_rope_store_ragged",
        "fused_qk_ragged_rope",
        "kv_cache_store_paged_ragged",
        "store_k_cache_ragged",
        "store_v_cache_ragged",
    }
)

_CACHE_OPS = sorted(
    name
    for name, fn in inspect.getmembers(kernels, inspect.isfunction)
    if fn.__module__ == kernels.__name__ and _cache_params_of(fn)
)


def test_cache_ops_are_enumerated() -> None:
    """Guards the enumeration itself: a typo in the annotation match would
    silently parametrize nothing."""
    assert len(_CACHE_OPS) > 50
    assert "flash_attention_ragged" in _CACHE_OPS
    assert "kv_cache_store_paged_ragged" in _CACHE_OPS


@pytest.mark.parametrize(
    "name", [n for n in _CACHE_OPS if n not in _SPLIT_AWARE]
)
def test_cache_op_refuses_split_cache(name: str) -> None:
    """Every cache op must refuse a split cache before emitting anything.

    The ops infer MLA from a one-plane cache, so an op that does not know the
    layout would read the value columns as part of a latent. The guard runs
    before the op looks at its other arguments, so they are left ``None``.
    """
    fn = getattr(kernels, name)
    params = _params(head_dim=128)
    with _cache_graph(params) as graph:
        split = dataclasses.replace(
            _graph_cache(params, graph), split_head_dims=SplitKVHeadDims()
        )
        kwargs: dict[str, Any] = {
            param: None for param in inspect.signature(fn).parameters
        }
        for param in _cache_params_of(fn):
            kwargs[param] = split
        with pytest.raises(ValueError, match="head dims differ"):
            fn(**kwargs)


def _params_args_of(fn: Callable[..., Any]) -> list[str]:
    return [
        name
        for name, param in inspect.signature(fn).parameters.items()
        if "KVCacheParams" in str(param.annotation)
    ]


_PARAMS_OPS = [
    n
    for n in _CACHE_OPS
    if n not in _SPLIT_AWARE and _params_args_of(getattr(kernels, n))
]


def test_params_ops_are_enumerated() -> None:
    assert len(_PARAMS_OPS) > 30
    assert "fused_qkv_ragged_matmul" in _PARAMS_OPS


@pytest.mark.parametrize("name", _PARAMS_OPS)
def test_cache_op_refuses_split_params_with_an_unmarked_cache(
    name: str,
) -> None:
    """A collection built field by field can lose its marker; the split
    params it travels with must still be refused."""
    fn = getattr(kernels, name)
    params = _params(head_dim=128)
    split_params = _params(head_dim=192, v_head_dim=128)
    with _cache_graph(params) as graph:
        kwargs: dict[str, Any] = {
            param: None for param in inspect.signature(fn).parameters
        }
        for param in _cache_params_of(fn):
            kwargs[param] = _graph_cache(params, graph)
        for param in _params_args_of(fn):
            kwargs[param] = split_params
        with pytest.raises(ValueError, match="head dims differ"):
            fn(**kwargs)


def test_cache_op_accepts_unsplit_cache() -> None:
    """The guard keys on the marker, not on the cache being present."""
    params = _params(head_dim=128)
    with _cache_graph(params) as graph:
        kwargs: dict[str, Any] = {
            "kv_collection": _graph_cache(params, graph),
            "x_k": None,
            "input_row_offsets": None,
            "layer_idx": None,
        }
        with pytest.raises(Exception) as excinfo:
            kernels.store_k_cache_ragged(**kwargs)
        assert "head dims differ" not in str(excinfo.value)


def test_store_needs_v_head_dim_exactly_for_a_split_cache() -> None:
    params = _params(head_dim=128)
    with _cache_graph(params) as graph:
        cache = _graph_cache(params, graph)
        split = dataclasses.replace(cache, split_head_dims=SplitKVHeadDims())
        kwargs: dict[str, Any] = {
            "x_v": None,
            "input_row_offsets": None,
            "layer_idx": None,
        }
        with pytest.raises(ValueError, match="v_head_dim must be given"):
            kernels.store_v_cache_ragged(kv_collection=split, **kwargs)
        with pytest.raises(ValueError, match="v_head_dim must be given"):
            kernels.store_v_cache_ragged(
                kv_collection=cache, v_head_dim=128, **kwargs
            )


def test_rope_refuses_a_mismatched_or_positioned_split_cache() -> None:
    params = _params(head_dim=128)
    split_params = _params(head_dim=192, v_head_dim=128)
    with _cache_graph(params) as graph:
        cache = _graph_cache(params, graph)
        split = dataclasses.replace(cache, split_head_dims=SplitKVHeadDims())
        kwargs: dict[str, Any] = {
            "input": None,
            "input_row_offsets": None,
            "freqs_cis": None,
            "layer_idx": None,
        }
        with pytest.raises(ValueError, match="disagree"):
            kernels.fused_qk_ragged_rope(
                kv_params=params, kv_collection=split, **kwargs
            )
        with pytest.raises(ValueError, match="disagree"):
            kernels.fused_qk_ragged_rope(
                kv_params=split_params, kv_collection=cache, **kwargs
            )


def _chunked_qkv_graph(params: MHAKVCacheParams, chunk_width: int) -> Graph:
    """A graph with a chunked QKV projection, its row offsets and a RoPE
    table, then ``params``' cache inputs."""
    gpu = DeviceRef.GPU()
    return Graph(
        "split_head_dims_chunked_qkv",
        input_types=[
            TensorType(DType.bfloat16, ["total", 1, chunk_width], gpu),
            TensorType(DType.uint32, ["offsets"], gpu),
            TensorType(DType.bfloat16, [512, 64], gpu),
            *tree.leaves(params.get_symbolic_inputs()),
        ],
    )


def _chunked_qkv(
    graph: Graph, cache: PagedCacheValues, *, v_head_dim: int = 128
) -> TensorValue:
    qkv, offsets, freqs = (value.tensor for value in graph.inputs[:3])
    return kernels.fused_chunked_qkv_rope_store_ragged(
        qkv,
        offsets,
        freqs,
        cache,
        ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
        0.5,
        num_q_heads=16,
        qk_head_dim=192,
        v_head_dim=v_head_dim,
    )


def test_chunked_qkv_store_writes_a_split_cache_unpadded() -> None:
    """The fused store names the layout to the kernel, and its queries are
    as wide as the key rather than the row."""
    params = _params(head_dim=192, v_head_dim=128)
    # 16 query and 4 KV heads of 192, then 4 values of 128.
    with _chunked_qkv_graph(params, 16 * 192 + 4 * 320) as graph:
        cache = params.unflatten_kv_inputs(iter(graph.inputs[3:]))[0]
        q = _chunked_qkv(graph, cache)
        assert int(q.shape[-1]) == 192
        graph.output(q)
    assert "split_cache = true" in str(graph)


def test_chunked_qkv_store_on_an_unsplit_cache_is_unchanged() -> None:
    params = _params(head_dim=256)
    with _chunked_qkv_graph(params, 16 * 192 + 4 * 320) as graph:
        cache = params.unflatten_kv_inputs(iter(graph.inputs[3:]))[0]
        q = _chunked_qkv(graph, cache)
        assert int(q.shape[-1]) == 256
        graph.output(q)
    assert "split_cache" not in str(graph)


@pytest.mark.parametrize("case", ["unmarked", "v_width"])
def test_chunked_qkv_store_refuses_a_mismatched_split_cache(
    case: str,
) -> None:
    params = _params(head_dim=192, v_head_dim=128)
    with _chunked_qkv_graph(params, 16 * 192 + 4 * 312) as graph:
        cache = params.unflatten_kv_inputs(iter(graph.inputs[3:]))[0]
        if case == "unmarked":
            # One plane without the marker reads as an MLA latent.
            with pytest.raises(ValueError, match="separate K/V planes"):
                _chunked_qkv(
                    graph, dataclasses.replace(cache, split_head_dims=None)
                )
        else:
            with pytest.raises(ValueError, match="rows must hold"):
                _chunked_qkv(graph, cache, v_head_dim=120)


def _attention_graph(
    params: MHAKVCacheParams, q_head_dim: int
) -> tuple[Graph, int]:
    """A graph with ``params``' cache inputs, then a ragged query, its row
    offsets and a layer index; returns it and the index of the query input."""
    gpu = DeviceRef.GPU()
    cache_inputs = tree.leaves(params.get_symbolic_inputs())
    graph = Graph(
        "split_head_dims_attention",
        input_types=[
            *cache_inputs,
            TensorType(DType.bfloat16, ["total", 64, q_head_dim], gpu),
            TensorType(DType.uint32, ["offsets"], gpu),
            TensorType(DType.uint32, [], DeviceRef.CPU()),
        ],
    )
    return graph, len(cache_inputs)


def test_attention_reads_a_split_cache_into_a_value_wide_output() -> None:
    """The op names the value width and the output is that wide."""
    params = _params(head_dim=128)
    split_params = _params(head_dim=192, v_head_dim=128)
    graph, q_idx = _attention_graph(params, q_head_dim=192)
    with graph:
        split = dataclasses.replace(
            _graph_cache(params, graph), split_head_dims=SplitKVHeadDims()
        )
        out = kernels.flash_attention_ragged(
            split_params,
            graph.inputs[q_idx].tensor,
            graph.inputs[q_idx + 1].tensor,
            split,
            graph.inputs[q_idx + 2].tensor,
            MHAMaskVariant.CAUSAL_MASK,
            scale=192**-0.5,
        )
        assert list(out.shape) == list(graph.inputs[q_idx].tensor.shape)[
            :-1
        ] + [128]
        graph.output(out)
    assert "v_head_dim = 128" in str(graph)


def test_attention_reads_split_graph_inputs() -> None:
    """Attention takes a split cache straight from the params' graph inputs."""
    params = _params(head_dim=192, v_head_dim=128)
    graph, q_idx = _attention_graph(params, q_head_dim=192)
    with graph:
        out = kernels.flash_attention_ragged(
            params,
            graph.inputs[q_idx].tensor,
            graph.inputs[q_idx + 1].tensor,
            _graph_cache(params, graph),
            graph.inputs[q_idx + 2].tensor,
            MHAMaskVariant.CAUSAL_MASK,
            scale=192**-0.5,
        )
        assert int(out.shape[-1]) == 128
        graph.output(out)
    assert "v_head_dim = 128" in str(graph)


def test_attention_on_an_unsplit_cache_is_unchanged() -> None:
    params = _params(head_dim=128)
    graph, q_idx = _attention_graph(params, q_head_dim=128)
    with graph:
        out = kernels.flash_attention_ragged(
            params,
            graph.inputs[q_idx].tensor,
            graph.inputs[q_idx + 1].tensor,
            _graph_cache(params, graph),
            graph.inputs[q_idx + 2].tensor,
            MHAMaskVariant.CAUSAL_MASK,
            scale=128**-0.5,
        )
        assert out.shape == graph.inputs[q_idx].tensor.shape
        graph.output(out)
    assert "v_head_dim" not in str(graph)


@pytest.mark.parametrize("case", ["disagree", "q_width", "rel_logits"])
def test_attention_refuses_a_mismatched_split_cache(case: str) -> None:
    params = _params(head_dim=128)
    split_params = _params(head_dim=192, v_head_dim=128)
    graph, q_idx = _attention_graph(
        params, q_head_dim=128 if case == "q_width" else 192
    )
    with graph:
        cache = _graph_cache(params, graph)
        split = dataclasses.replace(cache, split_head_dims=SplitKVHeadDims())
        q = graph.inputs[q_idx].tensor
        kwargs: dict[str, Any] = {
            "kv_params": split_params,
            "input": q,
            "input_row_offsets": graph.inputs[q_idx + 1].tensor,
            "kv_collection": split,
            "layer_idx": graph.inputs[q_idx + 2].tensor,
            "mask_variant": MHAMaskVariant.CAUSAL_MASK,
            "scale": 1.0,
        }
        if case == "disagree":
            kwargs["kv_collection"] = cache
            match = "disagree"
        elif case == "q_width":
            match = "expected input head dim 192"
        else:
            kwargs["rel_logits"] = q
            match = "rel_logits is not supported"
        with pytest.raises(ValueError, match=match):
            kernels.flash_attention_ragged(**kwargs)


_PARTITION_TABLE = (
    Path(__file__).parent / "testdata/split_kv_decode_partitions.csv"
)


def _partition_table() -> list[dict[str, int]]:
    with open(_PARTITION_TABLE) as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        return [{k: int(v) for k, v in row.items()} for row in rows]


def test_split_kv_decode_partitions_match_the_shared_table() -> None:
    """The exported Mojo policy gives every count in the table Mach's port
    is also checked against."""
    rows = _partition_table()
    assert rows
    for row in rows:
        got = mha_split_kv_decode_num_partitions(
            row["batch_size"],
            row["num_keys"],
            row["kv_heads_per_device"],
            row["sliding_window"] or None,
            row["sm_count"],
        )
        assert got == row["num_partitions"], row


_SPLIT_K_LADDER = (2, 4, 6, 8, 10, 12, 16, 18, 20, 24, 32, 48, 64, 96)
_P_MAX = 160


def _ladder512(
    batch_size: int,
    num_keys: int,
    kv_heads_per_device: int,
    sliding_window: int,
    sm_count: int,
) -> int:
    visible_keys = min(num_keys, sliding_window or num_keys)
    by_wave = max(1, sm_count // (batch_size * kv_heads_per_device))
    want = min(visible_keys // 512, by_wave, _P_MAX)
    rungs = (*_SPLIT_K_LADDER, sm_count)
    return max((rung for rung in rungs if rung <= want), default=1)


@pytest.mark.parametrize("sm_count", [1, 20, 66, 132, 148, 159, 160, 264])
def test_split_kv_decode_partitions_serve_any_sm_count(sm_count: int) -> None:
    """A device the table does not model, such as a MIG slice, gets its own
    SM count's ladder; one with more SMs than the combine's ``P_MAX`` gets
    the ``P_MAX`` ladder."""
    policy_sm_count = min(sm_count, _P_MAX)
    for row in _partition_table():
        args = (
            row["batch_size"],
            row["num_keys"],
            row["kv_heads_per_device"],
            row["sliding_window"],
        )
        if row["sm_count"] == sm_count:
            assert _ladder512(*args, sm_count) == row["num_partitions"], row
        got = mha_split_kv_decode_num_partitions(
            *args[:3], args[3] or None, sm_count
        )
        assert got == _ladder512(*args, policy_sm_count), (row, sm_count)


def _decode_partitions(
    params: MHAKVCacheParams,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    max_cache_valid_length: int,
) -> int:
    """The decode key's partition count on a stand-in 148-SM device, with
    the legacy policy returning a sentinel 7."""
    params.__dict__["_primary_device"] = object()
    monkeypatch.setattr(
        cache_params_module, "multiprocessor_count", lambda _: 148
    )
    monkeypatch.setattr(
        cache_params_module, "mha_decode_num_partitions", lambda *_: 7
    )
    key = params.resolve_attn_key(batch_size, 1, max_cache_valid_length)
    assert isinstance(key, MHAAttnKey)
    return key.num_partitions


def test_split_cache_resolves_the_split_partition_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 4 KV heads on the one device: 148 // 4 = 37 caps the want at 32.
    full = _params(head_dim=192, v_head_dim=128)
    assert _decode_partitions(full, monkeypatch, 1, 131072) == 32
    assert _decode_partitions(full, monkeypatch, 1, 4096) == 8
    # A 128-key window never splits, however long the cache.
    swa = _params(head_dim=192, v_head_dim=128, window_size=128)
    assert _decode_partitions(swa, monkeypatch, 1, 262144) == 1


def test_unsplit_cache_keeps_the_legacy_partition_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _decode_partitions(_params(), monkeypatch, 1, 131072) == 7


def test_split_partition_policy_reads_the_sm_count_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _params(head_dim=192, v_head_dim=128)
    params.__dict__["_primary_device"] = object()
    queries: list[object] = []

    def count(device: object) -> int:
        queries.append(device)
        return 148

    monkeypatch.setattr(cache_params_module, "multiprocessor_count", count)
    for length in (4096, 131072, 262144):
        params.resolve_attn_key(1, 1, length)
    assert len(queries) == 1
