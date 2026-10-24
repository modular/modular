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
"""CPU-only snapshot of each model's graph input tree.

The compiled graphs, max serve, and Mach's BatchProcessor depend on the exact
order, names, kinds and types of the inputs each model declares. The snapshot
in ``testdata/graph_input_trees.json`` records each tree as a role's
``in_layouts`` and ``in_tree`` in ``artifact.json``, and
``testdata/graph_group_kinds.json`` every group kind; Mach's planner tests
read both. A change to any model's inputs fails here until the snapshot is
regenerated deliberately with::

    UPDATE_GRAPH_INPUT_SNAPSHOT=1 ./bazelw test \
        //max/tests/tests/pipelines:test_graph_input_types \
        --test_env=UPDATE_GRAPH_INPUT_SNAPSHOT --strategy=TestRunner=local

Each model's input method reads only a few attributes of ``self``, so these
tests call it on a namespace holding just those, rather than building a model.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from max import tree
from max._artifact.export import per_step_dims
from max._artifact.value_types import treedef_from_json, treedef_to_json
from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.experimental.tensor_spec_serialization import (
    layout_to_json,
    layouts_from_json,
)
from max.graph import BufferType, DeviceRef, TensorType
from max.nn import graph_groups
from max.nn.comm.ep import EPBatchManager, EPConfig
from max.nn.graph_groups import RaggedBatch
from max.nn.kv_cache import MHAKVCacheParams, max_pages_per_request
from max.nn.kv_cache.cache_params import (
    KVCacheParamInterface,
    MultiKVCacheParams,
    RecurrentStateParams,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.deepseekV3.deepseekV3 import DeepseekV3
from max.pipelines.architectures.deepseekV3_2.deepseekV3_2 import DeepseekV3_2
from max.pipelines.architectures.deepseekV4.deepseekV4 import DeepseekV4
from max.pipelines.architectures.gemma3.gemma3 import Gemma3
from max.pipelines.architectures.gemma4.batch_processor import (
    Gemma4BatchProcessor,
)
from max.pipelines.architectures.gemma4.model import Gemma3_MultiModalModel
from max.pipelines.architectures.gpt_oss.gpt_oss import GptOss
from max.pipelines.architectures.llama3.batch_processor import (
    Llama3BatchProcessor,
)
from max.pipelines.architectures.llama3.distributed_llama import (
    DistributedLlama3,
)
from max.pipelines.architectures.llama3.llama3 import Llama3
from max.pipelines.architectures.mimo_v2.mimo_v2 import MiMoV2
from max.pipelines.architectures.qwen3_5.batch_processor import (
    Qwen3_5BatchProcessor,
)
from max.pipelines.architectures.qwen3_5.qwen3_5 import Qwen3_5
from max.pipelines.architectures.qwen3_5.state_cache import (
    ATTN_CACHE_KEY,
    STATE_CACHE_KEY,
    linear_state_regions,
)
from max.pipelines.lib.graph_specs import (
    GraphRole,
    _check_inputs,
    _input_signature,
)
from max.pipelines.lib.interfaces import (
    BatchProcessorRuntime,
    PipelineModel,
    modulev3_ragged_kv_symbolic_input_tree,
)

_SNAPSHOT = Path(__file__).parent / "testdata" / "graph_input_trees.json"
_KINDS = Path(__file__).parent / "testdata" / "graph_group_kinds.json"
_HIDDEN = 64

_InputTypes = tuple[TensorType | BufferType, ...]
_InputTree = OrderedDict[str, object]


def _fake(**attrs: object) -> Any:
    """Stands in for a model or config that exposes only ``attrs``."""
    return SimpleNamespace(**attrs)


def _model(cls: Any, **attrs: object) -> Any:
    """Stands in for a ``cls`` instance that exposes only ``attrs``.

    ``input_types`` calls ``input_tree`` on ``self``, so the stand-in carries
    ``cls``'s own ``input_tree`` bound to it.
    """
    model = SimpleNamespace(**attrs)
    model.input_tree = partial(cls.input_tree, model)
    return model


def _bare(cls: type, **attrs: object) -> Any:
    """Builds ``cls`` without ``__init__``, holding just ``attrs``."""
    obj: Any = object.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


def _gpus(n: int) -> list[DeviceRef]:
    return [DeviceRef.GPU(i) for i in range(n)]


def _kv_params(devices: list[DeviceRef]) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=16,
        num_layers=2,
        devices=tuple(devices),
    )


def _gemma4_kv_params(devices: list[DeviceRef]) -> MultiKVCacheParams:
    """Gemma 4's two attention caches, whose heads differ in width."""
    full = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=32,
        num_layers=1,
        devices=tuple(devices),
    )
    return MultiKVCacheParams.from_params(
        {"sliding_attention": _kv_params(devices), "full_attention": full}
    )


def _qwen3_5_kv_params(devices: list[DeviceRef]) -> MultiKVCacheParams:
    """Qwen3.5's attention cache beside its linear-attention state pool."""
    state = RecurrentStateParams(
        regions=linear_state_regions(
            num_linear_layers=3,
            key_head_dim=16,
            num_key_heads=4,
            value_head_dim=16,
            num_value_heads=4,
            conv_kernel_dim=4,
            dtype=DType.float32,
            num_devices=len(devices),
        ),
        devices=devices,
    )
    return MultiKVCacheParams.from_params(
        {ATTN_CACHE_KEY: _kv_params(devices), STATE_CACHE_KEY: state}
    )


def _ep_config(*, eplb: bool = False) -> EPConfig:
    return EPConfig(
        dispatch_dtype=DType.bfloat16,
        combine_dtype=DType.bfloat16,
        hidden_size=_HIDDEN,
        top_k=2,
        n_experts=16,
        max_tokens_per_rank=32,
        n_gpus_per_node=2,
        n_nodes=1,
        eplb_enabled=eplb,
        num_moe_layers=3,
        max_replicas=2,
    )


def _llama3(
    devices: list[DeviceRef], *, hidden: bool = False
) -> tuple[_InputTree, _InputTypes]:
    config = _fake(devices=devices, dtype=DType.bfloat16, hidden_size=_HIDDEN)
    model = _model(Llama3, config=config)
    kv_params = _kv_params(devices)
    return (
        Llama3.input_tree(model, kv_params, needs_hidden_state_input=hidden),
        Llama3.input_types(model, kv_params, needs_hidden_state_input=hidden),
    )


def _distributed_llama3() -> tuple[_InputTree, _InputTypes]:
    devices = _gpus(2)
    model = _model(DistributedLlama3, config=_fake(devices=devices))
    kv_params = _kv_params(devices)
    return (
        DistributedLlama3.input_tree(model, kv_params),
        DistributedLlama3.input_types(model, kv_params),
    )


def _qwen3_5(
    *, vision: bool, mrope: bool, hybrid: bool = False
) -> tuple[_InputTree, _InputTypes]:
    devices = _gpus(2)
    model = _model(
        Qwen3_5,
        devices=devices,
        config=_fake(compute_dtype=DType.bfloat16, hidden_size=_HIDDEN),
        vision_encoder=object() if vision else None,
        mrope_enabled=mrope,
    )
    kv_params: KVCacheParamInterface = (
        _qwen3_5_kv_params(devices) if hybrid else _kv_params(devices)
    )
    return (
        Qwen3_5.input_tree(model, kv_params),
        Qwen3_5.input_types(model, kv_params),
    )


def _gemma4(*, two_caches: bool = False) -> tuple[_InputTree, _InputTypes]:
    # Gemma4 reads driver-device attributes (label, id), not DeviceRefs.
    model = _fake(
        devices=[_fake(label="gpu", id=i) for i in range(2)],
        kv_params=(_gemma4_kv_params if two_caches else _kv_params)(_gpus(2)),
    )
    config = _fake(
        unquantized_dtype=DType.bfloat16,
        text_config=_fake(hidden_size=_HIDDEN),
    )
    input_tree = Gemma3_MultiModalModel._language_model_input_tree(
        model, config
    )
    return input_tree, tuple(tree.leaves(input_tree))


def _deepseek(
    cls: Any, *, ep: bool, eplb: bool = False, profile: bool = False
) -> tuple[_InputTree, _InputTypes]:
    devices = _gpus(2)
    config = _fake(
        devices=devices,
        data_parallel_degree=2,
        ep_config=_ep_config(eplb=eplb) if ep else None,
        # V3.2 builds no profiling counters, so only V3 reads these.
        eplb_profile_enabled=profile,
        num_hidden_layers=3,
        n_routed_experts=16,
    )
    model = _model(cls, config=config)
    kv_params = _kv_params(devices)
    return cls.input_tree(model, kv_params), cls.input_types(model, kv_params)


def _text_model(cls: Any, **attrs: object) -> tuple[_InputTree, _InputTypes]:
    """A model whose ``input_tree`` takes no arguments."""
    input_tree = cls.input_tree(_model(cls, **attrs))
    return input_tree, tuple(tree.leaves(input_tree))


def _per_device_offsets(
    cls: Any, *, nested: bool = False, two_caches: bool = False
) -> tuple[_InputTree, _InputTypes]:
    """A text model whose devices each take their own row offsets.

    ``nested`` places the devices and KV parameters on a ``language_model``,
    and ``two_caches`` gives it a sliding and a full attention cache.
    """
    devices = _gpus(2)
    kv_params = (_gemma4_kv_params if two_caches else _kv_params)(devices)
    attrs = {"devices": devices, "kv_params": kv_params}
    if nested:
        return _text_model(cls, language_model=_fake(**attrs))
    return _text_model(cls, **attrs)


def _deepseekV4(devices: list[DeviceRef]) -> tuple[_InputTree, _InputTypes]:
    config = _fake(devices=devices, kv_params=_kv_params(devices))
    return _text_model(DeepseekV4, config=config)


_CASES: dict[str, Callable[[], tuple[_InputTree, _InputTypes]]] = {
    "llama3_single_device": lambda: _llama3(_gpus(1)),
    "llama3_eagle_draft": lambda: _llama3(_gpus(1), hidden=True),
    "llama3_multi_device": _distributed_llama3,
    "qwen3_5_text": lambda: _qwen3_5(vision=False, mrope=False),
    "qwen3_5_vision": lambda: _qwen3_5(vision=True, mrope=False),
    "qwen3_5_vision_mrope": lambda: _qwen3_5(vision=True, mrope=True),
    "qwen3_5_hybrid": lambda: _qwen3_5(vision=False, mrope=False, hybrid=True),
    "gemma4": _gemma4,
    "gemma4_two_caches": lambda: _gemma4(two_caches=True),
    "deepseekV3": lambda: _deepseek(DeepseekV3, ep=False),
    "deepseekV3_ep": lambda: _deepseek(DeepseekV3, ep=True),
    "deepseekV3_ep_eplb": lambda: _deepseek(DeepseekV3, ep=True, eplb=True),
    "deepseekV3_eplb_profile": lambda: _deepseek(
        DeepseekV3, ep=True, profile=True
    ),
    "deepseekV3_2": lambda: _deepseek(DeepseekV3_2, ep=False),
    "deepseekV3_2_ep": lambda: _deepseek(DeepseekV3_2, ep=True),
    "deepseekV3_2_ep_eplb_profile": lambda: _deepseek(
        DeepseekV3_2, ep=True, eplb=True, profile=True
    ),
    "gemma3": lambda: _per_device_offsets(Gemma3, nested=True),
    "gpt_oss": lambda: _per_device_offsets(GptOss, nested=True),
    "mimo_v2": lambda: _per_device_offsets(MiMoV2, two_caches=True),
    "deepseekV4_single_device": lambda: _deepseekV4(_gpus(1)),
    "deepseekV4_multi_device": lambda: _deepseekV4(_gpus(2)),
}


@pytest.fixture(scope="module", autouse=True)
def virtual_gpus() -> Iterator[None]:
    """Lets a layout name a GPU on the CPU-only workers."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_80")
    set_virtual_device_count(2)
    try:
        yield
    finally:
        set_virtual_device_count(0)


def _leaf_paths(input_tree: object) -> list[str]:
    return list(tree.flatten(input_tree)[1].leaf_paths)


def _encode(input_tree: _InputTree) -> dict[str, Any]:
    """Returns a decoder role's inputs as ``artifact.json`` records them."""
    layouts, treedef = _input_signature(input_tree)
    return {
        "in_layouts": [layout_to_json(layout) for layout in layouts],
        "in_tree": treedef_to_json(treedef),
    }


def _dumps(value: object, indent: int = 0) -> str:
    """Formats JSON with any value that fits in 100 columns on one line.

    Each layout stays on one line too, so a retyped input changes one line of
    the snapshot rather than the two dozen ``indent=2`` spreads it over.
    """
    flat = json.dumps(value)
    if (
        not isinstance(value, (dict, list))
        or not value
        or indent + len(flat) <= 100
        or (isinstance(value, dict) and "dtype" in value)
    ):
        return flat
    pad = " " * (indent + 2)
    if isinstance(value, list):
        items = [pad + _dumps(item, indent + 2) for item in value]
        open_, close = "[", "]"
    else:
        items = [
            f"{pad}{json.dumps(key)}: {_dumps(item, indent + 2)}"
            for key, item in value.items()
        ]
        open_, close = "{", "}"
    return f"{open_}\n" + ",\n".join(items) + f"\n{' ' * indent}{close}"


def _snapshot(name: str) -> dict[str, Any]:
    return json.loads(_SNAPSHOT.read_text())[name]


def test_snapshot_covers_every_case() -> None:
    if os.environ.get("UPDATE_GRAPH_INPUT_SNAPSHOT"):
        trees = {name: _encode(build()[0]) for name, build in _CASES.items()}
        _SNAPSHOT.resolve().write_text(_dumps(trees) + "\n")
    assert set(json.loads(_SNAPSHOT.read_text())) == set(_CASES)


def test_graph_group_kinds_match_snapshot() -> None:
    """Mach's planner reads the kinds too, so a new group fails its tests
    until the engine's kind table names it or declares it unserved."""
    groups = (getattr(graph_groups, name) for name in graph_groups.__all__)
    kinds = sorted(
        kind
        for group in groups
        if isinstance(group, type)
        and issubclass(group, graph_groups.GraphGroup)
        and (kind := group.kind) is not None
    )
    if os.environ.get("UPDATE_GRAPH_INPUT_SNAPSHOT"):
        _KINDS.resolve().write_text(json.dumps(kinds, indent=2) + "\n")
    assert json.loads(_KINDS.read_text()) == kinds


@pytest.mark.parametrize("name", list(_CASES))
def test_graph_input_tree_matches_snapshot(name: str) -> None:
    assert _encode(_CASES[name]()[0]) == _snapshot(name), (
        "A graph input was renamed, reordered or retyped. If that is "
        "intended, regenerate with UPDATE_GRAPH_INPUT_SNAPSHOT=1 ./bazelw test "
        "//max/tests/tests/pipelines:test_graph_input_types "
        "--test_env=UPDATE_GRAPH_INPUT_SNAPSHOT --strategy=TestRunner=local"
    )


@pytest.mark.parametrize("name", list(_CASES))
def test_input_types_are_the_input_tree_leaves(name: str) -> None:
    input_tree, input_types = _CASES[name]()
    assert tuple(tree.leaves(input_tree)) == input_types


@pytest.mark.parametrize("name", list(_CASES))
def test_every_decoder_input_is_in_a_kinded_group(name: str) -> None:
    _check_inputs(GraphRole.DECODER, _input_signature(_CASES[name]()[0])[1])


def test_a_decoder_input_outside_a_kinded_group_fails_the_load() -> None:
    input_tree, _ = _CASES["llama3_single_device"]()
    input_tree["position_ids"] = TensorType(
        DType.int64, shape=[3, "total_seq_len"], device=DeviceRef.CPU()
    )
    in_tree = _input_signature(input_tree)[1]
    with pytest.raises(ValueError, match=r"0\.position_ids"):
        _check_inputs(GraphRole.DECODER, in_tree)
    # Only the decoder is planned by kind.
    _check_inputs(GraphRole.VISION, in_tree)


def test_a_positional_decoder_needs_no_kinds() -> None:
    _, input_types = _CASES["llama3_single_device"]()
    _check_inputs(GraphRole.DECODER, _input_signature(input_types)[1])


@pytest.mark.parametrize("name", list(_CASES))
def test_graph_input_tree_round_trips(name: str) -> None:
    layouts, treedef = _input_signature(_CASES[name]()[0])
    encoded = _snapshot(name)
    assert treedef_from_json(encoded["in_tree"]) == treedef
    assert layouts_from_json(encoded["in_layouts"], "in_layouts") == layouts


def _runtime() -> BatchProcessorRuntime:
    return BatchProcessorRuntime(
        pipeline_config=_fake(runtime=_fake(vision_cache_utilization=0.1)),
        devices=[],
        return_logits=ReturnLogits.LAST_TOKEN,
        max_batch_size=8,
        max_seq_len=1000,
        max_batch_input_tokens=512,
        data_parallel_degree=2,
    )


@pytest.mark.parametrize("name", list(_CASES))
def test_every_per_step_dim_has_a_maximum(name: str) -> None:
    dims = per_step_dims(_input_signature(_CASES[name]()[0])[0])
    assert set(_runtime().max_dims(dims, _kv_params(_gpus(1)))) == set(dims)
    # KV pages and state pool rows are sized by their owners.
    assert not [d for d in dims if d.endswith(("total_num_pages", "_rows"))]


def test_max_dims_bound_the_step() -> None:
    dims = per_step_dims(_input_signature(_CASES["gemma4_two_caches"]()[0])[0])
    assert _runtime().max_dims(dims, _kv_params(_gpus(1))) == {
        "total_seq_len": 1024,
        "return_n_logits": 1,
        "input_row_offsets_len": 17,
        "num_image_tokens": 1024,
        "total_image_tokens": 1024,
        "replica_0_batch_size": 8,
        "replica_0_sliding_attention_max_num_pages": 8,
        "replica_0_full_attention_max_num_pages": 8,
    }


def test_max_dims_bound_pages_past_max_seq_len_by_the_draft_slack() -> None:
    runtime = replace(_runtime(), max_seq_len=1024)
    kv_params = _kv_params(_gpus(1))
    drafting = replace(kv_params, num_draft_tokens=2)
    dims = ["replica_0_max_num_pages"]
    assert runtime.max_dims(dims, kv_params) == {dims[0]: 8}
    assert runtime.max_dims(dims, drafting) == {
        dims[0]: max_pages_per_request(drafting, 1024)
    }
    assert max_pages_per_request(drafting, 1024) > 8


def test_max_dims_refuses_image_rows_without_the_vision_cache() -> None:
    runtime = replace(
        _runtime(),
        pipeline_config=_fake(runtime=_fake(vision_cache_utilization=0)),
    )
    with pytest.raises(ValueError, match="'num_image_tokens'"):
        runtime.max_dims(["num_image_tokens"], _kv_params(_gpus(1)))


def test_max_dims_refuses_an_unknown_dim() -> None:
    with pytest.raises(ValueError, match="'mystery'"):
        _runtime().max_dims(["mystery"], _kv_params(_gpus(1)))


def test_deepseekv3_2_inputs_equal_deepseekv3_inputs() -> None:
    """V3.2 reuses V3's input tree."""
    assert _snapshot("deepseekV3_2") == _snapshot("deepseekV3")
    assert _snapshot("deepseekV3_2_ep") == _snapshot("deepseekV3_ep")


def test_batch_processor_input_tree_is_none_until_bound() -> None:
    processor = _bare(Llama3BatchProcessor)
    assert processor.input_tree is None

    input_tree, _ = _CASES["llama3_single_device"]()
    processor.bind_input_tree(input_tree)
    assert processor.input_tree is input_tree


def test_pipeline_model_binds_graph_input_tree() -> None:
    class concrete(PipelineModel[Any, Any, Any]):
        pass

    concrete.__abstractmethods__ = frozenset()
    input_tree, _ = _CASES["llama3_single_device"]()

    processor = _bare(Llama3BatchProcessor)
    model = _bare(concrete, _batch_processor=processor)
    assert model._bind_graph_inputs(input_tree) is input_tree
    assert processor.input_tree is input_tree

    without_processor = _bare(concrete, _batch_processor=None)
    assert without_processor._bind_graph_inputs(input_tree) is input_tree


@pytest.mark.parametrize(
    ("name", "make_processor"),
    [
        ("qwen3_5_text", Qwen3_5BatchProcessor),
        ("qwen3_5_vision", Qwen3_5BatchProcessor),
        ("qwen3_5_vision_mrope", Qwen3_5BatchProcessor),
        ("gemma4", Gemma4BatchProcessor),
    ],
)
def test_bound_processor_symbolic_inputs_match_model_input_types(
    name: str, make_processor: type
) -> None:
    devices = _gpus(2)
    processor = _bare(make_processor)
    with pytest.raises(AssertionError):
        processor.get_symbolic_inputs(
            kv_params=_kv_params(devices), device_refs=devices
        )

    input_tree, input_types = _CASES[name]()
    processor.bind_input_tree(input_tree)
    inputs = processor.get_symbolic_inputs(
        kv_params=_kv_params(devices), device_refs=devices
    )
    assert tuple(inputs) == input_types


def test_ep_fetch_buffers_reads_the_input_tree_in_order() -> None:
    """Each EP buffer comes from the tree leaf that names it."""
    manager = EPBatchManager(_ep_config(eplb=True))
    paths = _leaf_paths(manager.input_tree())
    manager.fetch_buffers([_fake(buffer=p, tensor=p) for p in paths])

    assert manager._atomic_counters == [
        ["atomic_counters.0", "atomic_counters.1"],
        ["atomic_counters.2", "atomic_counters.3"],
    ]
    assert manager._send_buf_ptrs == ["send_buf_ptrs.0", "send_buf_ptrs.1"]
    assert manager._recv_buf_ptrs == ["recv_buf_ptrs.0", "recv_buf_ptrs.1"]
    assert manager._recv_count_ptrs == [
        "recv_count_ptrs.0",
        "recv_count_ptrs.1",
    ]
    assert manager._eplb_log2phy_per_device == {
        0: "eplb_log2phy.0",
        1: "eplb_log2phy.1",
    }
    assert manager._eplb_logcnt_per_device == {
        0: "eplb_logcnt.0",
        1: "eplb_logcnt.1",
    }


def test_modulev3_input_tree_orders_return_n_logits_before_offsets() -> None:
    """ModuleV3 ``forward`` takes ``(tokens, return_n_logits, row_offsets, *kv)``.

    No snapshot covers it, since the models that use it are not touched here,
    so the expected order is spelled out.
    """
    devices = _gpus(1)
    inputs = modulev3_ragged_kv_symbolic_input_tree(
        kv_params=_kv_params(devices), device_refs=devices
    )
    paths = _leaf_paths(inputs)
    assert paths[:3] == [
        "batch.tokens",
        "batch.return_n_logits",
        "batch.row_offsets",
    ]
    assert paths[3:] == [
        f"kv.0.{field}"
        for field in (
            "kv_blocks",
            "cache_lengths",
            "lookup_table",
            "max_prompt_length",
            "max_cache_length",
            "page_stride",
            "attention_dispatch_metadata",
        )
    ]
    batch = inputs["batch"]
    assert isinstance(batch, RaggedBatch)
    assert [
        (t.dtype, str(t.device))
        for t in batch.values()
        if isinstance(t, TensorType)
    ] == [
        (DType.int64, "gpu:0"),
        (DType.int64, "cpu:0"),
        (DType.uint32, "gpu:0"),
    ]
