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
"""CPU-only snapshot of each model's flat graph input list.

The compiled graphs, and so max serve, depend on the exact order and types of
the inputs each model declares. The snapshot in
``testdata/graph_input_types.json`` records them; a change to any model's
inputs fails here until the snapshot is regenerated deliberately with::

    UPDATE_GRAPH_INPUT_SNAPSHOT=1 ./bazelw test \
        //max/tests/tests/pipelines:test_graph_input_types \
        --test_env=UPDATE_GRAPH_INPUT_SNAPSHOT --strategy=TestRunner=local

``testdata/graph_input_names.json`` records the path of each leaf of every
model's named input tree, in flatten order, so a rename or reorder of inputs
that share a type also fails.

Each model's input method reads only a few attributes of ``self``, so these
tests call it on a namespace holding just those, rather than building a model.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from max import tree
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.comm.ep import EPBatchManager, EPConfig
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.deepseekV3.deepseekV3 import DeepseekV3
from max.pipelines.architectures.deepseekV3_2.deepseekV3_2 import DeepseekV3_2
from max.pipelines.architectures.gemma4.batch_processor import (
    Gemma4BatchProcessor,
)
from max.pipelines.architectures.gemma4.model import Gemma3_MultiModalModel
from max.pipelines.architectures.llama3.batch_processor import (
    Llama3BatchProcessor,
)
from max.pipelines.architectures.llama3.distributed_llama import (
    DistributedLlama3,
)
from max.pipelines.architectures.llama3.llama3 import Llama3
from max.pipelines.architectures.qwen3_5.batch_processor import (
    Qwen3_5BatchProcessor,
)
from max.pipelines.architectures.qwen3_5.qwen3_5 import Qwen3_5
from max.pipelines.lib.interfaces import (
    PipelineModel,
    modulev3_ragged_kv_symbolic_input_tree,
)

_SNAPSHOT = Path(__file__).parent / "testdata" / "graph_input_types.json"
_NAMES = Path(__file__).parent / "testdata" / "graph_input_names.json"
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


def _qwen3_5(*, vision: bool, mrope: bool) -> tuple[_InputTree, _InputTypes]:
    devices = _gpus(2)
    model = _model(
        Qwen3_5,
        devices=devices,
        config=_fake(compute_dtype=DType.bfloat16, hidden_size=_HIDDEN),
        vision_encoder=object() if vision else None,
        mrope_enabled=mrope,
    )
    kv_params = _kv_params(devices)
    return (
        Qwen3_5.input_tree(model, kv_params),
        Qwen3_5.input_types(model, kv_params),
    )


def _gemma4() -> tuple[_InputTree, _InputTypes]:
    # Gemma4 reads driver-device attributes (label, id), not DeviceRefs.
    model = _fake(
        devices=[_fake(label="gpu", id=i) for i in range(2)],
        kv_params=_kv_params(_gpus(2)),
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


_CASES: dict[str, Callable[[], tuple[_InputTree, _InputTypes]]] = {
    "llama3_single_device": lambda: _llama3(_gpus(1)),
    "llama3_eagle_draft": lambda: _llama3(_gpus(1), hidden=True),
    "llama3_multi_device": _distributed_llama3,
    "qwen3_5_text": lambda: _qwen3_5(vision=False, mrope=False),
    "qwen3_5_vision": lambda: _qwen3_5(vision=True, mrope=False),
    "qwen3_5_vision_mrope": lambda: _qwen3_5(vision=True, mrope=True),
    "gemma4": _gemma4,
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
}


def _describe(types: _InputTypes) -> list[str]:
    return [
        f"{type(t).__name__}({t.dtype}, {list(t.shape)}, {t.device})"
        for t in types
    ]


def _leaf_paths(input_tree: Mapping[str, object]) -> list[str]:
    return list(tree.flatten(input_tree)[1].leaf_paths)


def _regenerate_snapshots() -> None:
    types = {name: _describe(build()[1]) for name, build in _CASES.items()}
    names = {name: _leaf_paths(build()[0]) for name, build in _CASES.items()}
    _SNAPSHOT.resolve().write_text(json.dumps(types, indent=2) + "\n")
    _NAMES.resolve().write_text(json.dumps(names, indent=2) + "\n")


@pytest.mark.parametrize("name", list(_CASES))
def test_graph_input_types_match_snapshot(name: str) -> None:
    if os.environ.get("UPDATE_GRAPH_INPUT_SNAPSHOT"):
        _regenerate_snapshots()
    expected = json.loads(_SNAPSHOT.read_text())
    assert _describe(_CASES[name]()[1]) == expected[name]


@pytest.mark.parametrize("name", list(_CASES))
def test_input_types_are_the_input_tree_leaves(name: str) -> None:
    input_tree, input_types = _CASES[name]()
    assert tuple(tree.leaves(input_tree)) == input_types


def test_deepseekv3_2_inputs_equal_deepseekv3_inputs() -> None:
    """V3.2 reuses V3's input list, which ``main`` recorded for ``deepseekV3``."""
    snapshot = json.loads(_SNAPSHOT.read_text())
    assert snapshot["deepseekV3_2"] == snapshot["deepseekV3"]
    assert snapshot["deepseekV3_2_ep"] == snapshot["deepseekV3_ep"]


def test_names_golden_covers_every_snapshot_case() -> None:
    assert set(json.loads(_NAMES.read_text())) == set(_CASES)
    assert set(json.loads(_SNAPSHOT.read_text())) == set(_CASES)


@pytest.mark.parametrize("name", list(_CASES))
def test_graph_input_names_match_snapshot(name: str) -> None:
    expected = json.loads(_NAMES.read_text())[name]
    input_tree, _ = _CASES[name]()
    assert _leaf_paths(input_tree) == expected, (
        "A graph input was renamed or reordered. If that is intended, "
        "regenerate with UPDATE_GRAPH_INPUT_SNAPSHOT=1 ./bazelw test "
        "//max/tests/tests/pipelines:test_graph_input_types "
        "--test_env=UPDATE_GRAPH_INPUT_SNAPSHOT --strategy=TestRunner=local"
    )
    snapshot = json.loads(_SNAPSHOT.read_text())[name]
    assert _describe(tuple(tree.leaves(input_tree))) == snapshot


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
    input_tree, input_types = _CASES["llama3_single_device"]()

    processor = _bare(Llama3BatchProcessor)
    model = _bare(concrete, _batch_processor=processor)
    assert model._bind_graph_inputs(input_tree) == input_types
    assert processor.input_tree is input_tree

    without_processor = _bare(concrete, _batch_processor=None)
    assert without_processor._bind_graph_inputs(input_tree) == input_types


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

    processor.bind_input_tree(_CASES[name]()[0])
    inputs = processor.get_symbolic_inputs(
        kv_params=_kv_params(devices), device_refs=devices
    )
    assert _describe(tuple(inputs)) == json.loads(_SNAPSHOT.read_text())[name]


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
    assert paths[3:] == [f"kv.0.{i}" for i in range(7)]
    assert _describe(tuple(tree.leaves(inputs)))[:3] == [
        "TensorType(DType.int64, [Dim('total_seq_len')], gpu:0)",
        "TensorType(DType.int64, [Dim('return_n_logits')], cpu:0)",
        "TensorType(DType.uint32, [Dim('input_row_offsets_len')], gpu:0)",
    ]
