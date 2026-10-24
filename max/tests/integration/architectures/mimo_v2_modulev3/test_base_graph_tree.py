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
"""The base (spec-off) graph's input tree, as Mach's registry binds it.

Mach wires the base MEF by position through the registry entry's slot map,
and the ModuleV3 base graph does not keep the graph-API build's tree: it
takes the ModuleV3 ragged ABI — one shared row-offsets input, no signal
buffer in the template — and a multi-device trace appends one signal buffer
per device past every other input, where the graph-API base declares
per-device row offsets with its signal buffers between them and the KV
tree. The ``-modulev3`` registry entry therefore carries its own slot map
rather than the V2 entry's, and this test pins the tree that map describes,
slot for slot, so the two cannot drift apart silently (DISTINF-791).

The graph is staged on virtual B200s over the tiny fixture and never
compiled, so the test needs no GPU — the same no-GPU lane the DFlash spec
graph's signature test
(``mach/tools/gen-mef/test_mimo_dflash_graph_signature.py``) uses, whose
inserted signal buffers match a real TP2 export's sidecar slot for slot
and so prove the trace mirrors what an export records.
"""

from __future__ import annotations

import json

import pytest
import tiny_fixture
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
from max.graph import BufferType, DeviceRef, StaticDim, TensorType
from max.graph.weights import SafetensorWeights
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import MiMoV2
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    MiMoV2Config,
)
from max.pipelines.architectures.mimo_v2_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from mimo_v3_runner import kv_params
from transformers.configuration_utils import PretrainedConfig

SIGNAL_BYTES = 294_387_728
"""One signal buffer's bytes, as ``Signals.allocate`` sizes it."""


@pytest.fixture(scope="module", params=[1, 2])
def staged(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> tuple[tuple[TensorType | BufferType, ...], int]:
    """Stages the base graph's input types on ``request.param`` virtual GPUs.

    The module builds under the contexts the serving path's ``graph_specs``
    gives it and stages through ``Module.stage`` with the symbolic inputs
    ``_get_compile_input_types`` derives — the same trace an export runs.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    n_devices = request.param
    set_virtual_device_count(n_devices)

    path = tmp_path_factory.mktemp(f"mimo_v3_tree_{n_devices}")
    tiny_fixture.write(path, tiny_fixture.tensors())
    hf = PretrainedConfig(**json.loads((path / "config.json").read_text()))
    # The mesh holds the virtual accelerators; the config and the symbolic
    # inputs hold their device refs, as the serving path splits them.
    devices = [Accelerator(i) for i in range(n_devices)]
    refs = [DeviceRef.GPU(i) for i in range(n_devices)]
    config = MiMoV2Config.from_huggingface_config(
        hf,
        devices=refs,
        kv_params=kv_params(hf, refs, 128),
        max_seq_len=256,
    )
    config.return_logits = ReturnLogits.ALL

    weights = SafetensorWeights(sorted(path.glob("*.safetensors")))
    adapted = convert_safetensor_state_dict(
        dict(weights.items()), huggingface_config=hf
    )

    mesh = DeviceMesh(tuple(devices), (n_devices,), ("tp",))
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2(config)
        inputs = modulev3_ragged_kv_symbolic_inputs(
            kv_params=config.kv_params, device_refs=refs
        )
    # A fresh lazy context for the stage, as the CPU harness does: sharing
    # the build's would leave the module's pending weight-prep tensors in
    # the context, which a lazy context refuses to realize.
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        staged, _registry = model.stage(*inputs, weights=adapted)
    return (
        tuple(
            value.type
            for value in staged.graph.inputs
            if isinstance(value.type, (TensorType, BufferType))
        ),
        n_devices,
    )


def _kv_block(
    types: tuple[TensorType | BufferType, ...],
    slot: int,
    layers: int,
    heads: int,
) -> None:
    """Builds one KV group's ``kv_blocks`` input at ``slot``, tiny-fixture sized.

    A block is the writable cache pool
    ``[pages, arity, layers, page_size, heads, head_dim]``; the tiny
    fixture runs 2 sliding and 2 full layers over 8 and 4 KV heads, and
    every head is padded to the 256-wide cache.
    """
    block = types[slot]
    assert isinstance(block, BufferType), f"slot {slot} is not a cache pool"
    assert block.dtype == DType.bfloat16, f"slot {slot}: {block.dtype}"
    dims = [
        int(dim) if isinstance(dim, StaticDim) else None for dim in block.shape
    ]
    assert dims == [None, 2, layers, 128, heads, 256], (
        f"slot {slot}: {block.shape}"
    )


def test_the_tree_is_the_modulev3_abi(
    staged: tuple[tuple[TensorType | BufferType, ...], int],
) -> None:
    """Pins the tokens, logit count and ONE shared row-offsets input, then the
    two KV groups — 17 inputs at one device. No per-device row offsets and
    no signal buffers: the graph-API base tree's extras are absent, which is
    why the ``-modulev3`` registry entry binds its own slot map."""
    types, n_devices = staged
    tokens, return_n_logits, row_offsets = types[:3]
    assert isinstance(tokens, TensorType)
    assert tokens.dtype == DType.int64
    assert isinstance(return_n_logits, TensorType)
    assert return_n_logits.dtype == DType.int64
    assert isinstance(row_offsets, TensorType)
    assert row_offsets.dtype == DType.uint32
    assert tokens.device == DeviceRef.GPU(0)
    assert row_offsets.device == DeviceRef.GPU(0), "one shared input"

    if n_devices == 1:
        assert len(types) == 17
        _kv_block(types, 3, layers=2, heads=8)
        _kv_block(types, 10, layers=2, heads=4)
        # A single-device trace appends no signal buffers: no input is a
        # uint8 buffer the size of one.
        assert not [
            i
            for i, t in enumerate(types)
            if isinstance(t, BufferType)
            and t.dtype == DType.uint8
            and [dim for dim in t.shape] == [StaticDim(SIGNAL_BYTES)]
        ]


def test_a_multi_device_trace_appends_the_signals_past_the_kv_tree(
    staged: tuple[tuple[TensorType | BufferType, ...], int],
) -> None:
    """Checks that at two devices the KV groups widen device-major — 14 slots each, the
    second group opening at 17 — and the trace appends its signal buffers
    past the KV tree, one per device: the slots the engine wires for an
    entry on ``SIGNAL_BUFFERS_AFTER_INPUTS``."""
    types, n_devices = staged
    if n_devices != 2:
        pytest.skip("the trailing signal run is a multi-device property")
    assert len(types) == 33
    # Group 0's two device records, then group 1's at 17.
    _kv_block(types, 3, layers=2, heads=4)
    _kv_block(types, 10, layers=2, heads=4)
    _kv_block(types, 17, layers=2, heads=2)
    _kv_block(types, 24, layers=2, heads=2)
    signals = types[31:]
    assert len(signals) == 2
    for device, signal in enumerate(signals):
        assert isinstance(signal, BufferType), f"slot {31 + device}"
        assert signal.dtype == DType.uint8
        assert [dim for dim in signal.shape] == [StaticDim(SIGNAL_BYTES)]
        assert signal.device == DeviceRef.GPU(device)
