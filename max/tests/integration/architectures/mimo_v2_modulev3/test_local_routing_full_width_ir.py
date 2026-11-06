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
"""Checks MiMo's allreduce fusion against the graphs the GPU test compiles.

The GPU replica test stages its trace with real weights. These checks compile
that trace at MiMo-V2.6's full width, where fusion can differ from the tiny
fixture's, and load it at the fixture's width, which is the cheap end of the
same check. The full-width compiles take minutes on a CPU, so the target runs
postsubmit; the fixture-width forward and verify graphs stay in presubmit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tiny_fixture
from max import functional as F
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)

set_virtual_device_api("cuda")
set_virtual_device_target_arch("sm_100a")
set_virtual_device_count(4)

from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.nn import _graph_ops as kernels
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import MiMoV2
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    MiMoV2Config,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.executor import CompilingExecutor, set_default_executor
from max.tensor.sharding import DeviceMesh
from mimo_v3_runner import kv_params
from replica_probe import (
    ALLREDUCES_PER_LAYER,
    MiMoReplicaTrace,
    allreduce_ops,
    assert_allreduce_form,
    first_four_layer_config,
    fused_allreduces,
    stage_replica_trace,
)
from transformers.configuration_utils import PretrainedConfig

NUM_DEVICES = 4


def _device_mesh() -> DeviceMesh:
    return DeviceMesh(
        tuple(Accelerator(i) for i in range(NUM_DEVICES)),
        (NUM_DEVICES,),
        ("tp",),
    )


def _full_width_model(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MiMoV2, MiMoV2Config]:
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)
    hf = first_four_layer_config()
    # Production shards the vocabulary, which a one-row table cannot split.
    hf.vocab_size = 1024
    refs = [DeviceRef.GPU(i) for i in range(NUM_DEVICES)]
    config = MiMoV2Config.from_huggingface_config(
        hf,
        devices=refs,
        kv_params=kv_params(hf, refs, 128),
        max_seq_len=256,
    )
    with F.lazy(), default_device(_device_mesh()), default_dtype(config.dtype):
        model = MiMoV2(config)
    assert model.route_on_every_device
    assert config.hidden_size == 4096
    return model, config


def _load_post_fusion_ir(
    graph: Graph, registry: dict[str, Tensor], ir_dir: Path
) -> str:
    session = InferenceSession(devices=[CPU(), *_device_mesh().devices])
    session.debug.ir_output_dir = str(ir_dir)
    session.load(graph, weights_registry=registry)
    return "\n".join(
        path.read_text() for path in sorted(ir_dir.glob("*.mo-pre-map.mlir"))
    )


def _moe_attention_allreduces(config: MiMoV2Config) -> list[int]:
    return [ALLREDUCES_PER_LAYER * layer for layer in sorted(config.moe_layers)]


def test_full_width_production_fuses_only_moe_attention_allreduces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins which of production's full-width allreduces fuse, per op index.

    The dense layer's norm feeds the MLP's 1x128 FP8 quantize, which the
    composite cannot take, so only the MoE layers' attention reductions
    fuse. The replica trace test compares against the same indices.
    """
    model, config = _full_width_model(monkeypatch)
    # The values are never read, so every weight of one shape and dtype
    # shares a single zero buffer.
    zeros: dict[tuple[tuple[int, ...], DType], Buffer] = {}
    weights: dict[str, Buffer] = {}
    for name, parameter in model.parameters:
        shape = tuple(int(dim) for dim in parameter.shape)
        key = (shape, parameter.dtype)
        if key not in zeros:
            raw = np.zeros(shape, f"u{parameter.dtype.size_in_bytes}")
            zeros[key] = Buffer.from_numpy(raw).view(
                parameter.dtype, list(shape)
            )
        weights[name] = zeros[key]
    staged, registry = model.stage(
        *modulev3_ragged_kv_symbolic_inputs(
            kv_params=config.kv_params, device_refs=list(config.devices)
        ),
        weights=weights,
    )
    ir = _load_post_fusion_ir(staged.graph, registry, tmp_path)
    assert fused_allreduces(ir) == _moe_attention_allreduces(config)


def test_full_width_replica_trace_fuses_as_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compiles the GPU test's full-width trace with its weights.

    The trace's extra outputs and planted factors must not move a fusion:
    the same allreduces fuse as in production, op by op.
    """
    model, config = _full_width_model(monkeypatch)
    staged, registry = stage_replica_trace(
        MiMoReplicaTrace(model), list(config.devices)
    )
    assert all(value.real for value in registry.values())
    ir = _load_post_fusion_ir(staged.graph, registry, tmp_path)
    ops = allreduce_ops(ir)
    assert len(ops) == ALLREDUCES_PER_LAYER * len(config.layer_types), ops
    assert fused_allreduces(ir) == _moe_attention_allreduces(config)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny fixture checkpoint, as the presubmit checks build it."""
    path = tmp_path_factory.mktemp("mimo_replica_trace")
    tiny_fixture.write(path, tiny_fixture.tensors())
    return path


def _tiny_fixture_model(
    hf: PretrainedConfig, monkeypatch: pytest.MonkeyPatch
) -> tuple[MiMoV2, MiMoV2Config]:
    """``hf``'s four layers on this module's mesh, without compiling."""
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)
    refs = [DeviceRef.GPU(i) for i in range(NUM_DEVICES)]
    config = MiMoV2Config.from_huggingface_config(
        hf,
        devices=refs,
        kv_params=kv_params(hf, refs, 128),
        max_seq_len=256,
    )
    with F.lazy(), default_device(_device_mesh()), default_dtype(config.dtype):
        model = MiMoV2(config)
    assert model.route_on_every_device
    return model, config


def test_replica_trace_loads_and_preserves_production_fusion(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loads the trace with the GPU test's weights and keeps its fusion.

    The fixture's width is the cheap end of the check: the full-width test
    above compiles the same trace at MiMo-V2.6's dimensions. ``session.init``
    reads each registry entry through DLPack, which realizes an unrealized
    one, and a load on virtual devices reads none: every entry must already
    be realized host data, or an entry left as one of the model's own
    parameters fails that assertion.
    """
    hf = PretrainedConfig(
        **json.loads((checkpoint / "config.json").read_text())
    )
    with set_default_executor(CompilingExecutor()):
        model, config = _tiny_fixture_model(hf, monkeypatch)
        staged, registry = stage_replica_trace(
            MiMoReplicaTrace(model), list(config.devices)
        )
        assert all(value.real for value in registry.values())
        ir = _load_post_fusion_ir(staged.graph, registry, tmp_path)
    ops = allreduce_ops(ir)
    assert len(ops) == ALLREDUCES_PER_LAYER * len(config.layer_types), ops
    assert_allreduce_form(ir)
    assert fused_allreduces(ir) == _moe_attention_allreduces(config)


def test_replica_weights_stage_the_real_first_four_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GPU replica test's weight builder covers every parameter of the
    full-width model with host data, without compiling it."""
    model, config = _tiny_fixture_model(first_four_layer_config(), monkeypatch)
    assert config.hidden_size == 4096
    staged, registry = stage_replica_trace(
        MiMoReplicaTrace(model), list(config.devices)
    )
    assert registry
    assert all(value.real for value in registry.values())
    assert str(staged).count("mo.call @mimo_v2_moe") == len(config.moe_layers)
