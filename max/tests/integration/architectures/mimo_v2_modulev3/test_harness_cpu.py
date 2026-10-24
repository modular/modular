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
"""The ModuleV3 numerics harness's no-GPU gate.

The GPU tests are the real gates, but everything they depend on that can be
checked without an accelerator is checked here, on the tiny fixture: the
harness's config build, the weight-adapter name contract (``Module.compile``
silently drops a checkpoint entry no parameter matches, so the contract has
to be pinned explicitly), the strict load, and the staged trace of the
whole graph — the same no-GPU verification lane (a) ran by hand, kept as a
test so every later change to the arch sees it.

The trace never executes, so the SM100 gates inside the kernel wrappers are
pinned to trace the shapes the B200 box will compile, as the nemotron-h
ModuleV3 test does.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import tiny_fixture
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device, default_dtype
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights, WeightData
from max.nn import kernels
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import MiMoV2
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    FULL,
    SLIDING,
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

N_DEVICES = 2


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny fixture checkpoint written to disk, as the GPU tests build it."""
    path = tmp_path_factory.mktemp("mimo_v3_cpu")
    tiny_fixture.write(path, tiny_fixture.tensors())
    return path


def _config(checkpoint: Path) -> MiMoV2Config:
    hf = PretrainedConfig(
        **json.loads((checkpoint / "config.json").read_text())
    )
    refs = [DeviceRef.CPU() for _ in range(N_DEVICES)]
    config = MiMoV2Config.from_huggingface_config(
        hf,
        devices=refs,
        kv_params=kv_params(hf, refs, 128),
        max_seq_len=256,
    )
    config.return_logits = ReturnLogits.ALL
    return config


def test_config_builds_from_the_tiny_fixture(checkpoint: Path) -> None:
    config = _config(checkpoint)
    assert config.layer_types == [FULL, SLIDING, SLIDING, FULL]
    assert config.moe_layers == {1, 2, 3}
    assert config.dtype == DType.bfloat16


def test_compile_inputs_follow_the_modulev3_ragged_abi(
    checkpoint: Path,
) -> None:
    """The ABI ``MiMoV2Inputs.buffers`` and the runner both rely on:
    ``(tokens, return_n_logits, row_offsets)`` on the first device, then the
    KV leaves — not the graph-API order, which puts the offsets first."""
    hf = PretrainedConfig(
        **json.loads((checkpoint / "config.json").read_text())
    )
    refs = [DeviceRef.CPU() for _ in range(N_DEVICES)]
    inputs = modulev3_ragged_kv_symbolic_inputs(
        kv_params=kv_params(hf, refs, 128), device_refs=refs
    )
    tokens, return_n_logits, row_offsets = inputs[:3]
    assert tokens.dtype == DType.int64
    assert return_n_logits.dtype == DType.int64
    assert return_n_logits.device == DeviceRef.CPU()
    assert row_offsets.dtype == DType.uint32
    assert len(inputs) > 3


@pytest.fixture(scope="module")
def adapted(
    checkpoint: Path,
) -> dict[str, WeightData]:
    """The adapter's output for the tiny checkpoint, on a 2-CPU mesh."""
    weights = SafetensorWeights(sorted(checkpoint.glob("*.safetensors")))
    hf = PretrainedConfig(
        **json.loads((checkpoint / "config.json").read_text())
    )
    return convert_safetensor_state_dict(
        dict(weights.items()), huggingface_config=hf
    )


def _mesh() -> DeviceMesh:
    """The CPU mesh standing in for the GPUs."""
    return DeviceMesh((CPU(),) * N_DEVICES, (N_DEVICES,), ("tp",))


def test_adapter_names_match_module_parameters(
    checkpoint: Path, adapted: dict[str, WeightData]
) -> None:
    """``Module.compile`` silently ignores a checkpoint entry no parameter
    matches and initializes the parameter to its default, so the contract is
    pinned by name; the strict load then pins shape and backing dtype
    (including the experts' E8M0 scale view)."""
    config = _config(checkpoint)
    mesh = _mesh()
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2(config)
        names = {name for name, _ in model.parameters}
        assert names == set(adapted), (
            f"module lacks {sorted(set(adapted) - names)[:8]}; no parameter "
            f"matches {sorted(names - set(adapted))[:8]}"
        )
        model.load_state_dict(adapted, strict=True)


def test_staged_trace_has_one_shared_moe_subgraph(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
    adapted: dict[str, WeightData],
) -> None:
    """The whole graph stages without an accelerator: the sharding rules,
    the wrapped quant ops, the per-device raw attention kernels and the
    shared MoE subgraph all trace. The 3 MoE layers share one body and call
    it once each; layer 0's dense MLP stays inline."""
    # The trace never runs, so it needs no GPU; the pin makes the kernel
    # wrappers trace the SM100 shape paths the B200 box will compile.
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)
    config = _config(checkpoint)
    mesh = _mesh()
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2(config)
        inputs = modulev3_ragged_kv_symbolic_inputs(
            kv_params=config.kv_params,
            device_refs=[DeviceRef.CPU() for _ in range(N_DEVICES)],
        )
    # A fresh lazy context for the stage: sharing the build's leaves the
    # module's pending weight-prep tensors in the context, and realizing
    # them mid-context is refused.
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        # Unpack before printing: rendering the registry's tensors would
        # realize them, which a lazy context refuses.
        staged, _registry = model.stage(*inputs, weights=adapted)
        text = str(staged)
    assert text.count("mo.graph @mimo_v2_moe") == 1
    assert text.count("mo.call @mimo_v2_moe") == len(config.moe_layers)
    # The MoE body is the only shared subgraph: layer 0's dense MLP and every
    # attention layer run inline in the main graph. Pin the subgraph set so
    # extracting another body must update this consciously, not slip past it.
    graphs = set(re.findall(r"mo\.graph @([A-Za-z0-9_]+)", text))
    assert graphs - {"MiMoV2"} == {"mimo_v2_moe"}, graphs
