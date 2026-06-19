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
"""Hermetic guard that the pipeline_to_mo_graph CLI builds its config via PipelineConfig.from_flat_kwargs."""

import json
from pathlib import Path

import pipeline_to_mo_graph as tool
import pytest
from click.testing import CliRunner
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig
from test_common.pipeline_model_dummy import DUMMY_LLAMA_ARCH
from test_common.registry import prepare_registry

_LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "num_hidden_layers": 2,
    "rope_theta": 10000.0,
    "max_position_embeddings": 2048,
    "intermediate_size": 11008,
    "vocab_size": 32000,
    "rms_norm_eps": 1e-5,
}


@prepare_registry
def test_main_builds_config_via_from_flat_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main routes the flat CLI flags through PipelineConfig.from_flat_kwargs.

    Registering only the dummy arch and pointing --model at a local repo keeps
    config construction off the network and the real architecture modules.
    --max-batch-size lives in a sub-config, so it reaches the resolved config
    only when main builds it via from_flat_kwargs; the raw PipelineConfig
    constructor rejects it as an extra input before retrieve is called.
    """
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
    (tmp_path / "config.json").write_text(json.dumps(_LLAMA_CONFIG))
    out_dir = tmp_path / "graphs"

    captured: dict[str, PipelineConfig] = {}

    def fake_retrieve(
        config: PipelineConfig, *args: object, **kwargs: object
    ) -> None:
        captured["config"] = config
        # Stand in for the graphs the tool would dump so main's final check
        # that something was written passes.
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "graph.mo.mlir").write_text("module {}")

    monkeypatch.setattr(PIPELINE_REGISTRY, "retrieve", fake_retrieve)

    result = CliRunner().invoke(
        tool.main,
        [
            "--model",
            str(tmp_path),
            "--devices",
            "gpu:0",
            "--target",
            "cuda:sm_100",
            "--build-only",
            "--max-batch-size",
            "2",
            "--output-dir",
            str(out_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].runtime.max_batch_size == 2
