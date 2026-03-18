# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import json
from pathlib import Path

from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.lib.hf_utils import HuggingFaceRepo


def _write_sparse_gpt_oss_config(
    repo_dir: Path, *, include_model_type: bool = False
) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {
        "initial_context_length": 4096,
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_attention_heads": 64,
        "swiglu_limit": 7.0,
        "head_dim": 64,
        "rope_theta": 150000,
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "sliding_window": 128,
        "vocab_size": 201088,
        "experts_per_token": 4,
        "num_experts": 32,
    }
    if include_model_type:
        config["model_type"] = "gpt_oss"
    with (repo_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def test_registry_handles_sparse_local_gpt_oss_config(tmp_path: Path) -> None:
    repo_dir = tmp_path / "gpt-oss-20b-local"
    _write_sparse_gpt_oss_config(repo_dir)

    repo = HuggingFaceRepo(str(repo_dir))
    hf_config = PIPELINE_REGISTRY.get_active_huggingface_config(repo)
    arch = PIPELINE_REGISTRY.retrieve_architecture(repo)

    assert hf_config.model_type == "gpt_oss"
    assert hf_config.max_position_embeddings == 131072
    assert hf_config.num_local_experts == 32
    assert hf_config.num_experts_per_tok == 4
    assert arch is not None
    assert arch.name == "GptOssForCausalLM"


def test_registry_normalizes_sparse_gpt_oss_config_with_model_type(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "gpt-oss-20b-local-with-model-type"
    _write_sparse_gpt_oss_config(repo_dir, include_model_type=True)

    repo = HuggingFaceRepo(str(repo_dir))
    hf_config = PIPELINE_REGISTRY.get_active_huggingface_config(repo)
    arch = PIPELINE_REGISTRY.retrieve_architecture(repo)

    assert hf_config.model_type == "gpt_oss"
    assert hf_config.num_local_experts == 32
    assert hf_config.num_experts_per_tok == 4
    assert arch is not None
    assert arch.name == "GptOssForCausalLM"
