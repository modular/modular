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

"""Select vLLM and SGLang reasoning parsers from smoke-test configuration."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download
from requests.structures import CaseInsensitiveDict


@cache
def _load_reasoning_parsers() -> dict[str, dict[str, dict[str, str | None]]]:
    path = Path(__file__).with_name("reasoning_parsers.yaml")
    return yaml.safe_load(path.read_text())


def _hf_architectures(model: str) -> list[str]:
    """Reads checkpoint metadata, respecting ``HF_HUB_OFFLINE``."""
    # CI's cache populator pins refs/main to hf-repo-lock.tsv.
    with open(hf_hub_download(model, "config.json"), encoding="utf-8") as f:
        return json.load(f).get("architectures") or []


def get_reasoning_parser_arg(framework: str, model: str) -> str | None:
    """Selects an exact-model override, architecture default, or backend default."""
    if framework not in ("vllm", "sglang"):
        return None
    config = _load_reasoning_parsers()
    override = CaseInsensitiveDict(config["models"]).get(model, {})
    if framework in override:
        return override[framework]
    for architecture in _hf_architectures(model):
        parsers = config["architectures"].get(architecture, {})
        if framework in parsers:
            return parsers[framework]
    return "auto" if framework == "sglang" else None
