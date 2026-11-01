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
"""CPU-only snapshot of the vision metadata each model exports.

Mach lays out a model's media from the ``vision_config.json`` its artifact
carries rather than from its model registry, and its tests check each entry of
``testdata/vision_metadata.json``, one model's file, against the layout
profiles its router shares. Regenerate it deliberately with::

    UPDATE_VISION_METADATA_SNAPSHOT=1 ./bazelw test \
        //max/tests/tests/pipelines:test_vision_metadata \
        --test_env=UPDATE_VISION_METADATA_SNAPSHOT --strategy=TestRunner=local
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from max._artifact.value_types import ArtifactFile
from max._artifact.vision import VisionMetadata
from max.pipelines.architectures.gemma4.image_processor import (
    Gemma4ImageProcessor,
)
from max.pipelines.architectures.gemma4.model import gemma4_vision_metadata
from max.pipelines.architectures.gemma4.video_processor import (
    Gemma4VideoProcessor,
)
from max.pipelines.architectures.qwen3_5.model import qwen3_5_vision_metadata

_SNAPSHOT = Path(__file__).parent / "testdata" / "vision_metadata.json"


def _config(**attrs: object) -> Any:
    return SimpleNamespace(**attrs)


def _gemma4(config: Any) -> VisionMetadata | None:
    return gemma4_vision_metadata(
        config, Gemma4ImageProcessor(), Gemma4VideoProcessor()
    )


def _gemma4_31b() -> VisionMetadata | None:
    """The ids `google/gemma-4-31B-it`'s `config.json` states, at the
    processors' default budgets."""
    return _gemma4(
        _config(
            vision_config=_config(patch_size=16, pooling_kernel_size=3),
            image_token_id=258_880,
            video_token_id=258_884,
            boi_token_id=255_999,
            eoi_token_id=258_882,
        )
    )


def _qwen3_5() -> VisionMetadata | None:
    """The ids `RadixArk/Qwen3.8-27B-NVFP4`'s `config.json` states."""
    return qwen3_5_vision_metadata(
        _config(
            vision_config=_config(
                patch_size=16, spatial_merge_size=2, temporal_patch_size=2
            ),
            image_token_id=248_056,
            video_token_id=248_057,
            vision_start_token_id=248_053,
            vision_end_token_id=248_054,
        )
    )


_CASES: dict[str, Callable[[], VisionMetadata | None]] = {
    "gemma4": _gemma4_31b,
    "qwen3_5": _qwen3_5,
}


def _encode(name: str) -> object:
    """Returns the decoded ``vision_config.json`` the case writes."""
    metadata = _CASES[name]()
    assert metadata is not None
    file = metadata.__serialize__()["config"]
    assert isinstance(file, ArtifactFile)
    assert file.name == "vision_config.json"
    assert isinstance(file.content, str)
    return json.loads(file.content)


def test_snapshot_covers_every_case() -> None:
    if os.environ.get("UPDATE_VISION_METADATA_SNAPSHOT"):
        cases = {name: _encode(name) for name in _CASES}
        _SNAPSHOT.resolve().write_text(json.dumps(cases, indent=2) + "\n")
    assert set(json.loads(_SNAPSHOT.read_text())) == set(_CASES)


@pytest.mark.parametrize("name", list(_CASES))
def test_vision_metadata_matches_snapshot(name: str) -> None:
    assert _encode(name) == json.loads(_SNAPSHOT.read_text())[name]


@pytest.mark.parametrize("name", list(_CASES))
def test_vision_metadata_round_trips(name: str, tmp_path: Path) -> None:
    metadata = _CASES[name]()
    assert metadata is not None
    serialized = metadata.__serialize__()
    assert VisionMetadata.__deserialize__(serialized) == metadata
    # Read from an artifact, the file's content is its path.
    file = serialized["config"]
    assert isinstance(file, ArtifactFile)
    path = tmp_path / file.name
    file.write_to(path)
    stored = {"config": ArtifactFile(file.name, path)}
    assert VisionMetadata.__deserialize__(stored) == metadata


def test_a_text_only_checkpoint_has_no_vision_metadata() -> None:
    assert _gemma4(_config(vision_config=None)) is None
    assert qwen3_5_vision_metadata(_config()) is None
