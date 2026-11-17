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
"""Pins the checkpoint prefetch a ModuleV3 decoder's load starts."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from max.engine import InferenceSession
from max.engine import Model as EngineModel
from max.pipelines.lib import GraphRole, ModuleV3PipelineModelWithKVCache, utils
from max.pipelines.lib.pipeline_variants import spec_decode_pipeline_model


def test_prefetch_checkpoint_reads_every_file_through_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every byte is read once, in chunks, including a file a symlink names
    (the Hugging Face cache's layout)."""
    monkeypatch.setattr(utils, "_PREFETCH_CHUNK", 1000)
    sizes = {"a.safetensors": 2500, "blob": 1000}
    for name, size in sizes.items():
        (tmp_path / name).write_bytes(bytes(size))
    (tmp_path / "b.safetensors").symlink_to(tmp_path / "blob")
    paths = [tmp_path / "a.safetensors", tmp_path / "b.safetensors"]
    assert utils.prefetch_checkpoint(paths) == sum(sizes.values())


_DECODER = MagicMock()


class _Model(ModuleV3PipelineModelWithKVCache[Any]):
    def load_graphs(self, session: InferenceSession) -> dict[str, EngineModel]:
        self.compiled_modules = {GraphRole.DECODER: _DECODER}
        self._checkpoint = {}
        return {}


def test_decoder_load_prefetches_the_weight_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading the decoder starts reading the model's weight files."""
    prefetched: queue.Queue[list[Path]] = queue.Queue()
    monkeypatch.setattr(utils, "prefetch_checkpoint", prefetched.put)
    paths = [Path("model-00001.safetensors"), Path("model-00002.safetensors")]
    monkeypatch.setattr(utils.os.path, "realpath", str)
    monkeypatch.setattr(utils, "is_virtual_device_mode", lambda: False)
    model = object.__new__(_Model)
    model.pipeline_config = MagicMock()
    model.pipeline_config.model.resolved_weight_paths.return_value = paths
    model.pipeline_config.model.force_download = False
    model.pipeline_config.model.header_only_weights = False

    assert model._load_decoder(MagicMock()) is _DECODER
    assert prefetched.get(timeout=30) == [str(path) for path in paths]


def test_prefetch_checkpoint_readers_do_not_hold_up_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its readers are daemon threads, so a process that stops early does
    not wait for the read to finish."""
    daemon: list[bool] = []

    class Recording(threading.Thread):
        def start(self) -> None:
            daemon.append(self.daemon)
            super().start()

    monkeypatch.setattr(threading, "Thread", Recording)
    (tmp_path / "a.safetensors").write_bytes(bytes(10))
    utils.prefetch_checkpoint([tmp_path / "a.safetensors"])
    assert daemon and all(daemon)


@pytest.mark.parametrize(
    "force_download, header_only, virtual, reaches",
    [
        (False, False, False, True),
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
    ],
)
def test_prefetch_only_when_weights_reach_the_devices(
    monkeypatch: pytest.MonkeyPatch,
    force_download: bool,
    header_only: bool,
    virtual: bool,
    reaches: bool,
) -> None:
    """A compile-only load never reads the weights, and a forced download
    has just written them through the page cache."""
    monkeypatch.setattr(utils, "is_virtual_device_mode", lambda: virtual)
    config = SimpleNamespace(
        force_download=force_download, header_only_weights=header_only
    )
    assert utils.weights_reach_devices(config) is reaches


def _config(*paths: Path) -> SimpleNamespace:
    return SimpleNamespace(
        force_download=False,
        header_only_weights=False,
        resolved_weight_paths=lambda: list(paths),
    )


def test_start_checkpoint_prefetch_reads_shared_files_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draft that shares the target's checkpoint adds no second read, and
    a missing draft is skipped."""
    prefetched: queue.Queue[list[str]] = queue.Queue()
    monkeypatch.setattr(utils, "prefetch_checkpoint", prefetched.put)
    monkeypatch.setattr(utils, "is_virtual_device_mode", lambda: False)
    shared, draft = tmp_path / "shared", tmp_path / "draft"
    (tmp_path / "link").symlink_to(shared)
    utils.start_checkpoint_prefetch(
        _config(shared), _config(tmp_path / "link", draft), None
    )
    assert prefetched.get(timeout=30) == [
        str(shared.resolve()),
        str(draft.resolve()),
    ]


class _Stop(Exception):
    pass


@pytest.mark.parametrize(
    "cls",
    [
        spec_decode_pipeline_model.SpecDecodePipelineModel,
        spec_decode_pipeline_model._SpecDecodeModuleV3PipelineModel,
    ],
)
def test_spec_decode_load_prefetches_target_and_draft(
    monkeypatch: pytest.MonkeyPatch, cls: type[Any]
) -> None:
    """Both spec-decode loads start reading the target's and the draft's
    weight files before building either."""
    started: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        spec_decode_pipeline_model,
        "start_checkpoint_prefetch",
        lambda *configs: started.append(configs),
    )
    monkeypatch.setattr(
        spec_decode_pipeline_model, "_stage_error", lambda config: None
    )

    def stop(self: object, kind: object) -> None:
        raise _Stop

    monkeypatch.setattr(cls, "arch_config_as", stop)
    model = object.__new__(cls)
    model.pipeline_config = MagicMock()
    with pytest.raises(_Stop):
        model.load_model(MagicMock())
    assert started == [
        (model.pipeline_config.model, model.pipeline_config.draft_model)
    ]
