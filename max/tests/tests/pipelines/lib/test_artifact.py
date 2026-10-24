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
"""Tests for the model artifact writer and reader."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import huggingface_hub
import pytest
from max import _core
from max._artifact import (
    ArtifactError,
    ArtifactFile,
    ArtifactType,
    HuggingFaceRef,
)
from max._artifact.batch_processor import BatchProcessorDims
from max._artifact.codec import encode, prefix_files
from max._artifact.metadata import ARTIFACT_VERSION
from max._artifact.model_artifact import ModelArtifact
from max._artifact.pipeline_modules import PipelineModules
from max._artifact.serialization import deserialize, serialize
from max._artifact.value_types import sha256_file
from typing_extensions import Self

MODEL_REVISION = "a" * 40
MODEL_REF = HuggingFaceRef("org/model", MODEL_REVISION)
DRAFT_REF = HuggingFaceRef("org/draft", "b" * 40)
TOKENIZER_REF = HuggingFaceRef("org/model", MODEL_REVISION, "tokenizer.json")


@pytest.fixture(autouse=True)
def hf_snapshots(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> dict[tuple[str, str], Path]:
    """Replaces ``huggingface_hub.snapshot_download`` with a fake local cache.

    The fake resolves only the snapshots in the returned mapping, keyed by
    ``(repo_id, revision)``, and never touches the network.
    """
    snapshots: dict[tuple[str, str], Path] = {}
    for ref in [MODEL_REF, DRAFT_REF, TOKENIZER_REF]:
        snapshot = tmp_path_factory.mktemp("snapshot")
        snapshot = snapshots.setdefault((ref.repo_id, ref.revision), snapshot)
        if ref.path is not None:
            (snapshot / ref.path).write_text("{}")

    def fake_snapshot_download(
        *, repo_id: str, revision: str, local_files_only: bool
    ) -> str:
        assert local_files_only
        snapshot = snapshots.get((repo_id, revision))
        if snapshot is None:
            raise huggingface_hub.errors.LocalEntryNotFoundError(
                f"{repo_id}@{revision} is not cached"
            )
        return str(snapshot)

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", fake_snapshot_download
    )
    return snapshots


@dataclass
class FakeComponent:
    """Stores its serialized form verbatim."""

    value: Mapping[str, ArtifactType] = field(default_factory=dict)

    def __serialize__(self) -> Mapping[str, ArtifactType]:
        return self.value

    @classmethod
    def __deserialize__(cls, value: Mapping[str, ArtifactType]) -> Self:
        return cls(value)


class FailingComponent(FakeComponent):
    def __serialize__(self) -> Mapping[str, ArtifactType]:
        raise RuntimeError("boom")

    @classmethod
    def __deserialize__(cls, value: Mapping[str, ArtifactType]) -> Self:
        raise RuntimeError("boom")


@dataclass(frozen=True)
class FakeArtifact(ModelArtifact):
    engine_config: FakeComponent | None = None
    tokenizer: FakeComponent | None = None


@dataclass(frozen=True)
class FailingArtifact(ModelArtifact):
    engine_config: FailingComponent | None = None


@dataclass(frozen=True)
class AmbiguousArtifact(ModelArtifact):
    engine_config: FakeComponent | FailingComponent | None = None


def _artifact(**components: FakeComponent) -> FakeArtifact:
    return FakeArtifact(
        engine_config=components.get("engine_config"),
        tokenizer=components.get("tokenizer"),
    )


def test_round_trip(tmp_path: Path) -> None:
    mef = tmp_path / "src.mef"
    mef.write_bytes(b"\x00mef")
    model = FakeComponent(
        {
            "engine_model": ArtifactFile("model.mef", mef),
            "graph": ArtifactFile("graph/graph.rmo.mlir", "module {}"),
            "blob": ArtifactFile("blob.bin", b"\x01\x02"),
            "nested": {"ints": [1, 2, (3, 4)], "f": 0.5, "b": True, "n": None},
            "sources": {"main": MODEL_REF, "draft": [DRAFT_REF, MODEL_REF]},
        }
    )
    tokenizer = FakeComponent({"files": TOKENIZER_REF})

    dest = serialize(
        _artifact(engine_config=model, tokenizer=tokenizer), tmp_path / "a"
    )
    assert dest == tmp_path / "a.m"
    assert (dest / "model.mef").read_bytes() == b"\x00mef"
    assert (dest / "graph/graph.rmo.mlir").read_text() == "module {}"
    assert (dest / "blob.bin").read_bytes() == b"\x01\x02"

    metadata = json.loads((dest / "metadata.json").read_text())
    assert metadata["model_sources"] == [
        {"repo_id": "org/draft", "revision": DRAFT_REF.revision, "path": None},
        {"repo_id": "org/model", "revision": MODEL_REVISION, "path": None},
        {
            "repo_id": "org/model",
            "revision": MODEL_REVISION,
            "path": "tokenizer.json",
        },
    ]

    loaded = deserialize(dest, FakeArtifact)
    assert isinstance(loaded, FakeArtifact)
    assert loaded.model_sources() == (
        DRAFT_REF,
        MODEL_REF,
        TOKENIZER_REF,
    )
    assert loaded.tokenizer == tokenizer
    assert loaded.format_spec is None
    assert isinstance(loaded.engine_config, FakeComponent)
    value = loaded.engine_config.value
    assert value["engine_model"] == ArtifactFile(
        "model.mef", dest / "model.mef"
    )
    assert value["blob"] == ArtifactFile("blob.bin", dest / "blob.bin")
    assert value["nested"] == {
        "ints": [1, 2, [3, 4]],
        "f": 0.5,
        "b": True,
        "n": None,
    }
    assert value["sources"] == {
        "main": MODEL_REF,
        "draft": [DRAFT_REF, MODEL_REF],
    }


def test_metadata(tmp_path: Path) -> None:
    dest = serialize(
        _artifact(
            engine_config=FakeComponent({"f": ArtifactFile("x.txt", "x")})
        ),
        tmp_path / "a.m",
    )
    assert dest == tmp_path / "a.m"
    metadata = json.loads((dest / "metadata.json").read_text())
    assert metadata["max_version"] == _core.__version__
    assert metadata["artifact_version"] == ARTIFACT_VERSION
    assert metadata["model_sources"] == []
    assert metadata["files"] == {
        "artifact.json": sha256_file(dest / "artifact.json"),
        "x.txt": sha256_file(dest / "x.txt"),
    }
    sections = json.loads((dest / "artifact.json").read_text())
    assert sections == {"engine_config": {"f": {"$file": "x.txt"}}}


def test_existing_destination_rejected(tmp_path: Path) -> None:
    serialize(_artifact(), tmp_path / "a")
    with pytest.raises(ArtifactError, match="already exists"):
        serialize(_artifact(), tmp_path / "a")


def test_failed_export_leaves_nothing(tmp_path: Path) -> None:
    ok = FakeComponent({"f": ArtifactFile("x.txt", "x")})
    bad = FakeComponent({"f": ArtifactFile("x.txt", Path("/nonexistent"))})
    with pytest.raises(ArtifactError):
        serialize(_artifact(engine_config=ok, tokenizer=bad), tmp_path / "a")
    with pytest.raises(ArtifactError, match="does not exist"):
        serialize(_artifact(engine_config=bad), tmp_path / "a")
    with pytest.raises(RuntimeError):
        serialize(_artifact(engine_config=FailingComponent()), tmp_path / "a")
    assert list(tmp_path.iterdir()) == []


def test_identical_files_deduplicated(tmp_path: Path) -> None:
    f = ArtifactFile("x.txt", "x")
    dest = serialize(
        _artifact(
            engine_config=FakeComponent({"f": f}),
            tokenizer=FakeComponent({"f": f}),
        ),
        tmp_path / "a",
    )
    assert (dest / "x.txt").read_text() == "x"


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda d: (d / "x.txt").write_text("tampered"),
        lambda d: (d / "x.txt").unlink(),
        lambda d: (d / "artifact.json").write_text("{}"),
        lambda d: (d / "metadata.json").unlink(),
    ],
)
def test_corrupt_artifact_rejected(
    tmp_path: Path, corrupt: Callable[[Path], object]
) -> None:
    dest = serialize(
        _artifact(
            engine_config=FakeComponent({"f": ArtifactFile("x.txt", "x")})
        ),
        tmp_path / "a",
    )
    corrupt(dest)
    with pytest.raises(ArtifactError):
        deserialize(dest, FakeArtifact)


def test_requires_m_suffix(tmp_path: Path) -> None:
    dest = serialize(_artifact(), tmp_path / "a")
    renamed = dest.rename(tmp_path / "a.dir")
    with pytest.raises(ArtifactError, match="not a model artifact"):
        deserialize(renamed, FakeArtifact)
    with pytest.raises(ArtifactError, match="does not exist"):
        deserialize(tmp_path / "missing.m", FakeArtifact)


def test_missing_component_type(tmp_path: Path) -> None:
    dest = serialize(_artifact(engine_config=FakeComponent()), tmp_path / "a")
    with pytest.raises(ArtifactError, match="no concrete component type"):
        deserialize(dest)


def test_component_deserialize_error_wrapped(tmp_path: Path) -> None:
    dest = serialize(_artifact(engine_config=FakeComponent()), tmp_path / "a")
    with pytest.raises(ArtifactError, match="re-export"):
        deserialize(dest, FailingArtifact)


def test_unknown_slot_rejected(tmp_path: Path) -> None:
    dest = serialize(_artifact(), tmp_path / "a")
    # Rewrite artifact.json and its checksum so only the slot check can fail.
    (dest / "artifact.json").write_text(json.dumps({"bogus": {}}))
    metadata_path = dest / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["files"]["artifact.json"] = sha256_file(dest / "artifact.json")
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ArtifactError, match="unknown component 'bogus'"):
        deserialize(dest, FakeArtifact)


@pytest.mark.parametrize(
    "value, match",
    [
        ({"$file": "x"}, "reserved"),
        ({"a": {"$hf": 1}}, "reserved"),
        ({"a": b"bytes"}, "unsupported type bytes"),
        ({"a": [object()]}, r"a\[0\]: unsupported type object"),
        ({"a": {1: 2}}, "not a string"),
        ({"a": float("nan")}, "not finite"),
    ],
)
def test_invalid_values_rejected(
    tmp_path: Path, value: Mapping[str, ArtifactType], match: str
) -> None:
    with pytest.raises(ArtifactError, match=match):
        serialize(_artifact(engine_config=FakeComponent(value)), tmp_path / "a")


@pytest.mark.parametrize(
    "name",
    [
        "",
        "../x",
        "/abs",
        "a/../b",
        "./x",
        "a//b",
        "metadata.json",
        "Artifact.JSON",
        "artifact.json/x",
    ],
)
def test_invalid_file_names_rejected(name: str) -> None:
    with pytest.raises(ArtifactError):
        ArtifactFile(name, "x")


def test_encode_returns_files() -> None:
    a = ArtifactFile("a.txt", "a")
    b = ArtifactFile("b.txt", "b")
    encoded, files = encode({"x": a, "y": [b, a]})
    assert encoded == {
        "x": {"$file": "a.txt"},
        "y": [{"$file": "b.txt"}, {"$file": "a.txt"}],
    }
    assert files == {"a.txt": a, "b.txt": b}


@pytest.mark.parametrize(
    "ref",
    [
        HuggingFaceRef("org/uncached", "d" * 40),
        HuggingFaceRef("org/model", "e" * 40, "tokenizer.json"),
    ],
)
def test_uncached_hf_repo_warns(
    tmp_path: Path, ref: HuggingFaceRef, caplog: pytest.LogCaptureFixture
) -> None:
    model = FakeComponent({"ref": ref})
    dest = serialize(_artifact(engine_config=model), tmp_path / "a")
    with caplog.at_level("WARNING", logger="max.pipelines"):
        assert deserialize(dest, FakeArtifact).engine_config == model
    assert f"{ref.repo_id}@{ref.revision} is not in the local" in caplog.text


def test_missing_path_in_cached_snapshot_rejected(tmp_path: Path) -> None:
    ref = HuggingFaceRef("org/model", MODEL_REVISION, "missing.json")
    dest = serialize(
        _artifact(engine_config=FakeComponent({"ref": ref})), tmp_path / "a"
    )
    with pytest.raises(ArtifactError, match=r"missing\.json from org/model"):
        deserialize(dest, FakeArtifact)


def test_local_repo_source(tmp_path: Path) -> None:
    repo = tmp_path / "local_repo"
    repo.mkdir()
    (repo / "config.json").write_text("{}")
    ref = HuggingFaceRef(str(repo), "main", "config.json")
    dest = serialize(
        _artifact(engine_config=FakeComponent({"ref": ref})), tmp_path / "a"
    )
    assert deserialize(dest, FakeArtifact).model_sources() == (ref,)


def test_same_bytes_in_different_forms_deduplicated(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("x")
    (src / "b.txt").write_text("x")
    model = FakeComponent(
        {
            "str": ArtifactFile("x.txt", "x"),
            "bytes": ArtifactFile("x.txt", b"x"),
            "path": ArtifactFile("x.txt", src / "a.txt"),
            "other_path": ArtifactFile("x.txt", src / "b.txt"),
        }
    )
    dest = serialize(_artifact(engine_config=model), tmp_path / "a")
    assert (dest / "x.txt").read_text() == "x"


def test_different_bytes_rejected(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("y")
    others: list[bytes | Path] = [b"y", src]
    for other in others:
        model = FakeComponent(
            {"a": ArtifactFile("x.txt", "x"), "b": ArtifactFile("x.txt", other)}
        )
        with pytest.raises(ArtifactError, match="different content"):
            serialize(_artifact(engine_config=model), tmp_path / "a")


@pytest.mark.parametrize(
    "first, second, match",
    [
        ("Model.mef", "model.mef", "differ only in case"),
        ("graph", "graph/x.mlir", "one is a directory"),
        ("graph/x.mlir", "graph", "one is a directory"),
        ("Graph", "graph/x.mlir", "one is a directory"),
    ],
)
def test_colliding_names_rejected(
    tmp_path: Path, first: str, second: str, match: str
) -> None:
    model = FakeComponent({"a": ArtifactFile(first, "a")})
    tokenizer = FakeComponent({"b": ArtifactFile(second, "b")})
    with pytest.raises(ArtifactError, match=match):
        serialize(
            _artifact(engine_config=model, tokenizer=tokenizer), tmp_path / "a"
        )
    assert list(tmp_path.iterdir()) == []


def test_component_types() -> None:
    # Resolving the real ModelArtifact guards against slot annotations that
    # can't be loaded.
    assert ModelArtifact.component_types() == {
        **dict.fromkeys(ModelArtifact.component_slots()),
        "model": PipelineModules,
        "batch_processor": BatchProcessorDims,
    }
    assert FakeArtifact.component_types() == {
        **dict.fromkeys(ModelArtifact.component_slots()),
        "model": PipelineModules,
        "batch_processor": BatchProcessorDims,
        "engine_config": FakeComponent,
        "tokenizer": FakeComponent,
    }


def test_ambiguous_slot_rejected() -> None:
    with pytest.raises(ArtifactError, match="exactly one class"):
        AmbiguousArtifact.component_types()


def test_from_components_checks_slot_types() -> None:
    with pytest.raises(ArtifactError, match="expects PipelineModules"):
        ModelArtifact.from_components({"model": FakeComponent()})
    with pytest.raises(ArtifactError, match="no slot 'bogus'"):
        ModelArtifact.from_components({"bogus": FakeComponent()})
    artifact = FakeArtifact.from_components({"tokenizer": FakeComponent()})
    assert artifact.tokenizer == FakeComponent()


def test_artifact_dir_respects_umask(tmp_path: Path) -> None:
    previous = os.umask(0o022)
    try:
        dest = serialize(_artifact(), tmp_path / "a")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(dest.stat().st_mode) == 0o755


def test_prefix_files_moves_every_file() -> None:
    value: ArtifactType = {
        "mef": ArtifactFile("model.mef", b"mef"),
        "nested": [{"graph": ArtifactFile("graph.mlir", "graph")}, 1],
    }
    assert prefix_files(value, "decoder") == {
        "mef": ArtifactFile("decoder/model.mef", b"mef"),
        "nested": [{"graph": ArtifactFile("decoder/graph.mlir", "graph")}, 1],
    }


def test_batch_processor_dims_round_trip(tmp_path: Path) -> None:
    dims = BatchProcessorDims(
        {"total_seq_len": 8192, "replica_0_batch_size": 64}
    )
    dest = serialize(ModelArtifact(batch_processor=dims), tmp_path / "a")
    sections = json.loads((dest / "artifact.json").read_text())
    assert sections["batch_processor"] == {
        "dims": {"replica_0_batch_size": 64, "total_seq_len": 8192}
    }
    assert deserialize(dest).batch_processor == dims


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "8"])
def test_batch_processor_dims_must_be_positive_integers(
    size: ArtifactType,
) -> None:
    with pytest.raises(ArtifactError, match="total_seq_len"):
        BatchProcessorDims.__deserialize__({"dims": {"total_seq_len": size}})
