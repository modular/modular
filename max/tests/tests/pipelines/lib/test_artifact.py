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
from typing import Any

import huggingface_hub
import pytest
from max import _core
from max._artifact import (
    ArtifactComponent,
    ArtifactError,
    ArtifactFile,
    ArtifactType,
    HuggingFaceRef,
)
from max._artifact.batch_processor import BatchProcessorDims
from max._artifact.codec import encode, prefix_files
from max._artifact.engine_config import (
    EngineConfig,
    EngineSamplingConfig,
    EngineSpeculativeConfig,
)
from max._artifact.format_spec import FormatSpec
from max._artifact.kv_cache import (
    KVCacheLayout,
    KVPoolLeaf,
    MHAKVCache,
    MLAKVCache,
    MSAKVCache,
    RecurrentStateCache,
    StateRegion,
)
from max._artifact.metadata import ARTIFACT_VERSION
from max._artifact.model_artifact import ModelArtifact
from max._artifact.pipeline_modules import PipelineModules
from max._artifact.serialization import deserialize, serialize
from max._artifact.value_types import sha256_file
from max.driver import DeviceSpec
from max.dtype import DType
from max.pipelines.lib import (
    CheckpointRole,
    Ignore,
    WeightEntry,
    WeightNaming,
    WeightRenaming,
    WeightTransform,
)
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


# The serializer tests write to a slot of their own, so they don't depend on
# which of ModelArtifact's slots have a concrete component type.
@dataclass(frozen=True)
class FakeArtifact(ModelArtifact):
    scratch: FakeComponent | None = None
    tokenizer: FakeComponent | None = None


@dataclass(frozen=True)
class FailingArtifact(ModelArtifact):
    scratch: FailingComponent | None = None


@dataclass(frozen=True)
class AmbiguousArtifact(ModelArtifact):
    scratch: FakeComponent | FailingComponent | None = None


@dataclass(frozen=True)
class ProtocolArtifact(ModelArtifact):
    scratch: ArtifactComponent | None = None


def _artifact(**components: FakeComponent) -> FakeArtifact:
    return FakeArtifact(
        scratch=components.get("scratch"),
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
        _artifact(scratch=model, tokenizer=tokenizer), tmp_path / "a"
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
    assert isinstance(loaded.scratch, FakeComponent)
    value = loaded.scratch.value
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
        _artifact(scratch=FakeComponent({"f": ArtifactFile("x.txt", "x")})),
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
    assert sections == {"scratch": {"f": {"$file": "x.txt"}}}


def test_weight_naming_round_trip(tmp_path: Path) -> None:
    target = WeightEntry(
        CheckpointRole.TARGET,
        WeightTransform(
            [
                Ignore(r"_scale$", reason="unused"),
                WeightRenaming(r"^model\.", ""),
            ]
        ),
    )
    naming = WeightNaming(
        {"spec_decoder": [target.with_prefix("target.")], "vision": [target]}
    )
    dest = serialize(ModelArtifact(weight_loader=naming), tmp_path / "a")
    sections = json.loads((dest / "artifact.json").read_text())
    assert sections == {"weight_loader": naming.__serialize__()}
    assert deserialize(dest).weight_loader == naming


def test_existing_destination_rejected(tmp_path: Path) -> None:
    serialize(_artifact(), tmp_path / "a")
    with pytest.raises(ArtifactError, match="already exists"):
        serialize(_artifact(), tmp_path / "a")


def test_failed_export_leaves_nothing(tmp_path: Path) -> None:
    ok = FakeComponent({"f": ArtifactFile("x.txt", "x")})
    bad = FakeComponent({"f": ArtifactFile("x.txt", Path("/nonexistent"))})
    with pytest.raises(ArtifactError):
        serialize(_artifact(scratch=ok, tokenizer=bad), tmp_path / "a")
    with pytest.raises(ArtifactError, match="does not exist"):
        serialize(_artifact(scratch=bad), tmp_path / "a")
    with pytest.raises(RuntimeError):
        serialize(_artifact(scratch=FailingComponent()), tmp_path / "a")
    assert list(tmp_path.iterdir()) == []


def test_identical_files_deduplicated(tmp_path: Path) -> None:
    f = ArtifactFile("x.txt", "x")
    dest = serialize(
        _artifact(
            scratch=FakeComponent({"f": f}),
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
        _artifact(scratch=FakeComponent({"f": ArtifactFile("x.txt", "x")})),
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
    dest = serialize(_artifact(scratch=FakeComponent()), tmp_path / "a")
    with pytest.raises(ArtifactError, match="no concrete component type"):
        deserialize(dest, ProtocolArtifact)


def test_component_deserialize_error_wrapped(tmp_path: Path) -> None:
    dest = serialize(_artifact(scratch=FakeComponent()), tmp_path / "a")
    with pytest.raises(ArtifactError, match="re-export"):
        deserialize(dest, FailingArtifact)


def _rewrite_sections(dest: Path, sections: Mapping[str, Any]) -> None:
    """Rewrites artifact.json and its checksum, so only what reads the
    sections can fail."""
    (dest / "artifact.json").write_text(json.dumps(sections))
    metadata_path = dest / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["files"]["artifact.json"] = sha256_file(dest / "artifact.json")
    metadata_path.write_text(json.dumps(metadata))


def test_unknown_slot_rejected(tmp_path: Path) -> None:
    dest = serialize(_artifact(), tmp_path / "a")
    _rewrite_sections(dest, {"bogus": {}})
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
        serialize(_artifact(scratch=FakeComponent(value)), tmp_path / "a")


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
    dest = serialize(_artifact(scratch=model), tmp_path / "a")
    with caplog.at_level("WARNING", logger="max.pipelines"):
        assert deserialize(dest, FakeArtifact).scratch == model
    assert f"{ref.repo_id}@{ref.revision} is not in the local" in caplog.text


def test_missing_path_in_cached_snapshot_rejected(tmp_path: Path) -> None:
    ref = HuggingFaceRef("org/model", MODEL_REVISION, "missing.json")
    dest = serialize(
        _artifact(scratch=FakeComponent({"ref": ref})), tmp_path / "a"
    )
    with pytest.raises(ArtifactError, match=r"missing\.json from org/model"):
        deserialize(dest, FakeArtifact)


def test_local_repo_source(tmp_path: Path) -> None:
    repo = tmp_path / "local_repo"
    repo.mkdir()
    (repo / "config.json").write_text("{}")
    ref = HuggingFaceRef(str(repo), "main", "config.json")
    dest = serialize(
        _artifact(scratch=FakeComponent({"ref": ref})), tmp_path / "a"
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
    dest = serialize(_artifact(scratch=model), tmp_path / "a")
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
            serialize(_artifact(scratch=model), tmp_path / "a")


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
        serialize(_artifact(scratch=model, tokenizer=tokenizer), tmp_path / "a")
    assert list(tmp_path.iterdir()) == []


def test_component_types() -> None:
    # Resolving the real ModelArtifact guards against slot annotations that
    # can't be loaded.
    assert ModelArtifact.component_types() == {
        **dict.fromkeys(ModelArtifact.component_slots()),
        "model": PipelineModules,
        "format_spec": FormatSpec,
        "batch_processor": BatchProcessorDims,
        "engine_config": EngineConfig,
        "kv_cache": KVCacheLayout,
        "weight_loader": WeightNaming,
    }
    assert FakeArtifact.component_types() == {
        **dict.fromkeys(ModelArtifact.component_slots()),
        "model": PipelineModules,
        "format_spec": FormatSpec,
        "batch_processor": BatchProcessorDims,
        "engine_config": EngineConfig,
        "kv_cache": KVCacheLayout,
        "weight_loader": WeightNaming,
        "scratch": FakeComponent,
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


ENGINE_CONFIG = EngineConfig(
    max_batch_size=16,
    max_length=4096,
    max_batch_input_tokens=8192,
    max_batch_total_tokens=None,
    devices=(DeviceSpec.accelerator(0), DeviceSpec.accelerator(1)),
    data_parallel_degree=1,
    ep_size=1,
    sampling=EngineSamplingConfig(
        needs_bitmask_constraints=True,
        enable_variable_logits=False,
        enable_penalties=False,
    ),
    speculative=EngineSpeculativeConfig(
        speculative_method="mtp",
        num_speculative_tokens=3,
        draft_proposal="argmax",
        reasoning_parser="gemma4",
    ),
)


def test_engine_config_round_trip(tmp_path: Path) -> None:
    dest = serialize(ModelArtifact(engine_config=ENGINE_CONFIG), tmp_path / "a")
    # The layout the engine reads.
    assert json.loads((dest / "artifact.json").read_text()) == {
        "engine_config": {
            "max_batch_size": 16,
            "max_length": 4096,
            "max_batch_input_tokens": 8192,
            "max_batch_total_tokens": None,
            "devices": [{"type": "gpu", "id": 0}, {"type": "gpu", "id": 1}],
            "data_parallel_degree": 1,
            "ep_size": 1,
            "sampling": {
                "needs_bitmask_constraints": True,
                "enable_variable_logits": False,
                "enable_penalties": False,
            },
            "speculative": {
                "speculative_method": "mtp",
                "num_speculative_tokens": 3,
                "draft_proposal": "argmax",
                "reasoning_parser": "gemma4",
            },
        }
    }
    assert deserialize(dest).engine_config == ENGINE_CONFIG


@pytest.mark.parametrize(
    "edit",
    [
        lambda section: section.update(bogus=1),
        lambda section: section.pop("ep_size"),
        lambda section: section["sampling"].update(enable_penalties=1),
        lambda section: section["devices"].append({"type": "tpu", "id": 0}),
    ],
)
def test_malformed_engine_config_rejected(
    tmp_path: Path, edit: Callable[[dict[str, Any]], object]
) -> None:
    dest = serialize(ModelArtifact(engine_config=ENGINE_CONFIG), tmp_path / "a")
    sections = json.loads((dest / "artifact.json").read_text())
    edit(sections["engine_config"])
    _rewrite_sections(dest, sections)
    with pytest.raises(ArtifactError, match="component 'engine_config'"):
        deserialize(dest)


def _attention_fields(name: str, group: str) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": DType.bfloat16,
        "num_layers": 2,
        "head_dim": 128,
        "v_head_dim": None,
        "window_size": None,
        "slots_per_page": 128,
        "stride_alignment": None,
        "scratch": False,
        "kvcache_quant_config": None,
        "leaves": (
            KVPoolLeaf(
                leaf_id=f"{name}.{group}",
                group="full",
                window_size=None,
                bytes_per_page=131072,
                stride_alignment=None,
            ),
        ),
    }


KV_CACHE = KVCacheLayout(
    page_size=128,
    caches=(
        MHAKVCache(
            **_attention_fields("attention", "full_group"),
            n_kv_heads=4,
            allow_kv_head_replication=False,
        ),
        MSAKVCache(
            **_attention_fields("indexer", "full_group"),
            n_kv_heads=1,
            allow_kv_head_replication=False,
        ),
        MLAKVCache(
            **_attention_fields("latent", "full_group"),
            num_q_heads=16,
            sparse_indices_stride=2048,
        ),
        RecurrentStateCache(
            name="state",
            regions=(
                StateRegion(
                    leaf_id="conv",
                    num_layers=2,
                    row_shape=(4, 8),
                    dtype=DType.float32,
                    scratch=False,
                ),
            ),
            leaves=(
                KVPoolLeaf(
                    leaf_id="conv",
                    group="recurrent",
                    window_size=None,
                    bytes_per_page=256,
                    stride_alignment=None,
                ),
            ),
        ),
    ),
)


def test_kv_cache_round_trip(tmp_path: Path) -> None:
    dest = serialize(ModelArtifact(kv_cache=KV_CACHE), tmp_path / "a")
    # The layout the engine reads: caches in graph-input order, each with its
    # kind and the pool leaves it occupies.
    caches = json.loads((dest / "artifact.json").read_text())["kv_cache"][
        "caches"
    ]
    assert [(cache["name"], cache["kind"]) for cache in caches] == [
        ("attention", "mha"),
        ("indexer", "msa"),
        ("latent", "mla"),
        ("state", "recurrent"),
    ]
    assert caches[0] == {
        "name": "attention",
        "kind": "mha",
        "dtype": "bfloat16",
        "num_layers": 2,
        "n_kv_heads": 4,
        "head_dim": 128,
        "v_head_dim": None,
        "window_size": None,
        "allow_kv_head_replication": False,
        "slots_per_page": 128,
        "stride_alignment": None,
        "scratch": False,
        "kvcache_quant_config": None,
        "leaves": [
            {
                "leaf_id": "attention.full_group",
                "group": "full",
                "window_size": None,
                "bytes_per_page": 131072,
                "stride_alignment": None,
            }
        ],
    }
    assert deserialize(dest).kv_cache == KV_CACHE


@pytest.mark.parametrize(
    "edit",
    [
        lambda section: section["caches"][0].update(kind="ssm"),
        lambda section: section["caches"][0].pop("kind"),
        lambda section: section["caches"][1].update(dtype="bfloat17"),
        lambda section: section["caches"][1].update(bogus=1),
        lambda section: section["caches"][2]["leaves"][0].update(group="lru"),
        lambda section: section["caches"][3]["regions"][0].pop("row_shape"),
    ],
)
def test_malformed_kv_cache_rejected(
    tmp_path: Path, edit: Callable[[dict[str, Any]], object]
) -> None:
    dest = serialize(ModelArtifact(kv_cache=KV_CACHE), tmp_path / "a")
    sections = json.loads((dest / "artifact.json").read_text())
    edit(sections["kv_cache"])
    _rewrite_sections(dest, sections)
    with pytest.raises(ArtifactError, match="component 'kv_cache'"):
        deserialize(dest)


def test_format_spec_stored_unchanged(tmp_path: Path) -> None:
    document = tmp_path / "spec.json"
    document.write_text('{"format": "formatspec.spec", "spec_version": 1}\n')
    dest = serialize(
        ModelArtifact(format_spec=FormatSpec(document)), tmp_path / "a"
    )
    sections = json.loads((dest / "artifact.json").read_text())
    assert sections["format_spec"] == {
        "document": {"$file": "format_spec.json"}
    }
    assert (dest / "format_spec.json").read_bytes() == document.read_bytes()
    assert deserialize(dest).format_spec == FormatSpec(
        dest / "format_spec.json"
    )


def test_format_spec_document_must_be_a_file() -> None:
    with pytest.raises(ArtifactError, match=r"format_spec\.document"):
        FormatSpec.__deserialize__({"document": "spec.json"})
