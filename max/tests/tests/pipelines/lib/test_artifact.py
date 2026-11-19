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

import dataclasses
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
from max._artifact.block_diffusion import BlockDiffusionMetadata
from max._artifact.codec import encode, prefix_files
from max._artifact.engine_config import (
    CapturedHostInput,
    EngineCaptureConfig,
    EngineConfig,
    EngineParallelConfig,
    EnginePrefillContextParallel,
    EngineSamplingConfig,
    EngineSpeculativeConfig,
)
from max._artifact.format_spec import FormatSpec
from max._artifact.kv_cache import (
    ContextParallel,
    KVCacheLayout,
    KVPoolLeaf,
    MHAKVCache,
    MLAKVCache,
    MSAIndexKVCache,
    MSAKVCache,
    RecurrentStateCache,
    StateRegion,
)
from max._artifact.metadata import ARTIFACT_VERSION
from max._artifact.model_artifact import ModelArtifact
from max._artifact.pipeline_modules import PipelineModules
from max._artifact.sampling_defaults import SamplingDefaults
from max._artifact.serialization import deserialize, serialize
from max._artifact.tokenizer import Tokenizer
from max._artifact.value_types import sha256_file
from max._artifact.vision import VisionMetadata
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
    extra: FakeComponent | None = None


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
        extra=components.get("extra"),
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

    dest = serialize(_artifact(scratch=model, extra=tokenizer), tmp_path / "a")
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
    assert loaded.extra == tokenizer
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


def test_block_diffusion_round_trip(tmp_path: Path) -> None:
    """The canvas geometry and denoise schedule survive a serialize/load."""
    metadata = BlockDiffusionMetadata.from_config(
        canvas_length=256,
        max_denoising_steps=48,
        t_min=0.4,
        t_max=0.8,
        entropy_bound=0.1,
        stability_threshold=1,
        confidence_threshold=0.005,
        pad_token_id=0,
        top_k=64,
        max_canvas_batch=16,
    )
    dest = serialize(ModelArtifact(block_diffusion=metadata), tmp_path / "bd")
    loaded = deserialize(dest, ModelArtifact)
    assert loaded.block_diffusion == metadata
    config = json.loads((dest / "block_diffusion_config.json").read_text())
    assert config["canvas_length"] == 256
    assert config["max_denoising_steps"] == 48
    assert config["top_k"] == 64


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
        serialize(_artifact(scratch=ok, extra=bad), tmp_path / "a")
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
            extra=FakeComponent({"f": f}),
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
        serialize(_artifact(scratch=model, extra=tokenizer), tmp_path / "a")
    assert list(tmp_path.iterdir()) == []


def test_component_types() -> None:
    # Resolving the real ModelArtifact guards against slot annotations that
    # can't be loaded.
    assert ModelArtifact.component_types() == {
        **dict.fromkeys(ModelArtifact.component_slots()),
        "model": PipelineModules,
        "format_spec": FormatSpec,
        "tokenizer": Tokenizer,
        "batch_processor": BatchProcessorDims,
        "engine_config": EngineConfig,
        "kv_cache": KVCacheLayout,
        "weight_loader": WeightNaming,
        "vision": VisionMetadata,
        "sampling_defaults": SamplingDefaults,
        "block_diffusion": BlockDiffusionMetadata,
    }
    assert FakeArtifact.component_types() == {
        **dict.fromkeys(ModelArtifact.component_slots()),
        "model": PipelineModules,
        "format_spec": FormatSpec,
        "tokenizer": Tokenizer,
        "batch_processor": BatchProcessorDims,
        "engine_config": EngineConfig,
        "kv_cache": KVCacheLayout,
        "weight_loader": WeightNaming,
        "vision": VisionMetadata,
        "sampling_defaults": SamplingDefaults,
        "block_diffusion": BlockDiffusionMetadata,
        "scratch": FakeComponent,
        "extra": FakeComponent,
    }


def test_ambiguous_slot_rejected() -> None:
    with pytest.raises(ArtifactError, match="exactly one class"):
        AmbiguousArtifact.component_types()


def test_from_components_checks_slot_types() -> None:
    with pytest.raises(ArtifactError, match="expects PipelineModules"):
        ModelArtifact.from_components({"model": FakeComponent()})
    with pytest.raises(ArtifactError, match="no slot 'bogus'"):
        ModelArtifact.from_components({"bogus": FakeComponent()})
    artifact = FakeArtifact.from_components({"extra": FakeComponent()})
    assert artifact.extra == FakeComponent()


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


def _parallel(
    dp: int = 1,
    tp: int = 2,
    ep: int = 1,
    dcp: int = 1,
    pcp: int = 1,
    dcp_interleave: int = 1,
) -> EngineParallelConfig:
    return EngineParallelConfig(
        dp=dp, tp=tp, ep=ep, dcp=dcp, pcp=pcp, dcp_interleave=dcp_interleave
    )


PCP = EnginePrefillContextParallel(
    min_step_tokens=7168, min_rows_per_request=8, max_batch_context=262144
)
_PCP_RECORD = {
    "min_step_tokens": 7168,
    "min_rows_per_request": 8,
    "max_batch_context": 262144,
}

SPECULATIVE = EngineSpeculativeConfig(
    speculative_method="mtp",
    num_speculative_tokens=3,
    draft_proposal="argmax",
    acceptance_mode="stochastic",
    reasoning_parser="gemma4",
    draft_query_len=1,
    proposer="sequential",
    verify_widths=(1, 3),
    mixed_verify_width=2,
    # Mixed values, so a swapped field cannot pass.
    verify_takes_prefill_rows=True,
    verify_commits_long_rows=False,
    successor_replay=True,
)

ENGINE_CONFIG = EngineConfig(
    max_batch_size=16,
    max_length=4096,
    max_batch_input_tokens=8192,
    prefill_token_budget=8000,
    max_batch_total_tokens=None,
    devices=(DeviceSpec.accelerator(0), DeviceSpec.accelerator(1)),
    parallel=_parallel(),
    capture=EngineCaptureConfig(
        max_batch_size=16,
        prefill_max_tokens=4096,
        host_inputs={
            "spec_decode.min_top_p": CapturedHostInput("pinned", 0.0),
            "paged_kv.max_cache_length": CapturedHostInput("shape"),
            "spec_decode.max_k": CapturedHostInput("pinned", -1),
        },
        cache_bound="bucketed",
    ),
    sampling=EngineSamplingConfig(
        needs_bitmask_constraints=True,
        enable_variable_logits=False,
        enable_penalties=False,
    ),
    speculative=SPECULATIVE,
    prefix_caching=False,
)


def test_engine_config_round_trip(tmp_path: Path) -> None:
    dest = serialize(ModelArtifact(engine_config=ENGINE_CONFIG), tmp_path / "a")
    # The layout the engine reads.
    assert json.loads((dest / "artifact.json").read_text()) == {
        "engine_config": {
            "max_batch_size": 16,
            "max_length": 4096,
            "max_batch_input_tokens": 8192,
            "prefill_token_budget": 8000,
            "max_batch_total_tokens": None,
            "devices": [{"type": "gpu", "id": 0}, {"type": "gpu", "id": 1}],
            "parallel": {
                "dp": 1,
                "tp": 2,
                "ep": 1,
                "dcp": 1,
                "pcp": 1,
                "dcp_interleave": 1,
            },
            "capture": {
                "max_batch_size": 16,
                "prefill_max_tokens": 4096,
                # Sorted by name; only a pinned input carries a value.
                "host_inputs": {
                    "paged_kv.max_cache_length": {
                        "mode": "shape",
                        "value": None,
                    },
                    "spec_decode.max_k": {"mode": "pinned", "value": -1},
                    "spec_decode.min_top_p": {"mode": "pinned", "value": 0.0},
                },
                "cache_bound": "bucketed",
            },
            "sampling": {
                "needs_bitmask_constraints": True,
                "enable_variable_logits": False,
                "enable_penalties": False,
            },
            "speculative": {
                "speculative_method": "mtp",
                "num_speculative_tokens": 3,
                "draft_proposal": "argmax",
                "acceptance_mode": "stochastic",
                "reasoning_parser": "gemma4",
                "draft_query_len": 1,
                "proposer": "sequential",
                "verify_widths": [1, 3],
                "mixed_verify_width": 2,
                "verify_takes_prefill_rows": True,
                "verify_commits_long_rows": False,
                "successor_replay": True,
            },
            "prefill_context_parallel": None,
            "prefix_caching": False,
        }
    }
    assert deserialize(dest).engine_config == ENGINE_CONFIG


@pytest.mark.parametrize(
    "devices, parallel",
    [
        # TP4 / EP4, one replica.
        (4, _parallel(tp=4, ep=4)),
        # DP4 / EP4: four one-device replicas sharing the experts.
        (4, _parallel(dp=4, tp=1, ep=4)),
        # DP8 / EP8.
        (8, _parallel(dp=8, tp=1, ep=8)),
        # DP2 x TP4 with DCP4 interleaved by page.
        (8, _parallel(dp=2, tp=4, dcp=4, dcp_interleave=128)),
        # DCP2 without PCP.
        (2, _parallel(tp=2, dcp=2, dcp_interleave=1)),
    ],
)
def test_parallel_layouts_round_trip(
    tmp_path: Path, devices: int, parallel: EngineParallelConfig
) -> None:
    config = dataclasses.replace(
        ENGINE_CONFIG,
        devices=tuple(DeviceSpec.accelerator(i) for i in range(devices)),
        parallel=parallel,
        # PCP ranks come with the record of when a step takes them.
        prefill_context_parallel=PCP if parallel.pcp > 1 else None,
    )
    dest = serialize(ModelArtifact(engine_config=config), tmp_path / "a")
    section = json.loads((dest / "artifact.json").read_text())["engine_config"]
    assert section["parallel"] == dataclasses.asdict(parallel)
    assert section["prefill_context_parallel"] == (
        _PCP_RECORD if parallel.pcp > 1 else None
    )
    assert deserialize(dest).engine_config == config


@pytest.mark.parametrize(
    "devices, degrees, match",
    [
        (4, {"dp": 2, "tp": 1}, "4 devices for dp 2 x tp 1"),
        (4, {"tp": 4, "dcp": 3}, "must divide"),
        (4, {"tp": 4, "dcp": 2, "pcp": 2}, "PCP is not supported"),
        (4, {"tp": 4, "dcp": 2, "pcp": 4}, "PCP is not supported"),
        (2, {"tp": 2, "dcp_interleave": 64}, "needs a decode context"),
        (4, {"tp": 4, "ep": 3}, "expert-parallel degree 3"),
        (2, {"tp": 2, "ep": 0}, "positive"),
    ],
)
def test_inconsistent_parallel_layouts_are_refused(
    devices: int, degrees: dict[str, int], match: str
) -> None:
    with pytest.raises(ArtifactError, match=match):
        dataclasses.replace(
            ENGINE_CONFIG,
            devices=tuple(DeviceSpec.accelerator(i) for i in range(devices)),
            parallel=_parallel(**degrees),
        )


def test_the_pcp_record_comes_only_with_pcp_ranks() -> None:
    with pytest.raises(ArtifactError, match="prefill_context_parallel"):
        dataclasses.replace(ENGINE_CONFIG, prefill_context_parallel=PCP)


def test_inconsistent_parallel_layouts_are_refused_at_load(
    tmp_path: Path,
) -> None:
    dest = serialize(ModelArtifact(engine_config=ENGINE_CONFIG), tmp_path / "a")
    sections = json.loads((dest / "artifact.json").read_text())
    sections["engine_config"]["parallel"]["dp"] = 2
    _rewrite_sections(dest, sections)
    with pytest.raises(ArtifactError, match="component 'engine_config'"):
        deserialize(dest)


def test_block_spec_facts_round_trip(tmp_path: Path) -> None:
    speculative = dataclasses.replace(
        SPECULATIVE,
        speculative_method="dflash2",
        num_speculative_tokens=15,
        draft_proposal="sampled",
        acceptance_mode="greedy",
        reasoning_parser=None,
        draft_query_len=15,
        proposer="block",
        verify_widths=(0, 4, 8, 15),
        mixed_verify_width=None,
        verify_takes_prefill_rows=False,
    )
    config = dataclasses.replace(ENGINE_CONFIG, speculative=speculative)
    dest = serialize(ModelArtifact(engine_config=config), tmp_path / "a")
    section = json.loads((dest / "artifact.json").read_text())["engine_config"]
    assert section["speculative"]["proposer"] == "block"
    assert section["speculative"]["draft_query_len"] == 15
    # A schedule that skips the drafter verifies 0 drafts.
    assert section["speculative"]["verify_widths"] == [0, 4, 8, 15]
    assert section["speculative"]["mixed_verify_width"] is None
    assert deserialize(dest).engine_config == config


def test_a_plain_pipeline_records_no_speculative_facts(tmp_path: Path) -> None:
    config = dataclasses.replace(
        ENGINE_CONFIG,
        speculative=None,
        prefill_token_budget=None,
        capture=EngineCaptureConfig(max_batch_size=0, prefill_max_tokens=0),
    )
    dest = serialize(ModelArtifact(engine_config=config), tmp_path / "a")
    section = json.loads((dest / "artifact.json").read_text())["engine_config"]
    assert section["speculative"] is None
    assert section["capture"] == {
        "max_batch_size": 0,
        "prefill_max_tokens": 0,
        "host_inputs": {},
        "cache_bound": "aligned",
    }
    assert deserialize(dest).engine_config == config


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"verify_widths": ()}, "verify_widths"),
        ({"verify_widths": (3, 1)}, "verify_widths"),
        ({"verify_widths": (1, 1)}, "verify_widths"),
        ({"verify_widths": (0,)}, "a width above 0"),
        ({"verify_widths": (-1, 3)}, "verify_widths"),
        ({"mixed_verify_width": 4}, "mixed_verify_width 4"),
        ({"mixed_verify_width": -1}, "mixed_verify_width -1"),
        (
            {"mixed_verify_width": 0, "verify_takes_prefill_rows": False},
            "takes prefill rows",
        ),
        ({"verify_widths": (4,)}, "exceeds the 3 drafted"),
        ({"draft_query_len": 0}, "draft_query_len"),
    ],
)
def test_inconsistent_spec_facts_are_refused(
    changes: dict[str, Any], match: str
) -> None:
    with pytest.raises(ArtifactError, match=match):
        dataclasses.replace(SPECULATIVE, **changes)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"prefill_token_budget": 0}, "must be positive"),
        ({"prefill_token_budget": 8193}, "compiled input capacity"),
        # Rust refuses an artifact whose capture exceeds its batch size.
        (
            {
                "capture": EngineCaptureConfig(
                    max_batch_size=17, prefill_max_tokens=4096
                )
            },
            "capture of 17 rows exceeds the batch size 16",
        ),
        (
            {
                "capture": EngineCaptureConfig(
                    max_batch_size=16, prefill_max_tokens=8001
                )
            },
            "exceeds the prefill token budget 8000",
        ),
        (
            {
                "prefill_token_budget": None,
                "capture": EngineCaptureConfig(
                    max_batch_size=16, prefill_max_tokens=8193
                ),
            },
            "exceeds the prefill token budget 8192",
        ),
    ],
)
def test_prefill_budget_and_capture_must_fit_the_input_capacity(
    changes: dict[str, Any], match: str
) -> None:
    with pytest.raises(ArtifactError, match=match):
        dataclasses.replace(ENGINE_CONFIG, **changes)


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: CapturedHostInput("pinned"), "pinned host input needs"),
        (lambda: CapturedHostInput("shape", 1), "shape host input takes no"),
        (
            lambda: EngineCaptureConfig(
                max_batch_size=1,
                prefill_max_tokens=0,
                host_inputs={"max_k": CapturedHostInput("shape")},
            ),
            "not <group kind>.<field>",
        ),
        (
            lambda: EngineCaptureConfig(
                max_batch_size=1,
                prefill_max_tokens=0,
                host_inputs={"a.b.c": CapturedHostInput("shape")},
            ),
            "not <group kind>.<field>",
        ),
    ],
)
def test_malformed_captured_host_inputs_are_refused(
    build: Callable[[], object], match: str
) -> None:
    # The Rust loader refuses the same records (EngineConfig::validate).
    with pytest.raises(ArtifactError, match=match):
        build()


def test_prefill_capture_needs_decode_capture() -> None:
    with pytest.raises(ArtifactError, match="max_batch_size is 0"):
        EngineCaptureConfig(max_batch_size=0, prefill_max_tokens=128)


@pytest.mark.parametrize(
    "edit",
    [
        lambda section: section.update(bogus=1),
        lambda section: section.pop("parallel"),
        lambda section: section.pop("prefill_token_budget"),
        lambda section: section.update(data_parallel_degree=1),
        lambda section: section.update(device_graph_capture=True),
        lambda section: section["parallel"].pop("dcp"),
        lambda section: section["parallel"].update(tp=2.0),
        lambda section: section["capture"].pop("prefill_max_tokens"),
        lambda section: section["capture"].pop("host_inputs"),
        lambda section: section["capture"]["host_inputs"].update(
            {"a.b": {"mode": "baked", "value": None}}
        ),
        lambda section: section["capture"]["host_inputs"].update(
            {"a.b": {"mode": "pinned", "value": "-1"}}
        ),
        lambda section: section["capture"]["host_inputs"].update(
            {"a.b": {"mode": "shape"}}
        ),
        lambda section: section["capture"].update(max_batch_size=True),
        lambda section: section["capture"].pop("cache_bound"),
        lambda section: section["capture"].update(cache_bound="sometimes"),
        lambda section: section["sampling"].update(enable_penalties=1),
        lambda section: section["devices"].append({"type": "tpu", "id": 0}),
        lambda section: section["speculative"].update(
            acceptance_mode="sampled"
        ),
        lambda section: section["speculative"].pop("acceptance_mode"),
        lambda section: section["speculative"].update(draft_proposal="greedy"),
        lambda section: section["speculative"].update(proposer="tree"),
        lambda section: section["speculative"].pop("draft_query_len"),
        lambda section: section["speculative"].update(verify_widths=3),
        lambda section: section["speculative"].pop("mixed_verify_width"),
        lambda section: section["speculative"].update(mixed_verify_width=1.0),
        lambda section: section["speculative"].pop("successor_replay"),
        lambda section: section["speculative"].update(
            verify_takes_prefill_rows=1
        ),
        lambda section: section.pop("prefill_context_parallel"),
        # A PCP record without pcp ranks, and one short a field.
        lambda section: section.update(
            prefill_context_parallel=dict(_PCP_RECORD)
        ),
        lambda section: section.update(
            prefill_context_parallel={
                "min_step_tokens": 7168,
                "min_rows_per_request": 8,
            }
        ),
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
        "speculative_method": None,
        "num_draft_tokens": 0,
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
            **_attention_fields("attention", "full_group")
            | {"speculative_method": "mtp", "num_draft_tokens": 3},
            n_kv_heads=4,
            allow_kv_head_replication=False,
        ),
        MSAKVCache(
            **_attention_fields("sparse", "full_group"),
            n_kv_heads=1,
            allow_kv_head_replication=False,
        ),
        MLAKVCache(
            **_attention_fields("latent", "full_group"),
            num_q_heads=16,
            sparse_indices_stride=2048,
            dispatch_unread=True,
            sparse_kv_form="plain",
            context_parallel=ContextParallel(degree=4, interleave=128),
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
        MSAIndexKVCache(
            **_attention_fields("indexer", "full_group"),
            n_kv_heads=1,
            allow_kv_head_replication=False,
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
        ("sparse", "msa"),
        ("latent", "mla"),
        ("state", "recurrent"),
        ("indexer", "msa_index"),
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
        "speculative_method": "mtp",
        "num_draft_tokens": 3,
        "leaves": [
            {
                "leaf_id": "attention.full_group",
                "group": "full",
                "window_size": None,
                "bytes_per_page": 131072,
                "stride_alignment": None,
            }
        ],
        "context_parallel": None,
    }
    assert caches[2]["context_parallel"] == {"degree": 4, "interleave": 128}
    assert deserialize(dest).kv_cache == KV_CACHE


@pytest.mark.parametrize(
    "edit",
    [
        lambda section: section["caches"][0].update(kind="ssm"),
        lambda section: section["caches"][0].pop("kind"),
        lambda section: section["caches"][1].update(dtype="bfloat17"),
        lambda section: section["caches"][1].update(bogus=1),
        lambda section: section["caches"][2]["leaves"][0].update(group="lru"),
        lambda section: section["caches"][2].update(sparse_kv_form="fp4"),
        lambda section: section["caches"][3]["regions"][0].pop("row_shape"),
        lambda section: section["caches"][2]["context_parallel"].pop("degree"),
        lambda section: section["caches"][1].pop("context_parallel"),
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


@pytest.mark.parametrize("field", ["sparse_kv_form", "dispatch_unread"])
def test_a_cache_field_is_required(tmp_path: Path, field: str) -> None:
    dest = serialize(ModelArtifact(kv_cache=KV_CACHE), tmp_path / "a")
    sections = json.loads((dest / "artifact.json").read_text())
    del sections["kv_cache"]["caches"][2][field]
    _rewrite_sections(dest, sections)
    with pytest.raises(ArtifactError, match="component 'kv_cache'"):
        deserialize(dest)


def test_compressed_scratch_and_windowed_mla_caches_round_trip(
    tmp_path: Path,
) -> None:
    def mla(name: str, group: str, **changes: Any) -> MLAKVCache:
        fields = _attention_fields(name, group)
        fields["leaves"] = (
            dataclasses.replace(
                fields["leaves"][0],
                group=changes.pop("leaf_group", "full"),
                window_size=changes.get("window_size"),
            ),
        )
        return MLAKVCache(
            **fields | changes, num_q_heads=64, sparse_indices_stride=None
        )

    layout = KVCacheLayout(
        page_size=256,
        caches=(
            # A sliding window.
            mla(
                "target.swa",
                "sliding_window_group(128)",
                window_size=128,
                leaf_group="sliding_window",
            ),
            # Compressed: four tokens to a slot.
            mla("target.c4a", "full_group", slots_per_page=64),
            # A per-request scratch state.
            mla(
                "target.c4a_state",
                "scratch_group",
                scratch=True,
                leaf_group="scratch",
            ),
            mla("target.idx_c4a", "full_group", slots_per_page=64),
            dataclasses.replace(
                KV_CACHE.caches[0],
                name="draft",
                leaves=(
                    dataclasses.replace(
                        KV_CACHE.caches[0].leaves[0],
                        leaf_id="draft.full_group",
                    ),
                ),
            ),
        ),
    )
    dest = serialize(ModelArtifact(kv_cache=layout), tmp_path / "a")
    caches = json.loads((dest / "artifact.json").read_text())["kv_cache"][
        "caches"
    ]
    assert [cache["name"] for cache in caches] == [
        "target.swa",
        "target.c4a",
        "target.c4a_state",
        "target.idx_c4a",
        "draft",
    ]
    assert caches[0]["window_size"] == 128
    assert caches[1]["slots_per_page"] == 64
    assert caches[2]["scratch"] is True
    assert deserialize(dest).kv_cache == layout


def test_another_format_version_is_refused(tmp_path: Path) -> None:
    dest = serialize(ModelArtifact(kv_cache=KV_CACHE), tmp_path / "a")
    metadata = json.loads((dest / "metadata.json").read_text())
    metadata["artifact_version"] = ARTIFACT_VERSION - 1
    (dest / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ArtifactError, match="format version"):
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


def test_tokenizer_round_trip(tmp_path: Path) -> None:
    tokenizer = Tokenizer(
        family="gemma4",
        flavor="thinking",
        files={"tokenizer.json": TOKENIZER_REF},
        chat_templates={"default": "{{ messages }}"},
        vocab_size=262144,
        sampleable_vocab_size=262144,
        eos_token_ids=(1, 106),
    )
    dest = serialize(ModelArtifact(tokenizer=tokenizer), tmp_path / "a")
    sections = json.loads((dest / "artifact.json").read_text())
    assert sections["tokenizer"] == {
        "family": "gemma4",
        "flavor": "thinking",
        "files": {
            "tokenizer.json": {
                "$hf": {
                    "repo_id": "org/model",
                    "revision": MODEL_REVISION,
                    "path": "tokenizer.json",
                }
            }
        },
        "chat_templates": {
            "default": {"$file": "tokenizer/chat_templates/default.jinja"}
        },
        "vocab_size": 262144,
        "sampleable_vocab_size": 262144,
        "eos_token_ids": [1, 106],
    }
    metadata = json.loads((dest / "metadata.json").read_text())
    assert "tokenizer/chat_templates/default.jinja" in metadata["files"]
    assert not any(
        name.startswith("tokenizer/tokenizer") for name in metadata["files"]
    )
    assert deserialize(dest).tokenizer == Tokenizer(
        family="gemma4",
        flavor="thinking",
        files={"tokenizer.json": TOKENIZER_REF},
        chat_templates={
            "default": dest / "tokenizer/chat_templates/default.jinja"
        },
        vocab_size=262144,
        sampleable_vocab_size=262144,
        eos_token_ids=(1, 106),
    )


def test_sampling_defaults_round_trip(tmp_path: Path) -> None:
    defaults = SamplingDefaults(
        temperature=1.0, top_k=None, top_p=0.95, repetition_penalty=1.05
    )
    dest = serialize(ModelArtifact(sampling_defaults=defaults), tmp_path / "a")
    sections = json.loads((dest / "artifact.json").read_text())
    assert sections["sampling_defaults"] == {
        "temperature": 1.0,
        "top_k": None,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
    }
    assert deserialize(dest).sampling_defaults == defaults


@pytest.mark.parametrize(
    "value",
    [
        {"temperature": 1.0, "top_k": None, "top_p": 1.0},
        {
            "temperature": "1",
            "top_k": None,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        {
            "temperature": 1.0,
            "top_k": 1.5,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        {
            "temperature": 1.0,
            "top_k": None,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "min_p": 0.1,
        },
    ],
)
def test_malformed_sampling_defaults_rejected(value: dict[str, Any]) -> None:
    with pytest.raises(ArtifactError):
        SamplingDefaults.__deserialize__(value)


def test_malformed_tokenizer_rejected(tmp_path: Path) -> None:
    tokenizer = Tokenizer(None, None, {}, {}, 8, 8, (7,))
    dest = serialize(ModelArtifact(tokenizer=tokenizer), tmp_path / "a")
    assert deserialize(dest).tokenizer == tokenizer
    sections = json.loads((dest / "artifact.json").read_text())
    sections["tokenizer"]["eos_token_ids"] = ["7"]
    _rewrite_sections(dest, sections)
    with pytest.raises(ArtifactError, match="component 'tokenizer'"):
        deserialize(dest)
