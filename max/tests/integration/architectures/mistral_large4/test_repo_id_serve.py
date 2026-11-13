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
"""Serving Mistral Large 4 straight from its Hugging Face repo id.

The NVFP4 release publishes ``params.json``, safetensors and the base release's
tokenizer artifacts; it publishes no ``config.json``. So a repo-id serve has to
derive that config from ``params.json`` at the revision the repo was asked for,
and has to fetch the ``params.json`` itself, which a weights download never
brings. These tests pin what the derivation produces (the same fields the
bring-up's staged directory holds, field by field against the HF converter's
oracle), that it touches no file -- a Hub checkpoint is a read-only cache
snapshot, and a copy staged from an older generator would be a second, staler
statement of the checkpoint's own metadata -- that a checkpoint which ships a
config keeps it, and which repo the tokenizer loads from.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from max.pipelines.architectures.mistral_large4_modulev3 import (
    checkpoint as ml4_checkpoint,
)
from max.pipelines.architectures.mistral_large4_modulev3 import (
    model_config as ml4_model_config,
)
from max.pipelines.architectures.mistral_large4_modulev3.arch import (
    mistral_large4_modulev3_arch,
)
from max.pipelines.architectures.mistral_large4_modulev3.model_config import (
    TOKENIZER_ARTIFACT_REVISION,
    TOKENIZER_REPO_ID,
    derive_hf_config,
    write_hf_config,
)
from max.pipelines.lib import MAXModelConfig, _hf_config
from max.pipelines.lib._hf_config import load_huggingface_config
from max.pipelines.weights.hf_utils import HuggingFaceRepo
from transformers import PretrainedConfig

_TESTDATA = Path(__file__).parent / "testdata"

# The generated config differs from the HF converter's oracle in two fields the
# oracle cannot pin (see `test_model_config.py`): it stamps another
# `transformers_version`, and it emits `architectures`, which the converter only
# serializes for a checkpoint that declares one and the published NVFP4 repo --
# which ships no config of its own -- therefore does not. `indexer_norm_eps` is
# MAX-side plumbing for the checkpoint's `dsa.norm_eps`, which the converter
# drops.
_ORACLE_IGNORED = frozenset(
    {"architectures", "transformers_version", "text_config.indexer_norm_eps"}
)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Every leaf of a nested config, keyed by its dotted path."""
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flat.update(
                _flatten(item, f"{prefix}{key}." if prefix else f"{key}.")
            )
    else:
        flat[prefix[:-1]] = value
    return flat


def _write_params(directory: Path, source: Path) -> Path:
    """Stages a checkpoint's ``params.json``, which is all a repo may ship."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "params.json"
    path.write_text(source.read_text())
    return path


def _fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drops the process-wide config cache, as a new process would start."""
    monkeypatch.setattr(_hf_config, "_config_cache", {})


@pytest.fixture
def nvfp4_checkpoint(tmp_path: Path) -> Path:
    """A directory holding the NVFP4 checkpoint's metadata and nothing else."""
    return _write_params(
        tmp_path / "checkpoint", _TESTDATA / "ml4_nvfp4_params.json"
    ).parent


def test_a_repo_with_no_config_json_loads_by_deriving_one(
    nvfp4_checkpoint: Path,
) -> None:
    """The model resolves from ``params.json`` alone, as a repo id does."""
    config = load_huggingface_config(HuggingFaceRepo(str(nvfp4_checkpoint)))

    assert config.architectures == ["MistralLarge4ForConditionalGeneration"]
    assert config.model_type == "mistral_large4"
    text = config.text_config
    assert text.num_hidden_layers == 72
    assert text.hidden_size == 6144
    assert text.n_routed_experts == 128
    assert text.num_experts_per_tok == 4
    # The shim flattens the text half onto the wrapper, which the DeepSeek-V3.2
    # spine this architecture forks reads.
    assert config.max_position_embeddings == 1048576


def test_a_derived_config_is_built_without_writing_anything(
    nvfp4_checkpoint: Path,
) -> None:
    """No file is staged beside the checkpoint, and none is read back.

    A repo-id checkpoint is a Hugging Face cache snapshot: its files are
    symlinks into shared blobs, a write can land on one of those blobs, and the
    cache may be read-only to begin with. The derived config is therefore built
    in memory, so the load leaves the checkpoint byte-for-byte as it found it.
    """
    before = sorted(
        (path.name, path.stat().st_mtime_ns)
        for path in nvfp4_checkpoint.iterdir()
    )

    load_huggingface_config(HuggingFaceRepo(str(nvfp4_checkpoint)))

    after = sorted(
        (path.name, path.stat().st_mtime_ns)
        for path in nvfp4_checkpoint.iterdir()
    )
    assert after == before
    assert not (nvfp4_checkpoint / "config.json").exists()


def _forbid_writes(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Fails the load on any write into *directory*, whatever the euid.

    The ``chmod`` the test below applies is what a read-only cache, PVC or
    non-root mount looks like, but root writes through it and some bazel
    sandboxes run as root, which leaves the permission bits proving nothing.
    Every write path the load could take under the checkpoint raises instead.
    """
    root = directory.resolve()

    def under(path: object) -> bool:
        try:
            return Path(str(path)).resolve().is_relative_to(root)
        except (OSError, ValueError):
            return False

    real_open = builtins.open

    def guarded_open(
        file: Any, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        if any(flag in mode for flag in "wax+") and under(file):
            raise AssertionError(f"wrote {file} while loading a checkpoint")
        return real_open(file, mode, *args, **kwargs)

    def guarded(method: str) -> Any:
        original = getattr(Path, method)

        def wrapper(self: Path, *args: Any, **kwargs: Any) -> Any:
            if under(self):
                raise AssertionError(f"wrote {self} while loading a checkpoint")
            return original(self, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(builtins, "open", guarded_open)
    for method in ("write_text", "write_bytes", "mkdir", "touch", "unlink"):
        monkeypatch.setattr(Path, method, guarded(method))


def test_a_read_only_checkpoint_still_loads(
    nvfp4_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only cache, PVC, or non-root mount cannot fail the derivation.

    Nothing is written, so there is nothing to fail: the same load that works
    on a writable directory works on one the process cannot write at all -- or
    would not be able to write even where the permission bits are no obstacle.
    """
    _forbid_writes(monkeypatch, nvfp4_checkpoint)
    params = nvfp4_checkpoint / "params.json"
    params.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    nvfp4_checkpoint.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    try:
        config = load_huggingface_config(HuggingFaceRepo(str(nvfp4_checkpoint)))
        assert config.text_config.num_hidden_layers == 72
    finally:
        nvfp4_checkpoint.chmod(stat.S_IRWXU)
        params.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_the_derived_config_is_the_one_a_prepared_directory_holds(
    nvfp4_checkpoint: Path, tmp_path: Path
) -> None:
    """A repo id and a prepared directory read the same config.

    The bring-up staged its export directory by running the family's generator
    over ``params.json``; a repo that ships no config must resolve to that same
    config on its own, so an export from a repo id and one from a prepared
    directory read the same metadata.
    """
    assert derive_hf_config(nvfp4_checkpoint / "params.json") == (
        write_hf_config(
            nvfp4_checkpoint / "params.json", tmp_path / "prepared.json"
        )
    )

    # And field by field against the oracle: the HF converter run on this
    # checkpoint's own ``params.json``.
    derived_fields = _flatten(
        json.loads((tmp_path / "prepared.json").read_text())
    )
    oracle = _flatten(
        json.loads((_TESTDATA / "ml4_nvfp4_hf_config.json").read_text())
    )
    assert not set(derived_fields) - set(oracle) - _ORACLE_IGNORED
    assert not set(oracle) - set(derived_fields) - _ORACLE_IGNORED
    differing = {
        key
        for key in set(derived_fields) & set(oracle)
        if key not in _ORACLE_IGNORED and derived_fields[key] != oracle[key]
    }
    assert not differing, differing


def test_a_repo_id_and_a_prepared_directory_load_the_same_config(
    nvfp4_checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both paths reach one config object, not just one generator output.

    A prepared directory ships ``config.json``; a repo id has none and the
    loader builds it. If the two constructions disagreed, an export from a repo
    id and one from a staged directory would compile different graphs.
    """
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    write_hf_config(nvfp4_checkpoint / "params.json", prepared / "config.json")

    from_repo_id = load_huggingface_config(
        HuggingFaceRepo(str(nvfp4_checkpoint))
    )
    _fresh_process(monkeypatch)
    from_directory = load_huggingface_config(HuggingFaceRepo(str(prepared)))

    assert type(from_repo_id) is type(from_directory)
    assert _comparable(from_repo_id) == _comparable(from_directory)


def _comparable(config: Any) -> dict[str, Any]:
    """A config's fields, without the provenance of where they were read.

    Two checkpoints are two directories, so ``name_or_path`` is their paths and
    disagrees by construction; that a config *has* one, and that it is the id
    the caller named, is pinned by
    :func:`test_a_derived_config_records_the_repo_it_came_from`.
    """
    return {
        key: value
        for key, value in config.to_dict().items()
        if not key.startswith("_")
        and key not in ("name_or_path", "transformers_version")
    }


def test_a_derived_config_records_the_repo_it_came_from(
    nvfp4_checkpoint: Path,
) -> None:
    """A derived config carries the two fields `from_pretrained` would stamp.

    Building the config in memory skips what `AutoConfig.from_pretrained` does
    last: record the checkpoint it read and, where the revision already names
    it, the commit. A reader uses `name_or_path` (a standalone quant config is
    skipped for an empty one), so a derived config that lost it would differ
    from the same checkpoint loaded through a prepared directory. A branch
    revision is left unresolved rather than reported as the branch name.
    """
    derived = load_huggingface_config(HuggingFaceRepo(str(nvfp4_checkpoint)))
    assert derived.name_or_path == str(nvfp4_checkpoint)
    assert derived._commit_hash is None

    sha = "bb6d00fe8ff7" + "0" * 28
    at_commit = load_huggingface_config(
        HuggingFaceRepo(str(nvfp4_checkpoint), revision=sha)
    )
    assert at_commit.name_or_path == str(nvfp4_checkpoint)
    assert at_commit._commit_hash == sha


def test_a_derived_config_is_rebuilt_on_every_load(
    nvfp4_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing persists, so a moved checkpoint is read, not remembered.

    A config staged beside the checkpoint would be the first load's, and every
    later process on the node would keep reading it however far the checkpoint
    moved. Here the second load sees the edit.
    """
    first = load_huggingface_config(HuggingFaceRepo(str(nvfp4_checkpoint)))
    assert first.text_config.vocab_size == 204800

    params = json.loads((nvfp4_checkpoint / "params.json").read_text())
    params["vocab_size"] = 204801
    (nvfp4_checkpoint / "params.json").write_text(json.dumps(params))
    _fresh_process(monkeypatch)

    second = load_huggingface_config(HuggingFaceRepo(str(nvfp4_checkpoint)))
    assert second.text_config.vocab_size == 204801


def test_max_resolves_the_architecture_from_the_derived_config(
    nvfp4_checkpoint: Path,
) -> None:
    """MAX's own config resolution, not just the loader, reads a repo id."""
    model = MAXModelConfig(model_path=str(nvfp4_checkpoint))

    assert model.architecture_name == "MistralLarge4ForConditionalGeneration"
    assert model.huggingface_config.text_config.vocab_size == 204800


def test_a_checkpoint_that_ships_a_config_json_keeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The base release publishes a ``config.json``; nothing derives over it."""
    directory = tmp_path / "base"
    directory.mkdir()
    published = (_TESTDATA / "ml4_base_config.json").read_text()
    (directory / "config.json").write_text(published)
    (directory / "params.json").write_text(
        (_TESTDATA / "ml4_base_params.json").read_text()
    )

    config = load_huggingface_config(HuggingFaceRepo(str(directory)))

    assert config.text_config.num_hidden_layers == 72
    assert (directory / "config.json").read_text() == published


def test_a_checkpoint_with_a_config_json_fetches_no_params_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Derivation is a fallback, so no other model pays for its lookup.

    A repo that has a config of its own never reaches the derivation: not a
    fetch, and not the import of the generator that would have answered it.
    """
    directory = tmp_path / "base"
    directory.mkdir()
    (directory / "config.json").write_text(
        (_TESTDATA / "ml4_base_config.json").read_text()
    )
    (directory / "params.json").write_text(
        (_TESTDATA / "ml4_base_params.json").read_text()
    )

    def unreachable(repo: Any, filename: str) -> Path:
        raise AssertionError(f"fetched {filename} for a repo that has a config")

    monkeypatch.setattr(_hf_config, "fetch_repo_file", unreachable)

    config = load_huggingface_config(HuggingFaceRepo(str(directory)))
    assert config.text_config.num_hidden_layers == 72


def test_a_config_json_auto_config_rejects_takes_the_raw_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config of its own always wins, even one AutoConfig cannot load.

    Derivation is the last resort: it runs only for a checkpoint with no config
    file at all, which the raw fallback's ``FileNotFoundError`` establishes. So
    a ``config.json`` with no ``model_type`` -- non-transformers, as far as the
    fallback can tell -- is wrapped and returned rather than passed over, even
    beside a ``params.json`` this family's generator accepts. Both halves of
    that matter: deriving here would answer for a checkpoint MAX cannot read,
    and fetching ``params.json`` would make every such load pay for a probe.
    """
    directory = tmp_path / "component"
    directory.mkdir()
    # AutoConfig refuses it ("Unrecognized model") and a derived Mistral Large 4
    # config would be accepted far too late, so the wrap is what makes the two
    # orderings distinguishable.
    (directory / "config.json").write_text(json.dumps({"latent_channels": 16}))
    (directory / "params.json").write_text(
        (_TESTDATA / "ml4_nvfp4_params.json").read_text()
    )

    def unreachable(repo: Any, filename: str) -> Path:
        raise AssertionError(
            f"fetched {filename} for a checkpoint that has a config"
        )

    monkeypatch.setattr(_hf_config, "fetch_repo_file", unreachable)

    config = load_huggingface_config(HuggingFaceRepo(str(directory)))

    assert type(config) is PretrainedConfig
    assert config.latent_channels == 16


def test_a_params_json_of_another_family_derives_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo with neither a config nor this family's params still refuses."""
    for tag, text in (
        ("another-family", json.dumps({"dim": 4096, "n_layers": 2})),
        ("malformed", "{not json"),
        ("not-an-object", json.dumps([1, 2, 3])),
    ):
        directory = tmp_path / tag
        directory.mkdir()
        (directory / "params.json").write_text(text)
        _fresh_process(monkeypatch)

        with pytest.raises(FileNotFoundError):
            load_huggingface_config(HuggingFaceRepo(str(directory)))

        assert sorted(path.name for path in directory.iterdir()) == [
            "params.json"
        ]


def test_another_family_s_checkpoint_imports_no_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The claim is data, so an unmatched checkpoint imports nothing.

    A registered generator lives in an architecture package. A checkpoint whose
    ``params.json`` lacks the family's keys is dropped before that import, which
    is what keeps every other model's load out of this code: an import attempt
    here would log the failure the registry reports.
    """
    directory = tmp_path / "other"
    directory.mkdir()
    (directory / "params.json").write_text(json.dumps({"dim": 4096}))
    monkeypatch.setattr(
        _hf_config,
        "_CONFIG_GENERATORS",
        [
            _hf_config._ConfigGenerator(
                "max.pipelines.architectures.no_such_module:derive",
                frozenset({"nope", "dsa", "kv_sharing", "ssmax_scaling"}),
            )
        ],
    )

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        with pytest.raises(FileNotFoundError):
            load_huggingface_config(HuggingFaceRepo(str(directory)))

    assert "Cannot import config generator" not in caplog.text


def test_the_registration_claims_this_family_s_params_keys() -> None:
    """The claim the shim registers is the generator's own family signature."""
    registered = [
        spec.params_keys
        for spec in _hf_config._CONFIG_GENERATORS
        if spec.spec.endswith(":derive_hf_config")
    ]

    assert registered == [ml4_model_config._PARAMS_SIGNATURE]


def test_an_empty_registry_says_so_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A process that imported MAX narrowly is told once, not per load."""
    directory = tmp_path / "checkpoint"
    _write_params(directory, _TESTDATA / "ml4_nvfp4_params.json")
    monkeypatch.setattr(_hf_config, "_CONFIG_GENERATORS", [])
    monkeypatch.setattr(_hf_config, "_no_generator_warned", False)

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        for _ in range(2):
            _fresh_process(monkeypatch)
            with pytest.raises(FileNotFoundError):
                load_huggingface_config(HuggingFaceRepo(str(directory)))

    assert caplog.text.count("No checkpoint config generator") == 1


def test_a_derived_config_fetches_the_params_json_it_derives_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint with no metadata of its own fetches its params.json.

    The weights download brings safetensors, never the checkpoint's metadata,
    so the derivation asks ``fetch_repo_file`` for the ``params.json`` -- the
    only way a repo-id serve gets one, since a downloaded snapshot holds no
    copy to read from disk -- and it derives from the file it got, writing
    nothing back into either directory.
    """
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    staged = tmp_path / "fetched" / "params.json"
    staged.parent.mkdir()
    staged.write_text((_TESTDATA / "ml4_nvfp4_params.json").read_text())
    fetched: list[tuple[str, str]] = []

    def fetch(repo: Any, filename: str) -> Path:
        fetched.append((repo.repo_id, filename))
        return staged

    monkeypatch.setattr(_hf_config, "fetch_repo_file", fetch)

    config = load_huggingface_config(HuggingFaceRepo(str(checkpoint)))

    assert config.text_config.num_hidden_layers == 72
    assert fetched == [(str(checkpoint), "params.json")]
    assert list(checkpoint.iterdir()) == []


def _candidates(model: Any) -> list[Path]:
    """The ``params.json`` candidates a reader walks, stopping where one exists."""
    walked: list[Path] = []
    for path in ml4_checkpoint.params_json_candidates(model):
        walked.append(path)
        if path.is_file():
            break
    return walked


def test_a_hub_serve_fetches_the_params_json_the_weights_never_bring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A downloaded snapshot holds the shards and no ``params.json``.

    The weights download fetches what the loader asks for, which is
    safetensors; ``params.json`` is metadata. Without fetching it, a fresh Hub
    serve captures no drafter taps and exports no drafter geometry, so this is
    the difference between a repo-id export and a staged directory.
    """
    snapshot = tmp_path / "hub" / "models--mistralai--Mistral-Large-4-NVFP4"
    snapshot.mkdir(parents=True)
    staged = tmp_path / "fetched" / "params.json"
    staged.parent.mkdir()
    staged.write_text((_TESTDATA / "ml4_nvfp4_params.json").read_text())
    fetched: list[tuple[str, str]] = []

    def fetch(repo: Any, filename: str) -> Path:
        fetched.append((repo.repo_id, filename))
        return staged

    monkeypatch.setattr(ml4_checkpoint, "fetch_repo_file", fetch)
    monkeypatch.setattr(
        ml4_checkpoint,
        "generate_local_model_path",
        lambda repo_id, revision="main": str(snapshot),
    )
    model = _ServedModel(
        model_path="mistralai/Mistral-Large-4-1T-A52B-NVFP4",
        weight_path=[Path("consolidated-00001-of-00032.safetensors")],
        repo=_OnlineRepo(repo_id="mistralai/Mistral-Large-4-1T-A52B-NVFP4"),
    )

    walked = _candidates(model)
    # The snapshot the weights land in holds none, so the last candidate tried
    # is the fetched file.
    assert walked == [snapshot / "params.json", staged]
    assert fetched == [
        ("mistralai/Mistral-Large-4-1T-A52B-NVFP4", "params.json")
    ]


def test_a_snapshot_that_holds_the_params_json_fetches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An offline, fully snapshotted cache resolves from disk alone."""
    snapshot = tmp_path / "hub" / "models--mistralai--Mistral-Large-4-NVFP4"
    _write_params(snapshot, _TESTDATA / "ml4_nvfp4_params.json")

    def unreachable(repo: Any, filename: str) -> Path:
        raise AssertionError(f"fetched {filename} for a repo that has it")

    monkeypatch.setattr(ml4_checkpoint, "fetch_repo_file", unreachable)
    monkeypatch.setattr(
        ml4_checkpoint,
        "generate_local_model_path",
        lambda repo_id, revision="main": str(snapshot),
    )
    model = _ServedModel(
        model_path="mistralai/Mistral-Large-4-1T-A52B-NVFP4",
        weight_path=[Path("consolidated-00001-of-00032.safetensors")],
        repo=_OnlineRepo(repo_id="mistralai/Mistral-Large-4-1T-A52B-NVFP4"),
    )

    assert _candidates(model) == [snapshot / "params.json"]


def test_the_tokenizer_loads_the_pinned_base_repo_artifacts() -> None:
    """Both releases tokenize with one pinned set of artifacts.

    The NVFP4 release's own ``tokenizer.json`` is the base release's, byte for
    byte, and the native tokenizer pins the base repo's revision; MAX has to
    read the same commit or the two engines frame one prompt differently.
    """
    served = _ServedModel(
        model_path="mistralai/Mistral-Large-4-1T-A52B-NVFP4",
        weight_path=[],
        repo=_OnlineRepo(repo_id="mistralai/Mistral-Large-4-1T-A52B-NVFP4"),
        huggingface_model_revision="main",
    )

    assert mistral_large4_modulev3_arch.tokenizer_source(served) == (
        TOKENIZER_REPO_ID,
        TOKENIZER_ARTIFACT_REVISION,
    )
    assert TOKENIZER_REPO_ID == "mistralai/Mistral-Large-4-1T-A52B"
    # The pin is the base repo's own locked revision, not a branch.
    assert len(TOKENIZER_ARTIFACT_REVISION) == 40


def test_the_tokenizer_pin_is_the_native_tokenizers() -> None:
    """MAX's pin is the one the native tokenizer resolves the family to.

    Two engines booting one tokenizer means one revision; a drift between the
    two pins is invisible until the two engines frame the same prompt
    differently, so the native tokenizer's own table is read here rather than
    trusted. That table is a JSON object keyed by the repo it pins, in
    ``tokenizer/crates/tokenizer-core/data/pinned_revisions.json``.
    """
    source = os.environ.get("ML4_TOKENIZER_PINNED_REVISIONS")
    if not source:
        pytest.skip(
            "set ML4_TOKENIZER_PINNED_REVISIONS to pinned_revisions.json"
        )

    pins = json.loads(Path(source).read_text())
    assert pins.get(TOKENIZER_REPO_ID) == TOKENIZER_ARTIFACT_REVISION, (
        f"the native tokenizer pins {pins.get(TOKENIZER_REPO_ID)} for"
        f" {TOKENIZER_REPO_ID}, MAX pins {TOKENIZER_ARTIFACT_REVISION}"
    )
    # The pin belongs to the repo that owns the artifacts; a sibling release
    # that reuses them is an alias of that repo, not a second entry.
    assert not [
        repo
        for repo in pins
        if repo.startswith(TOKENIZER_REPO_ID) and repo != TOKENIZER_REPO_ID
    ]


def test_a_tokenizer_pin_overruling_a_requested_revision_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A caller who asked for another revision sees that the pin won."""
    served = _ServedModel(
        model_path="mistralai/Mistral-Large-4-1T-A52B",
        weight_path=[],
        repo=_OnlineRepo(repo_id="mistralai/Mistral-Large-4-1T-A52B"),
        huggingface_model_revision="b" * 40,
    )

    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        source = mistral_large4_modulev3_arch.tokenizer_source(served)

    assert source == (TOKENIZER_REPO_ID, TOKENIZER_ARTIFACT_REVISION)
    assert TOKENIZER_ARTIFACT_REVISION in caplog.text
    assert "b" * 40 in caplog.text


def test_every_other_architecture_tokenizes_its_own_repo() -> None:
    """The default is the checkpoint's repo, so no other model moves."""
    unaliased = replace(mistral_large4_modulev3_arch, tokenizer_artifacts=None)
    served = _ServedModel(
        model_path="mistralai/Mistral-Large-4-1T-A52B-NVFP4",
        weight_path=[],
        repo=_OnlineRepo(repo_id="mistralai/Mistral-Large-4-1T-A52B-NVFP4"),
        huggingface_model_revision="abc123",
    )

    assert unaliased.tokenizer_source(served) == (
        "mistralai/Mistral-Large-4-1T-A52B-NVFP4",
        "abc123",
    )


class _OnlineRepo:
    """A weights repo handle that names no local directory.

    ``HuggingFaceRepo`` with the shape a Hub serve builds: an id, a revision and
    no on-disk path. ``local_path`` is never read for an online repo; the
    resolver's protocol declares it because its local branch reads it.
    """

    def __init__(self, repo_id: str, revision: str = "main") -> None:
        self.repo_type = "online"
        self.repo_id = repo_id
        self.revision = revision
        self.subfolder: str | None = None
        self.local_path = ""

    def __str__(self) -> str:
        return self.repo_id


class _ServedModel:
    """A served checkpoint config, as the resolvers here read it.

    The fields mirror ``MAXModelConfig``'s; the real config cannot be built for
    a repo id here without resolving its weights.
    """

    def __init__(
        self,
        model_path: str,
        weight_path: list[Path],
        repo: _OnlineRepo,
        huggingface_model_revision: str = "main",
    ) -> None:
        self.model_path = model_path
        self.weight_path = weight_path
        self.huggingface_weight_repo = repo
        self.huggingface_model_revision = huggingface_model_revision
