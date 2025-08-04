# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging

import hf_repo_lock
import pytest
from test_common.mocks import DummyPipelineConfig
from test_common.pipeline_model_dummy import DUMMY_LLAMA_ARCH

logger = logging.getLogger("max.pipelines")

EXAMPLE_KEY = "000EXAMPLE-for-unit-test"
EXAMPLE_VALUE = "0123456789abcdef0123456789abcdef01234567"
EXAMPLE_NONEXISTENT_KEY = "000EXAMPLE-for-unit-test-nonexistent"


def test_load_db() -> None:
    db = hf_repo_lock.load_db()
    assert db[EXAMPLE_KEY] == EXAMPLE_VALUE
    assert EXAMPLE_NONEXISTENT_KEY not in db


def test_revision_for_hf_repo(caplog: pytest.LogCaptureFixture) -> None:
    assert hf_repo_lock.revision_for_hf_repo(EXAMPLE_KEY) == EXAMPLE_VALUE

    with caplog.at_level(logging.WARNING):
        assert (
            hf_repo_lock.revision_for_hf_repo(EXAMPLE_NONEXISTENT_KEY) is None
        )

    assert len(caplog.records) == 1
    warning_record = caplog.records[0]
    assert warning_record.levelname == "WARNING"
    assert (
        f"No lock revision available for Hugging Face repo {EXAMPLE_NONEXISTENT_KEY!r}"
        in warning_record.message
    )
    assert (
        "Add a row to hf-repo-lock.tsv to resolve this error"
        in warning_record.message
    )


def test_apply_to_config() -> None:
    config = DummyPipelineConfig(
        model_path=EXAMPLE_KEY,
        max_batch_size=None,
        max_length=None,
        device_specs=[],
        quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
    )
    assert config.model_config.huggingface_model_revision == "main"
    hf_repo_lock.apply_to_config(config)
    assert config.model_config.huggingface_model_revision == EXAMPLE_VALUE


def test_apply_to_config_raises_on_missing_revision() -> None:
    config = DummyPipelineConfig(
        model_path=EXAMPLE_NONEXISTENT_KEY,
        max_batch_size=None,
        max_length=None,
        device_specs=[],
        quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
    )
    with pytest.raises(
        ValueError, match="No locked revision found for model repository"
    ):
        hf_repo_lock.apply_to_config(config)
