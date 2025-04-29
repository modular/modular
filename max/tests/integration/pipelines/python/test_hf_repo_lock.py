# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging

import hf_repo_lock
import pytest
from test_common.mocks import DummyPipelineConfig
from test_common.pipeline_model_dummy import DUMMY_ARCH

logger = logging.getLogger("max.pipelines")

EXAMPLE_KEY = "000EXAMPLE-for-unit-test"
EXAMPLE_VALUE = "0123456789abcdef0123456789abcdef01234567"
EXAMPLE_NONEXISTENT_KEY = "000EXAMPLE-for-unit-test-nonexistent"


def test_load_db() -> None:
    db = hf_repo_lock.load_db()
    assert db[EXAMPLE_KEY] == EXAMPLE_VALUE
    assert EXAMPLE_NONEXISTENT_KEY not in db


def test_revision_for_hf_repo() -> None:
    assert hf_repo_lock.revision_for_hf_repo(EXAMPLE_KEY) == EXAMPLE_VALUE
    with pytest.raises(KeyError):
        hf_repo_lock.revision_for_hf_repo(EXAMPLE_NONEXISTENT_KEY)


def test_apply_to_config() -> None:
    config = DummyPipelineConfig(
        model_path=EXAMPLE_KEY,
        max_batch_size=None,
        max_length=None,
        device_specs=[],
        quantization_encoding=DUMMY_ARCH.default_encoding,
    )
    assert config.model_config.huggingface_model_revision == "main"
    hf_repo_lock.apply_to_config(config)
    assert config.model_config.huggingface_model_revision == EXAMPLE_VALUE
