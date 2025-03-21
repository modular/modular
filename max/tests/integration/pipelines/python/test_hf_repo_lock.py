# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from unittest import mock

import hf_repo_lock
import pytest
from max import pipelines

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


@mock.patch("huggingface_hub.repo_exists", return_value=True)
@pytest.mark.skip("TODO: AITLIB-280")
def test_apply_to_config(repo_exists_mock: mock.Mock) -> None:
    config = pipelines.PipelineConfig(model_path=EXAMPLE_KEY)
    assert config.model_config.huggingface_revision == "main"
    hf_repo_lock.apply_to_config(config)
    assert config.model_config.huggingface_revision == EXAMPLE_VALUE
