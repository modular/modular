# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import logging
import re

import hf_repo_lock
import pytest
from pytest_mock import MockerFixture
from test_common.mocks import DummyPipelineConfig
from test_common.pipeline_model_dummy import DUMMY_LLAMA_ARCH

logger = logging.getLogger("max.pipelines")

EXAMPLE_KEY = "000EXAMPLE-for-unit-test/repo"
EXAMPLE_VALUE = "0123456789abcdef0123456789abcdef01234567"
EXAMPLE_NONEXISTENT_KEY = "000EXAMPLE-for-unit-test/nonexistent"


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


def test_revision_for_hf_repo_local_path() -> None:
    assert hf_repo_lock.revision_for_hf_repo("/path/to/model") is None


def test_revision_for_hf_repo_invalid_format() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid Hugging Face repository ID: '000EXAMPLE-for-unit-test/repo/subrepo'.  "
            "It must be in the format 'org/model'."
        ),
    ):
        hf_repo_lock.revision_for_hf_repo(
            "000EXAMPLE-for-unit-test/repo/subrepo"
        )


def test_revision_for_hf_repo_no_local_leakage(mocker: MockerFixture) -> None:
    list_mock = mocker.patch("huggingface_hub.list_repo_refs")
    # A variety of paths that "could" be local (and there's no way they're
    # Hugging Face repos).
    for local_path in [
        "pathtomodel",
        "/pathtomodel",
        "/path/to/model",
        "path/to/model",
    ]:
        # Don't care what this raises or returns, we just want to make sure it doesn't
        # call list_repo_refs.
        try:
            _ = hf_repo_lock.revision_for_hf_repo(local_path)
        except ValueError:
            pass
    list_mock.assert_not_called()


def test_apply_to_config() -> None:
    config = DummyPipelineConfig(
        model_path=EXAMPLE_KEY,
        max_batch_size=None,
        max_length=None,
        device_specs=[],
        quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
    )
    assert config.model.huggingface_model_revision == "main"
    hf_repo_lock.apply_to_config(config)
    assert config.model.huggingface_model_revision == EXAMPLE_VALUE


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
