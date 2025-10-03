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

import pytest
from download import Download
from packaging.tags import Tag


def test_throws_missing_keys() -> None:
    with pytest.raises(ValueError, match="Missing required keys in blob"):
        Download({})


def test_initialize_wheel() -> None:
    blob = {
        "hash": "sha256:1234567890abcdef",
        "url": "https://example.com/package_thing-1.0.0-py3-none-any.whl",
    }
    download = Download(blob)
    assert download.hash == "1234567890abcdef"
    assert (
        download.url
        == "https://example.com/package_thing-1.0.0-py3-none-any.whl"
    )
    assert download.filename == "package_thing-1.0.0-py3-none-any.whl"
    assert download.is_wheel is True
    assert (
        download.name
        == "pycross_lock_file_wheel_package_thing_1.0.0_py3_none_any"
    )
    assert download.tags == {Tag("py3", "none", "any")}

    assert (
        download.render()
        == """\
    maybe(
        http_file,
        name = "pycross_lock_file_wheel_package_thing_1.0.0_py3_none_any",
        urls = [
            "https://example.com/package_thing-1.0.0-py3-none-any.whl",
        ],
        sha256 = "1234567890abcdef",
        downloaded_file_path = "package_thing-1.0.0-py3-none-any.whl",
    )
"""
    )


def test_initialize_sdist() -> None:
    blob = {
        "hash": "sha256:abcdef1234567890",
        "url": "https://example.com/package_thing-1.0.0.tar.gz",
    }
    download = Download(blob)
    assert download.hash == "abcdef1234567890"
    assert download.url == "https://example.com/package_thing-1.0.0.tar.gz"
    assert download.filename == "package_thing-1.0.0.tar.gz"
    assert download.is_wheel is False
    assert download.name == "pycross_lock_file_sdist_package_thing_1.0.0"
    assert (
        download.render()
        == """\
    maybe(
        http_file,
        name = "pycross_lock_file_sdist_package_thing_1.0.0",
        urls = [
            "https://example.com/package_thing-1.0.0.tar.gz",
        ],
        sha256 = "abcdef1234567890",
        downloaded_file_path = "package_thing-1.0.0.tar.gz",
    )
"""
    )

    with pytest.raises(
        NotImplementedError, match="Tags are only supported for wheels."
    ):
        _ = download.tags
