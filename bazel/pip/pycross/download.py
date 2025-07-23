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

import functools
import os
from typing import Any

from packaging.tags import Tag
from packaging.utils import parse_wheel_filename
from utils import assert_keys


class Download:
    def __init__(self, blob: dict[str, Any]):
        assert_keys(
            blob, required={"hash", "url"}, optional={"upload-time", "size"}
        )

        download_hash = blob["hash"]
        assert download_hash.startswith("sha256:")
        self.hash = download_hash[len("sha256:") :]
        self.url = blob["url"]

        self.filename = os.path.basename(self.url).replace("%2B", "+")
        self.is_wheel = self.filename.endswith(".whl")
        filename_without_ext = os.path.splitext(self.filename)[0]
        filename_without_ext = filename_without_ext.removesuffix(".tar")

        self.name = (
            "pycross_lock_file_"
            + ("wheel_" if self.is_wheel else "sdist_")
            + filename_without_ext.replace("-", "_").replace("+", "_").lower()
        )

    def __repr__(self) -> str:
        return f"Download(filename={self.filename!r}, url={self.url!r}"

    def __lt__(self, other: "Download") -> bool:
        return self.name < other.name

    def __hash__(self) -> int:
        return hash((self.name, self.url, self.hash))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Download):
            return NotImplemented
        return self.__dict__ == other.__dict__

    @functools.cached_property
    def tags(self) -> set[Tag]:
        if not self.is_wheel:
            raise NotImplementedError(
                "Tags are only supported for wheels.", self.filename
            )

        wheel_info = parse_wheel_filename(self.filename)
        return {tag for tag in wheel_info[3]}

    def render(self) -> str:
        return f"""\
    maybe(
        http_file,
        name = "{self.name}",
        urls = [
            "{self.url}",
        ],
        sha256 = "{self.hash}",
        downloaded_file_path = "{self.filename}",
    )
"""
