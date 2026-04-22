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

import functools
import os
from typing import Any

from packaging.tags import Tag
from packaging.utils import parse_wheel_filename
from utils import assert_keys

# URL -> sha256 in format 'sha256:<hash>'
_MISSING_HASHES: dict[str, str] = {
    "https://download.pytorch.org/whl/triton_rocm-3.6.0-cp310-cp310-linux_x86_64.whl": "sha256:043c2d44e24632cb5aba814b547731d8b46a58a7a69818720221d0e406600605",
    "https://download.pytorch.org/whl/triton_rocm-3.6.0-cp311-cp311-linux_x86_64.whl": "sha256:3286c59eb97e65ab705e207689b6a47807cb73a27ce53e9e774e46bab01318fe",
    "https://download.pytorch.org/whl/triton_rocm-3.6.0-cp312-cp312-linux_x86_64.whl": "sha256:cff15082784c7056b0af9347770e034ab0a8ccbce0642723ddc8c8de1bd6af3f",
    "https://download.pytorch.org/whl/triton_rocm-3.6.0-cp313-cp313-linux_x86_64.whl": "sha256:d43b44f045d7f78d1dfe03b2debce36e0d756041a853633a2677ce5a890a269e",
    "https://download.pytorch.org/whl/triton_rocm-3.6.0-cp313-cp313t-linux_x86_64.whl": "sha256:bc37c5382ba637f00738729cbeaa21c4200255da530bf0dacbfa1c7fa4fa433a",
    "https://download.pytorch.org/whl/triton_rocm-3.6.0-cp314-cp314-linux_x86_64.whl": "sha256:f77e6da822a8a76e097a061e61a7fef8ae0a33e8b6989498736e16ed120fc6f8",
    "https://download.pytorch.org/whl/triton_rocm-3.6.0-cp314-cp314t-linux_x86_64.whl": "sha256:1ef3560cf10d120da52ef6273033d952726a08893cc4f264c45600402cc608d7",
}


class Download:
    def __init__(self, blob: dict[str, Any]):
        assert_keys(
            blob,
            required={
                "url",
            },
            optional={"upload-time", "size", "hash"},
        )

        # NOTE: Hashes can be missing if the registry is missing them, but we need them for bazel downloading
        # https://github.com/pytorch/pytorch/issues/173099
        download_hash = blob.get("hash") or _MISSING_HASHES[blob["url"]]
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
