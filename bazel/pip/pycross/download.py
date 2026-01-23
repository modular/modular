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

_MISSING_HASHES = {
    "https://download.pytorch.org/whl/triton-3.5.0-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl": "sha256:253c88932fd558df52f86c663729e87cd38a19a7151581be5bd5f4bfd58f869a",
    "https://download.pytorch.org/whl/triton-3.5.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": "sha256:bba3ea19cc181953483959988f4fd793a75983ebfecf6547d583a8806ab8dcfc",
    "https://download.pytorch.org/whl/triton-3.5.0-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl": "sha256:498125ea35ce48969a46fa7939d7e6802540a13957a26230d66a38b7ace7df68",
    "https://download.pytorch.org/whl/triton-3.5.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": "sha256:263881cac8477df84d3964be765b7fac17dd3ffa0ad6fcb030b841bd74f9408b",
    "https://download.pytorch.org/whl/triton-3.5.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl": "sha256:9b82b46df35ae9e0b85a382d99a076e6ebea23a3a5dbaca7dc24a7571e6bebad",
    "https://download.pytorch.org/whl/triton-3.5.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": "sha256:76f8651a5e38c2a7da6fa2b2e41cbc00a5a32cb52bf3f520113fe90b723a310d",
    "https://download.pytorch.org/whl/triton-3.5.0-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl": "sha256:05e145b51a53573bff260431ff40fadce0838ad9928c5ee1883b53d59884e198",
    "https://download.pytorch.org/whl/triton-3.5.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": "sha256:b6f6db89501a6dc4a492ff281460c1b15563420bc90934770aa6a7b80fd51c95",
    "https://download.pytorch.org/whl/triton-3.5.0-cp313-cp313t-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl": "sha256:8743eeb3f383ad3a33d508d13cc368abaa5bc6c06f61e80503aa7e004f49e24d",
    "https://download.pytorch.org/whl/triton-3.5.0-cp313-cp313t-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": "sha256:581a43b2da8048db6fb73dbe8a2fe6c922f2c577ee65d23b3b76ff616737a7bc",
    "https://download.pytorch.org/whl/triton-3.5.0-cp314-cp314-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl": "sha256:4e1a856d1d731734ee9ed62dde548f342d34c988b8d7e235bf2037e428de2258",
    "https://download.pytorch.org/whl/triton-3.5.0-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": "sha256:8d98b78a83e910256ec703486b8c275ec53975e5ac4a8cb7ce07c696e08f6b5a",
    "https://download.pytorch.org/whl/triton-3.5.0-cp314-cp314t-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl": "sha256:0a97fdbcf6b71d73f4c04f5819ca12786ee40e4a83144bebae2616d7c0942182",
    "https://download.pytorch.org/whl/triton-3.5.0-cp314-cp314t-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": "sha256:ff47e20dbefdaaa2c6968bdcc69633871bda425f2801c67bc8d8df472f1d12d4",
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

        # TODO: Hash should be required https://github.com/pytorch/pytorch/issues/173099
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
