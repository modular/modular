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
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp313-cp313-manylinux_2_28_aarch64.whl": "sha256:cec113fb60f0fe9d2c04d8061d560ddae3f8bf782984c1e7b993083608d70b91",
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp313-cp313-manylinux_2_28_x86_64.whl": "sha256:95bdbad2f0786bd448932e4b72a6404aa22290db6ec954e24edfa6d33c833c8f",
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp313-cp313-win_amd64.whl": "sha256:cf23236e9deed7d3510d14d9b9592d75d272ef7b35bbfee31a02bea339c73971",
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp314-cp314-manylinux_2_28_aarch64.whl": "sha256:0ad188cdfa7a5ccc67366c8e80269d32b45ebffc759c282d121516922405662a",
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp314-cp314-manylinux_2_28_x86_64.whl": "sha256:e231302a457298d0236f7bde31082568f6cd0613b66b4eb46849e8ad53c2e38d",
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp314-cp314-win_amd64.whl": "sha256:590b2a2b53ef295dcfcd703df17929bd579705bb0e42d9e43c462fe0f0b1e088",
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp314-cp314t-manylinux_2_28_aarch64.whl": "sha256:bddf05e9306ee873ae31a616a494c41e2c213406f76b60e5076c68555f27831c",
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp314-cp314t-manylinux_2_28_x86_64.whl": "sha256:4acf984b81a9e17c0e1d7cd47acd603eb03b484bfedd116271fa778dd34e25a2",
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp314-cp314t-win_amd64.whl": "sha256:9612d7ba74c7cb590b028d9c95ef619fcc1c903a5ff19e277b84807466c34cdd",
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp315-cp315-manylinux_2_28_aarch64.whl": "sha256:5e18ff87b8d29de21f8e49b9bdffe020b8aa7e8349ede49490d847a61cb534c7",
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp315-cp315-manylinux_2_28_x86_64.whl": "sha256:38fba3d5e731d7c8916f90b2ae3bd972ca1790db02970b9eb0940743cda5bc96",
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp315-cp315t-manylinux_2_28_aarch64.whl": "sha256:4bbc72efd1e9c38e3910626514275cc0de694bbbba81cf03a14e0f1f8b01ac10",
    "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp315-cp315t-manylinux_2_28_x86_64.whl": "sha256:5a7fe6ccfa0f17ab9e072e11d88edf6d7c8b7e5323daeaabccdf78dd4d50f8d2",
    "https://download-r2.pytorch.org/whl/cu130/torchvision-0.28.0%2Bcu130-cp310-cp310-win_amd64.whl": "sha256:e9d7fbbc4026d920d4f720b7da790a2f90be370c32e89a64af2c393acb0918c7",
    "https://download-r2.pytorch.org/whl/cu130/torchvision-0.28.0%2Bcu130-cp311-cp311-win_amd64.whl": "sha256:0c6921bb5e3e58d926f80ff894739c8b9b0e72fed59a062e19c934f13dfd53ea",
    "https://download-r2.pytorch.org/whl/cu130/torchvision-0.28.0%2Bcu130-cp312-cp312-win_amd64.whl": "sha256:5ae0ff50fcb43a189250a5fdb36a7dc780ea5caff469653cab1e3f32ed5d385c",
    "https://download-r2.pytorch.org/whl/cu130/torchvision-0.28.0%2Bcu130-cp313-cp313-win_amd64.whl": "sha256:6ac4fadfd4f908f2983a46478d4ef56735b579da57ec215494b9812b6ebd3526",
    "https://download-r2.pytorch.org/whl/cu130/torchvision-0.28.0%2Bcu130-cp314-cp314-win_amd64.whl": "sha256:a51e85f6133741ff89a941497dd8dbb208707c68f472c1983ae8dbed728876e1",
    "https://download-r2.pytorch.org/whl/cu130/torchvision-0.28.0%2Bcu130-cp314-cp314t-win_amd64.whl": "sha256:33200e879dd8a9a14b073c442fa127276e4fc69b40b1c8ec874bdb0e06701f1f",
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
