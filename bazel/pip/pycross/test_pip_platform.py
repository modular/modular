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
from pip_platform import Platform


@pytest.mark.parametrize(
    "url, platforms",
    [
        (
            "https://files.pythonhosted.org/packages/6b/7a/032d40ea0eda6d9cb0456e0b9b3d27a4c9fb8c2e4404cfe515ee5f486b93/nixl-0.2.0-cp310-cp310-manylinux_2_28_x86_64.whl",
            [
                Platform(
                    python_version="3.10",
                    operating_system="linux",
                    arch="x86_64",
                )
            ],
        ),
        (
            "https://files.pythonhosted.org/packages/6b/7a/032d40ea0eda6d9cb0456e0b9b3d27a4c9fb8c2e4404cfe515ee5f486b93/nixl-0.2.0-cp312-cp312-manylinux_2_28_aarch64.whl",
            [
                Platform(
                    python_version="3.12",
                    operating_system="linux",
                    arch="aarch64",
                )
            ],
        ),
        (
            "https://files.pythonhosted.org/packages/a5/45/30bb92d442636f570cb5651bc661f52b610e2eec3f891a5dc3a4c3667db0/aiofiles-24.1.0-py3-none-any.whl",
            [
                Platform(
                    python_version="3.12",
                    operating_system="linux",
                    arch="aarch64",
                ),
                Platform(
                    python_version="3.11",
                    operating_system="linux",
                    arch="x86_64",
                ),
                Platform(
                    python_version="3.9",
                    operating_system="darwin",
                    arch="arm64",
                ),
            ],
        ),
        (
            "https://files.pythonhosted.org/packages/45/8e/4297556be5a07b713bb42dde0f748354de9a6918dee251c0e6bdcda341e7/kaleido-0.2.1-py2.py3-none-macosx_11_0_arm64.whl",
            [
                Platform(
                    python_version="3.9",
                    operating_system="darwin",
                    arch="arm64",
                ),
            ],
        ),
        (
            "https://files.pythonhosted.org/packages/d5/00/40f760cc27007912b327fe15bf6bfd8eaecbe451687f72a8abc587d503b3/Brotli-1.1.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl",
            [
                Platform(
                    python_version="3.10",
                    operating_system="linux",
                    arch="x86_64",
                ),
            ],
        ),
    ],
)
def test_platform_tags_match(platforms: list[Platform], url: str) -> None:
    download = Download({"url": url, "hash": "sha256:deadbeef"})
    for platform in platforms:
        assert platform.is_compatible_with(download), (
            f"Expected {platform.tags} to be compatible with {download.tags}"
        )
