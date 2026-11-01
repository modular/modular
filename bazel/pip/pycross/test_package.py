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

from package import Package


def test_skips_only_listed_optional_deps() -> None:
    all_versions = {"host": "1.0.0", "plugin": "1.0.0", "extra": "1.0.0"}
    package = Package(
        {
            "name": "host",
            "version": "1.0.0",
            "optional-dependencies": {
                "plugins": [{"name": "plugin"}],
                "extras": [{"name": "extra"}],
            },
        },
        all_versions,
        skip_optional_deps={("host", "plugin")},
    )

    assert [dep.name for dep in package.dependencies] == ["extra@1.0.0"]
