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
from pip_platform import Platform


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


def _package(
    wheel_files: list[str], platform_keys: list[tuple[str, str]]
) -> Package:
    """A package with the given wheels, on the default Python (3.13) only."""
    return Package(
        {
            "name": "foo",
            "version": "1.0",
            "wheels": [
                {"url": f"https://example.com/{name}", "hash": "sha256:00"}
                for name in wheel_files
            ],
        },
        {"foo": "1.0"},
        platforms={
            Platform(python_version="3.13", operating_system=os_, arch=arch)
            for os_, arch in platform_keys
        },
    )


_MACS = [("darwin", "arm64"), ("darwin", "x86_64")]
_ARM_CONSTRAINT = ":_env_python_3.13_aarch64-apple-darwin"
_INTEL_CONSTRAINT = ":_env_python_3.13_x86_64-apple-darwin"


def test_arm64_only_mac_wheel_is_not_unconditional() -> None:
    package = _package(["foo-1.0-cp313-cp313-macosx_14_0_arm64.whl"], _MACS)

    text, downloads = package.render()

    assert len(downloads) == 1
    assert f'"{_ARM_CONSTRAINT}": ' in text
    # Intel Macs get no wheel, rather than the Apple silicon one.
    assert _INTEL_CONSTRAINT not in text
    assert "actual = select(" in text
    assert 'actual = "@' not in text


def test_universal2_mac_wheel_matches_both_mac_architectures() -> None:
    # Linux has no wheel, so the Mac wheel can't be the one answer for all.
    package = _package(
        ["foo-1.0-cp313-cp313-macosx_14_0_universal2.whl"],
        [*_MACS, ("linux", "x86_64")],
    )

    text, _ = package.render()

    assert f'"{_ARM_CONSTRAINT}": ' in text
    assert f'"{_INTEL_CONSTRAINT}": ' in text


def test_pure_python_wheel_keeps_unconditional_alias() -> None:
    package = _package(["foo-1.0-py3-none-any.whl"], _MACS)

    text, _ = package.render()

    assert 'actual = "@' in text
    assert "select(" not in text
