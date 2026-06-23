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

from mojodoc_api_href import (
    MAX_KERNELS_ORIGIN,
    MOJOLANG_ORIGIN,
    resolve_api_href,
)


def test_std_path_kernel_tarball_uses_absolute_mojolang() -> None:
    assert resolve_api_href(
        "/std/collections/List",
        hosted_on_mojolang=False,
    ) == (f"{MOJOLANG_ORIGIN}/docs/std/collections/List")


def test_std_path_mojolang_tarball_root_relative() -> None:
    assert resolve_api_href(
        "/std/collections/List",
        hosted_on_mojolang=True,
    ) == ("/docs/std/collections/List")


def test_std_alias_fragment_kernel_tarball() -> None:
    assert resolve_api_href(
        "/std/builtin/#mutorigin",
        hosted_on_mojolang=False,
    ) == (f"{MOJOLANG_ORIGIN}/docs/std/builtin/#mutorigin")


def test_kernels_layout_maps_to_mojolang_docs_layout() -> None:
    assert resolve_api_href(
        "/kernels/layout/tile_layout/TensorLayout",
        hosted_on_mojolang=False,
    ) == (f"{MOJOLANG_ORIGIN}/docs/layout/tile_layout/TensorLayout")


def test_kernels_layout_hosted_on_mojolang_root_relative() -> None:
    assert resolve_api_href(
        "/kernels/layout/tile_layout/TensorLayout",
        hosted_on_mojolang=True,
    ) == ("/docs/layout/tile_layout/TensorLayout")


def test_kernels_layout_exact_module() -> None:
    assert resolve_api_href(
        "/kernels/layout",
        hosted_on_mojolang=False,
    ) == (f"{MOJOLANG_ORIGIN}/docs/layout")


def test_kernels_other_kernel_tarball_root_relative() -> None:
    assert resolve_api_href(
        "/kernels/linalg/foo/Bar",
        hosted_on_mojolang=False,
    ) == ("/max/api/kernels/linalg/foo/Bar")


def test_kernels_other_mojolang_tarball_uses_absolute_max_kernels() -> None:
    """Kernel cross-links from mojolang-hosted Markdown (e.g. layout)
    must be absolute so they resolve to docs.modular.com."""
    assert resolve_api_href(
        "/kernels/linalg/foo/Bar",
        hosted_on_mojolang=True,
    ) == (f"{MAX_KERNELS_ORIGIN}/max/api/kernels/linalg/foo/Bar")


def test_layout_path_hosted_on_mojolang() -> None:
    assert resolve_api_href(
        "/layout/tile_layout/X",
        hosted_on_mojolang=True,
    ) == ("/docs/layout/tile_layout/X")


def test_extensibility_tensor_kernel_tarball_root_relative() -> None:
    assert resolve_api_href(
        "/extensibility/tensor/foo/Bar",
        hosted_on_mojolang=False,
    ) == ("/max/api/kernels/extensibility/tensor/foo/Bar")


def test_extensibility_tensor_mojolang_tarball_uses_absolute_max_kernels() -> (
    None
):
    assert resolve_api_href(
        "/extensibility/tensor/foo/Bar",
        hosted_on_mojolang=True,
    ) == (f"{MAX_KERNELS_ORIGIN}/max/api/kernels/extensibility/tensor/foo/Bar")


def test_empty_path() -> None:
    assert resolve_api_href("") == ""
    assert resolve_api_href(None) == ""


def test_fragment_only_after_split() -> None:
    assert resolve_api_href("#frag", hosted_on_mojolang=True) == "#frag"


def test_normalize_missing_leading_slash() -> None:
    assert resolve_api_href(
        "std/builtin/Int",
        hosted_on_mojolang=False,
    ) == (f"{MOJOLANG_ORIGIN}/docs/std/builtin/Int")
