#!/usr/bin/env python3
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

"""Resolve ``mojo doc`` JSON ``path`` fields to hyperlinks.

``mojo doc`` emits logical paths (e.g. ``/std/builtin/Int`` for stdlib types,
``/kernels/...`` for kernel packages). This module is the single place that
knows the published site layout and rewrites those paths into hyperlinks.

The ``hosted_on_mojolang`` flag chooses root-relative vs absolute, so that
each href works regardless of where the rendered Markdown lives:

- Mojolang-hosted (stdlib, layout): std/layout hrefs are root-relative;
  cross-site kernel hrefs are absolute ``https://docs.modular.com/...``.
- Docs.modular.com-hosted (other kernels): kernel hrefs are root-relative;
  cross-site std/layout hrefs are absolute ``https://mojolang.org/...``."""

from __future__ import annotations

MOJOLANG_ORIGIN = "https://mojolang.org"
MOJOLANG_PATH_PREFIX = "/docs"
MAX_KERNELS_ORIGIN = "https://docs.modular.com"
MAX_KERNELS_PATH_PREFIX = "/max/api/kernels"


def _mojolang_href(site_path: str, *, hosted_on_mojolang: bool) -> str:
    assert site_path.startswith("/"), site_path
    if hosted_on_mojolang:
        return site_path
    return f"{MOJOLANG_ORIGIN}{site_path}"


def _max_kernels_href(site_path: str, *, hosted_on_mojolang: bool) -> str:
    assert site_path.startswith("/"), site_path
    if hosted_on_mojolang:
        return f"{MAX_KERNELS_ORIGIN}{site_path}"
    return site_path


def resolve_api_href(
    path: str | None,
    *,
    hosted_on_mojolang: bool = False,
) -> str:
    """Return the URL or root-relative path for a Mojo API doc cross-reference.

    Parameters:
        path: JSON ``path`` from ``mojo doc`` (root-relative, e.g.
            ``/std/builtin/Int`` or ``/kernels/linalg/foo/Bar``).
        hosted_on_mojolang: Set True for ``mojo_library`` targets whose Markdown
            ships on mojolang.org (stdlib, layout).

    Returns:
        Empty string when ``path`` is empty; otherwise the resolved href.
    """
    if path is None or path == "":
        return ""

    fragment = ""
    if "#" in path:
        base, fragment = path.split("#", 1)
        path = base
    if path == "":
        return f"#{fragment}" if fragment else ""

    if not path.startswith("/"):
        path = "/" + path

    # Stdlib type referenced from any package
    if path == "/std" or path.startswith("/std/"):
        href = _mojolang_href(
            f"{MOJOLANG_PATH_PREFIX}{path}",
            hosted_on_mojolang=hosted_on_mojolang,
        )
    # Layout type referenced from a kernel package (linalg uses `LayoutTensor`)
    elif path == "/kernels/layout" or path.startswith("/kernels/layout/"):
        layout_suffix = path[len("/kernels/layout") :]
        href = _mojolang_href(
            f"{MOJOLANG_PATH_PREFIX}/layout{layout_suffix}",
            hosted_on_mojolang=hosted_on_mojolang,
        )
    # Layout package referencing its own internal types
    elif path == "/layout" or path.startswith("/layout/"):
        href = _mojolang_href(
            f"{MOJOLANG_PATH_PREFIX}{path}",
            hosted_on_mojolang=hosted_on_mojolang,
        )
    # Cross-reference between non-layout kernel packages (linalg, nn, etc.)
    elif path.startswith("/kernels/"):
        href = _max_kernels_href(
            f"{MAX_KERNELS_PATH_PREFIX}{path[len('/kernels') :]}",
            hosted_on_mojolang=hosted_on_mojolang,
        )
    # Fallback for kernel packages that omit `docs_base_path`
    else:
        href = _max_kernels_href(
            f"{MAX_KERNELS_PATH_PREFIX}{path}",
            hosted_on_mojolang=hosted_on_mojolang,
        )

    if fragment:
        href = f"{href}#{fragment}"
    return href
