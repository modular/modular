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

from render import render_constrained_deps


def test_render_no_deps() -> None:
    assert render_constrained_deps("foo", {}, {}) == ""


def test_render_normal_deps() -> None:
    output = render_constrained_deps("foo_deps", {"": ["bar", "baz"]}, {})
    assert (
        output
        == """\
    foo_deps = [
        ":bar",
        ":baz",
    ]

"""
    )


def test_render_non_gpu_conditional_deps() -> None:
    output = render_constrained_deps(
        "foo_deps",
        {
            "": ["foo"],
            "condition1": ["bar", "baz"],
            "condition2": ["qux"],
        },
        {},
    )
    assert (
        output
        == """\
    foo_deps = [
        ":foo",
    ] + select({
        "condition1": [
            ":bar",
            ":baz",
        ],
        "condition2": [
            ":qux",
        ],
        "//conditions:default": [],
    })

"""
    )


def test_render_gpu_conditional_deps() -> None:
    output = render_constrained_deps(
        "foo_deps",
        {
            "": ["foo"],
        },
        {
            "condition1": ["bar", "baz"],
            "condition2": ["qux"],
        },
    )
    assert (
        output
        == """\
    foo_deps = [
        ":foo",
    ] + select({
        "condition1": [
            ":bar",
            ":baz",
        ],
        "condition2": [
            ":qux",
        ],
    })

"""
    )


def test_render_gpu_conditional_only_deps() -> None:
    output = render_constrained_deps(
        "foo_deps",
        {},
        {
            "condition1": ["bar", "baz"],
            "condition2": ["qux"],
        },
    )
    assert (
        output
        == """\
    foo_deps = [
    ] + select({
        "condition1": [
            ":bar",
            ":baz",
        ],
        "condition2": [
            ":qux",
        ],
    })

"""
    )


def test_render_both_conditional_deps() -> None:
    output = render_constrained_deps(
        "foo_deps",
        {
            "": ["foo"],
            "condition1": ["bar", "baz"],
        },
        {
            "condition2": ["qux"],
        },
    )
    assert (
        output
        == """\
    foo_deps = [
        ":foo",
    ] + select({
        "condition1": [
            ":bar",
            ":baz",
        ],
        "//conditions:default": [],
    }) + select({
        "condition2": [
            ":qux",
        ],
    })

"""
    )
