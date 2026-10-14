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
"""A name-transparent :class:`Module`."""

from __future__ import annotations

from typing_extensions import ParamSpec, TypeVar

from .module import Module

_P = ParamSpec("_P")
_R = TypeVar("_R")


class TransparentModule(Module[_P, _R]):
    """A :class:`Module` that drops its own name from its descendants' paths.

    Its children and parameters appear attached directly to its parent, with
    this module's own attribute name dropped from their qualified paths. Its
    rebased paths must not collide with a sibling's (enforced by
    :attr:`~max.experimental.nn.Module.parameters`). Set
    :attr:`name_transparent` to ``False`` on an instance to keep the name and
    behave as an ordinary opaque :class:`Module`.
    """

    #: Settable per instance. A module that holds either separate weights or
    #: one fused weight sets this to ``False`` in the fused case.
    name_transparent: bool = True
