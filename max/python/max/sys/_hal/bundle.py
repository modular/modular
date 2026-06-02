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
"""``Bundle`` — Python projection of HAL ``RuntimeBundle``."""

from __future__ import annotations

from typing import Any


class Bundle:
    """A loaded device module produced by compiling a Mojo kernel.

    Mirrors HAL ``RuntimeBundle``. A bundle holds compiled GPU code
    (CUmodule / MTLLibrary / ...) loaded into device memory. One bundle
    can contain multiple compiled functions; resolve specific symbols
    via ``Context.load_function(bundle, name)``.

    Not constructed directly; obtain via a Mojo kernel module's
    ``compile()`` helper, which calls
    ``compile_to_python_bundle[type_of(kernel), kernel]`` internally.
    """

    _inner: Any

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        raise TypeError(
            "Bundle is not directly constructible; use a Mojo kernel "
            "module's compile() helper."
        )

    @classmethod
    def _wrap(cls, inner: object) -> Bundle:
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    @property
    def function_name(self) -> str:
        """Mangled symbol name of the compiled function.

        Pass this to ``Context.load_function(bundle, name)`` to resolve
        a launchable ``Function``.
        """
        return self._inner.get_function_name()

    def __repr__(self) -> str:
        return f"Bundle(function_name={self.function_name!r})"

    __str__ = __repr__
