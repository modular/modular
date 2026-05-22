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
"""``Driver`` — Python projection of HAL ``Driver``."""

from __future__ import annotations

from typing import Any

from .device import Device
from .mojo_module import Driver as _MojoDriver  # type: ignore[import-not-found]


class Driver:
    """A loaded HAL plugin.

    Construction reads the plugin spec from ``MODULAR_DRIVER_PLUGINS``.
    """

    _inner: Any

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        self._inner = _MojoDriver()

    @classmethod
    def _wrap(cls, inner: object) -> Driver:
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    @property
    def name(self) -> str:
        return self._inner.get_name()

    @property
    def device_count(self) -> int:
        return self._inner.device_count()

    def get_device(self, id: int) -> Device:
        return Device._wrap(self._inner.get_device(id))

    def __repr__(self) -> str:
        return f"Driver(name={self.name!r})"

    __str__ = __repr__
