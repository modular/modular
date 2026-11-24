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

"""Warns code outside MAX that it is using the deprecated ``max.nn.legacy``.

``max.nn`` imports ``max.nn.legacy`` itself, so the legacy package always
first loads from inside MAX and a warning raised while it loads would never
reach the user. Instead, the legacy modules check every attribute access, as
in ``from max.nn.legacy import Linear``, and warn the first caller outside the
``max`` package once per process.
"""

from __future__ import annotations

import sys
import types
import warnings

MESSAGE = (
    "max.nn.legacy is deprecated and will be removed in MAX 27.1. Use max.nn"
    " instead."
)

_warned = False


def _caller_outside_max(depth: int) -> types.FrameType | None:
    """Returns the frame that triggered the access if it is outside MAX."""
    # Frames: 0 is this helper, 1 is warn_if_outside_max, 2 is the hook
    # that called it, and depth frames above that is the trigger.
    frame: types.FrameType | None = sys._getframe(depth + 2)
    # The import system resolves `from max.nn.legacy import x` in its own
    # frames; the statement that asked for it is the next frame out.
    while frame is not None and frame.f_globals.get("__name__", "").startswith(
        ("importlib", "_frozen_importlib")
    ):
        frame = frame.f_back
    if frame is None:
        return None
    name = frame.f_globals.get("__name__", "")
    if name == "max" or name.startswith("max."):
        return None
    return frame


def warn_if_outside_max(depth: int = 1) -> None:
    """Warns once if the code ``depth`` frames above the caller is outside MAX."""
    global _warned
    if _warned:
        return
    frame = _caller_outside_max(depth)
    if frame is None:
        return
    _warned = True
    # warn_explicit attributes the warning to the caller's line on every
    # supported Python; stacklevel treats importlib frames differently
    # before and after 3.12. module_globals is left out: for a script's
    # __main__ it makes warn_explicit raise ImportError.
    warnings.warn_explicit(
        MESSAGE,
        DeprecationWarning,
        filename=frame.f_code.co_filename,
        lineno=frame.f_lineno,
        module=frame.f_globals.get("__name__"),
        registry=frame.f_globals.setdefault("__warningregistry__", {}),
    )


class _LegacyModule(types.ModuleType):
    def __getattribute__(self, name: str) -> object:
        if not _warned and not name.startswith("__"):
            warn_if_outside_max()
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: object) -> None:
        # The import system binds each newly loaded submodule on its parent,
        # so this hooks submodules that load after install() ran.
        if isinstance(value, types.ModuleType) and value.__name__.startswith(
            f"{self.__name__}."
        ):
            _hook(value.__name__)
        super().__setattr__(name, value)


def _hook(package: str) -> None:
    """Hooks ``package`` and its already-loaded submodules.

    A package's own ``__init__`` imports its children before the package is
    bound on its parent, so they load before the package itself is hooked.
    """
    for name, module in list(sys.modules.items()):
        if (name == package or name.startswith(f"{package}.")) and type(
            module
        ) is types.ModuleType:
            module.__class__ = _LegacyModule


def install() -> None:
    """Hooks every loaded ``max.nn.legacy`` module."""
    _hook("max.nn.legacy")
