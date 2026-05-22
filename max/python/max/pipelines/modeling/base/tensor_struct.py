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
"""Structured tensor container base class with compile-time field enforcement."""

from __future__ import annotations

import types
from dataclasses import fields, replace
from typing import Any, Union, get_args, get_origin, get_type_hints

from max._core.driver import Device
from max.driver import Buffer
from max.experimental.tensor import Tensor
from typing_extensions import Self


class TensorStruct:
    """Base for structured tensor containers used as pipeline inputs/outputs.

    Enforces at class-definition time (via ``__init_subclass__``) that
    every field annotation is ``Tensor``, ``Buffer``, or
    ``Optional[Tensor | Buffer]``.  Scalars, numpy arrays, ints,
    strings, and other non-tensor types are rejected with a
    ``TypeError`` when the subclass is defined (i.e. at import time).

    No runtime validation overhead -- the frozen dataclass ``__init__``
    assigns fields directly with no extra checks on the hot path.

    Subclasses should be decorated with ``@dataclass(frozen=True)``::

        @dataclass(frozen=True)
        class MyInputs(TensorStruct):
            tokens: Tensor
            latents: Tensor
            image: Tensor | None = None  # optional feature
    """

    _ALLOWED_TYPES = frozenset({Tensor, Buffer})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        try:
            hints = get_type_hints(cls)
        except Exception:
            # get_type_hints can fail for forward references that are not
            # yet resolvable.  Skip validation in that case -- the class
            # will be validated when it is actually imported normally.
            return

        for name, hint in hints.items():
            if name.startswith("_"):
                continue
            origin = get_origin(hint)
            if origin in (Union, types.UnionType):
                non_none = [a for a in get_args(hint) if a is not type(None)]
                if not all(a in cls._ALLOWED_TYPES for a in non_none):
                    raise TypeError(
                        f"Field {name!r} on {cls.__qualname__}: "
                        f"TensorStruct fields must be Tensor, Buffer, "
                        f"or Optional[Tensor | Buffer], got {hint}"
                    )
            elif hint not in cls._ALLOWED_TYPES:
                raise TypeError(
                    f"Field {name!r} on {cls.__qualname__}: "
                    f"TensorStruct fields must be Tensor, Buffer, "
                    f"or Optional[Tensor | Buffer], got {hint}"
                )

    def to(self, device: Device) -> Self:
        """Transfer all present tensors to *device*, returning a new instance.

        ``None``-valued optional fields are left as ``None``.
        """
        updates: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            if isinstance(val, (Tensor, Buffer)):
                updates[f.name] = val.to(device)
        return replace(self, **updates)  # type: ignore[type-var]
