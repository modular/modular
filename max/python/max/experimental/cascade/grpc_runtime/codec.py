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
"""Value-slot codec for the Cascade gRPC transport.

Args, kwargs, and return values travel as :py:class:`ValueSlot` envelopes
so they're portable across language implementations. Composite/structured
values are JSON-encoded; binary leaves (``bytes``, ``np.ndarray``) and
in-runtime :py:class:`Result` references get dedicated slots.

See cascade_runtime_v1.proto for reference.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
from max.experimental.cascade.core import Result, ResultIter, Runtime
from pydantic import BaseModel

from . import cascade_runtime_v1_pb2 as pb

# ---------------------------------------------------------------------------
# ValueSlot encode / decode
# ---------------------------------------------------------------------------


def _target_from_runtime(runtime: Any) -> str:
    """Extract the gRPC target address from a runtime carrying one.

    Accepts both :py:class:`GrpcRuntimeClient` and
    :py:class:`SubprocGrpcRuntimeClient` (the latter exposes ``.target``
    too). Lazy import breaks the circular dependency with ``runtime.py``.
    """
    from .runtime import GrpcRuntimeClient, SubprocGrpcRuntimeClient

    if not isinstance(runtime, (GrpcRuntimeClient, SubprocGrpcRuntimeClient)):
        raise TypeError(
            f"Expected a gRPC runtime for the gRPC codec but got"
            f" {type(runtime).__name__!r}"
        )
    return cast(str, runtime.target)


def encode_value_slot(value: Any) -> pb.ValueSlot:
    """Encode one top-level argument, kwarg, or return value into a ``ValueSlot``.

    :py:class:`Result` and :py:class:`ResultIter` values become wire references
    carrying the server ``target`` and ``result_id``; the backing runtime must
    be a gRPC runtime so the target address is available to encode.
    """
    if isinstance(value, ResultIter):
        return pb.ValueSlot(
            stream_ref=pb.ResultStreamReference(
                target=_target_from_runtime(value.runtime),
                result_id=value.result_id,
            )
        )
    if isinstance(value, Result):
        return pb.ValueSlot(
            result_ref=pb.ResultReference(
                target=_target_from_runtime(value.runtime),
                result_id=value.result_id,
            )
        )
    if isinstance(value, np.ndarray):
        return pb.ValueSlot(ndarray=_encode_ndarray(value))
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return pb.ValueSlot(bytes_value=bytes(value))
    return pb.ValueSlot(
        json_string=json.dumps(
            value, default=_json_default, separators=(",", ":")
        )
    )


def is_remote_ref(slot: pb.ValueSlot) -> bool:
    """Checks if the value slot is a reference to a result on another server."""
    kind = slot.WhichOneof("kind")
    return kind == "result_ref" or kind == "stream_ref"


async def decode_value_slot(
    slot: pb.ValueSlot, dial: Callable[[str], Runtime]
) -> Any:
    """Decode one ValueSlot back into a Python value.

    ``Result`` / ``ResultIter`` slots carry the gRPC ``target`` of the runtime
    that owns the referenced result, which may be a *different* server than the
    one we received the slot from. ``dial`` resolves that target by returning a
    lazy runtime client to that worker.
    """
    kind = slot.WhichOneof("kind")
    if kind == "json_string":
        return json.loads(slot.json_string, object_hook=_object_hook)
    if kind == "ndarray":
        return _decode_ndarray(slot.ndarray)
    if kind == "result_ref":
        return Result(
            runtime=dial(slot.result_ref.target),
            result_id=slot.result_ref.result_id,
        )
    if kind == "stream_ref":
        return ResultIter(
            runtime=dial(slot.stream_ref.target),
            result_id=slot.stream_ref.result_id,
        )
    if kind == "bytes_value":
        return bytes(slot.bytes_value)
    raise TypeError(f"Unsupported ValueSlot kind {kind!r}")


# ---------------------------------------------------------------------------
# args / kwargs convenience wrappers
# ---------------------------------------------------------------------------


def encode_args(args: Sequence[Any]) -> list[pb.ValueSlot]:
    """Encode positional arguments into a list of ValueSlots."""
    return [encode_value_slot(value) for value in args]


async def decode_args(
    args: Sequence[pb.ValueSlot], dial: Callable[[str], Runtime]
) -> list[Any]:
    """Decode a list of ValueSlots back into positional arguments."""
    return [await decode_value_slot(value, dial) for value in args]


def encode_kwargs(kwargs: Mapping[str, Any]) -> dict[str, pb.ValueSlot]:
    """Encode keyword arguments into a name -> ValueSlot mapping."""
    return {key: encode_value_slot(value) for key, value in kwargs.items()}


async def decode_kwargs(
    kwargs: Mapping[str, pb.ValueSlot], dial: Callable[[str], Runtime]
) -> dict[str, Any]:
    """Decode a name -> ValueSlot mapping back into keyword arguments."""
    return {
        key: await decode_value_slot(value, dial)
        for key, value in kwargs.items()
    }


# ---------------------------------------------------------------------------
# Error envelope encode / decode
# ---------------------------------------------------------------------------


def encode_error(exc: BaseException) -> pb.ErrorEnvelope:
    """Encode an exception into an :py:class:`ErrorEnvelope`.

    ``message`` and ``type_name`` are always populated and are the
    interoperable surface;
    """
    return pb.ErrorEnvelope(
        message=str(exc),
        type_name=type(exc).__name__,
    )


def decode_error(envelope: pb.ErrorEnvelope) -> BaseException:
    """Report errors as :py:class:`RuntimeError` carrying ``message`` and ``type_name``.

    Errors can come from other language environments, so the original type is not preserved.
    """
    return RuntimeError(f"{envelope.type_name}: {envelope.message}")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _encode_ndarray(value: np.ndarray) -> pb.NDArray:
    """Encode a numpy array into an ``NDArray`` protobuf message."""
    return pb.NDArray(
        dtype=value.dtype.str,
        shape=list(value.shape),
        data=value.tobytes(),
    )


def _decode_ndarray(value: pb.NDArray) -> np.ndarray:
    """Decode an ``NDArray`` protobuf message back into a numpy array."""
    dtype = np.dtype(value.dtype)
    return (
        np.frombuffer(value.data, dtype=dtype)
        .copy()
        .reshape(tuple(value.shape))
    )


# Reserved key used to tag JSON-encoded pydantic models so the decoder can
# rebuild the original class. Non-Python clients can ignore it (treating the
# rest of the dict as the model's structured payload) or strip it.
_PYCLASS_KEY = "__pyclass__"


def _json_default(value: Any) -> Any:
    """``json.dumps`` fallback hook for types stdlib JSON doesn't know.

    Pydantic models are tagged with ``__pyclass__`` so :py:func:`_object_hook`
    can rebuild the exact class on decode. ``json.dumps`` recurses into the
    returned dict, so nested models, lists of models, etc. are handled
    automatically without us walking the tree by hand.
    """
    if isinstance(value, BaseModel):
        cls = type(value)
        return {
            _PYCLASS_KEY: f"{cls.__module__}:{cls.__qualname__}",
            **value.model_dump(mode="json"),
        }
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _object_hook(obj: dict[str, Any]) -> Any:
    """``json.loads`` hook that rebuilds ``__pyclass__``-tagged dicts.

    Called bottom-up by ``json.loads``, so nested models are already
    rebuilt by the time their parent is processed.
    """
    spec = obj.pop(_PYCLASS_KEY, None)
    if spec is None:
        return obj
    return _import_pyclass(spec).model_validate(obj)


def _import_pyclass(spec: str) -> type[BaseModel]:
    """Import a ``module:qualname`` reference and validate it's a BaseModel.

    Using ``:`` as the separator (rather than a final ``.``) keeps nested
    classes (``Outer.Inner``) unambiguous. The :py:class:`BaseModel`
    subclass guard is the security boundary: without it the tag would let
    a sender pick any importable callable to invoke during decode.
    """
    module_name, _, attr_path = spec.partition(":")
    obj: Any = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise TypeError(
            f"{_PYCLASS_KEY} {spec!r} does not resolve to a BaseModel subclass"
        )
    return obj
