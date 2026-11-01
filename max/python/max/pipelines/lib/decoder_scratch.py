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
"""Workspace buffers a registered decoder's kernels write each step.

A decoder whose kernels need workspace sized for the largest batch, such as
a sparse attention indexer's score table, declares it with a
``scratch_types`` classmethod that returns one
:class:`~max.graph.BufferType` per buffer. The framework then:

- allocates each buffer once, at its declared size;
- adds them to the graph's inputs as the
  :attr:`~max.pipelines.lib.role_contracts.DecoderInputs.scratch` field,
  after the per-request state;
- passes the same buffers every step.

Unlike per-request state, nothing in them outlives a step, so a request
owns no part of them. Their shapes are static, so an engine that serves the
graph allocates them at the declared size too.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from max.driver import Buffer, Device
from max.graph import BufferType, StaticDim

__all__ = ["DecoderScratch", "build_decoder_scratch"]


@dataclass(frozen=True)
class DecoderScratch:
    """A decoder's workspace buffers and the graph inputs they fill."""

    input_types: tuple[BufferType, ...]
    """The graph inputs, in the order the decoder declares them."""

    buffers: tuple[Buffer, ...]
    """One buffer per :attr:`input_types` entry, allocated once."""


def build_decoder_scratch(
    decoder_cls: type[Any],
    config: Any,
    *,
    max_batch_size: int,
    devices: Sequence[Device],
) -> DecoderScratch | None:
    """Allocates the workspace a registered decoder declares, if any.

    Args:
        decoder_cls: The registered ``DECODER`` class.
        config: The finalized config the decoder is built from.
        max_batch_size: The most requests a step holds.
        devices: The devices the decoder runs on.

    Raises:
        TypeError: If ``scratch_types`` returns something other than buffer
            types.
        ValueError: If a buffer's shape isn't static.
    """
    declare = getattr(decoder_cls, "scratch_types", None)
    if declare is None:
        return None
    input_types = tuple(
        declare(config, max_batch_size=max_batch_size, devices=devices)
    )
    if not input_types:
        return None
    buffers = []
    for input_type in input_types:
        if not isinstance(input_type, BufferType):
            raise TypeError(
                f"{decoder_cls.__qualname__}.scratch_types() returned "
                f"{type(input_type).__qualname__}, not a BufferType"
            )
        if not all(isinstance(dim, StaticDim) for dim in input_type.shape):
            raise ValueError(
                f"{decoder_cls.__qualname__}.scratch_types() declared "
                f"{input_type}, but workspace is allocated once at a static "
                "size"
            )
        buffers.append(
            Buffer(
                dtype=input_type.dtype,
                shape=[int(dim) for dim in input_type.shape],
                device=input_type.device.to_device(),
            )
        )
    return DecoderScratch(input_types, tuple(buffers))
