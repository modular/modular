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
"""Registers the ring-state short convolution graph ops; see
`nn.short_conv_ring`."""

import extensibility
from extensibility import InputTensor, OutputTensor
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from max.gpu.host import DeviceContext

from nn.short_conv_ring import short_conv_ring_commit, short_conv_ring_fwd


@extensibility.register("mo.short_conv_ring_fwd")
struct Struct_short_conv_ring_fwd:
    """`x + conv(x)` over a ragged batch, pre-chunk taps from the ring."""

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        ring_dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        # Only read here; mutable so the graph orders the commit after it.
        ring: MutableInputTensor[dtype=ring_dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        positions: InputTensor[dtype=.uint32, rank=1, ...],
        conv_row: InputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        short_conv_ring_fwd[target=target](
            x.to_tile_tensor[.int64](),
            weight.to_tile_tensor[.int64](),
            ring.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            positions.to_tile_tensor[.int64](),
            conv_row.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.short_conv_ring_commit")
struct Struct_short_conv_ring_commit:
    """Writes each sequence's last inputs into its conv ring slot."""

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        ring_dtype: DType,
        //,
        target: StaticString,
    ](
        ring: MutableInputTensor[dtype=ring_dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        positions: InputTensor[dtype=.uint32, rank=1, ...],
        conv_row: InputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        short_conv_ring_commit[target=target](
            x.to_tile_tensor[.int64](),
            ring.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            positions.to_tile_tensor[.int64](),
            conv_row.to_tile_tensor[.int64](),
            context,
        )
