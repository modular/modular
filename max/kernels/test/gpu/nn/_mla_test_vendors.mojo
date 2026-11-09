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
"""Per-vendor pieces of the MLA kernel tests.

A test written against `MLATestVendor` runs on whichever vendor its binary
names, and reaches only that vendor's MLA kernels.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from nn.attention.gpu.mla import AMDMLA, LegacyNVMLA, MLAKernels, SM100MLA
from nn.attention.gpu.mla_graph import (
    AMDMLAGraph,
    LegacyNVMLAGraph,
    MLAGraphKernels,
    SM100MLAGraph,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)


trait MLATestVendor:
    """The vendor-specific pieces an MLA test runs with."""

    comptime Kernels: MLAKernels
    """The vendor's MLA decode and prefill kernels."""

    comptime GraphKernels: MLAGraphKernels
    """The vendor's MLA graph-branch kernels."""

    @staticmethod
    def scalar_args[
        num_heads: Int,
        *,
        is_fp8_kv: Bool = False,
        _is_cache_length_accurate: Bool = False,
    ](
        batch_size: Int,
        max_cache_len: Int,
        q_max_seq_len: Int,
        ctx: DeviceContext,
    ) raises -> DeviceBuffer[DType.int64]:
        """Returns the decode dispatch-scalar buffer for this vendor.

        Parameters:
            num_heads: Number of query heads.
            is_fp8_kv: Whether the KV cache is FP8.
            _is_cache_length_accurate: Whether the cache length already
                counts the new tokens.

        Args:
            batch_size: Sequences in the batch.
            max_cache_len: Longest cache in the batch.
            q_max_seq_len: Longest query in the batch.
            ctx: Device context.

        Returns:
            A 3-element `[batch, q_max_seq_len, num_partitions]` buffer.
        """
        ...


struct AMDVendor(MLATestVendor):
    """AMD kernels; the AMD decode sizes its own split and ignores the
    dispatch scalars."""

    comptime Kernels = AMDMLA
    comptime GraphKernels = AMDMLAGraph

    @staticmethod
    def scalar_args[
        num_heads: Int,
        *,
        is_fp8_kv: Bool = False,
        _is_cache_length_accurate: Bool = False,
    ](
        batch_size: Int,
        max_cache_len: Int,
        q_max_seq_len: Int,
        ctx: DeviceContext,
    ) raises -> DeviceBuffer[DType.int64]:
        """Returns an unread scalar buffer."""
        return ctx.enqueue_create_buffer[DType.int64](3)


struct SM100Vendor(MLATestVendor):
    """SM100 kernels, with the dispatch scalars their capturable path reads."""

    comptime Kernels = SM100MLA
    comptime GraphKernels = SM100MLAGraph

    @staticmethod
    def scalar_args[
        num_heads: Int,
        *,
        is_fp8_kv: Bool = False,
        _is_cache_length_accurate: Bool = False,
    ](
        batch_size: Int,
        max_cache_len: Int,
        q_max_seq_len: Int,
        ctx: DeviceContext,
    ) raises -> DeviceBuffer[DType.int64]:
        """Precomputes the SM100 decode dispatch scalars."""
        var args = MLADispatchScalarArgs[
            num_heads=num_heads,
            _is_cache_length_accurate=_is_cache_length_accurate,
            is_fp8_kv=is_fp8_kv,
        ](batch_size, max_cache_len, q_max_seq_len, ctx)
        return args.gpu_buf


struct LegacyNVVendor(MLATestVendor):
    """Pre-SM100 NVIDIA kernels, which ignore the dispatch scalars."""

    comptime Kernels = LegacyNVMLA
    comptime GraphKernels = LegacyNVMLAGraph

    @staticmethod
    def scalar_args[
        num_heads: Int,
        *,
        is_fp8_kv: Bool = False,
        _is_cache_length_accurate: Bool = False,
    ](
        batch_size: Int,
        max_cache_len: Int,
        q_max_seq_len: Int,
        ctx: DeviceContext,
    ) raises -> DeviceBuffer[DType.int64]:
        """Returns an unread scalar buffer."""
        return ctx.enqueue_create_buffer[DType.int64](3)
