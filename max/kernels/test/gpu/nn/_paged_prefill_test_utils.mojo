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
"""Shared scaffolding for paged-KV MLA prefill tests.

The three SM100 MLA prefill kernel variants (generic, blockscale,
per-token-scale) share the same paged-KV cache layout for K_rope
(``cache_depth = 576`` per token, with the rope window in the last 64
elements). This module provides the boilerplate to:

  - Initialize the paged blocks for K_rope with random data and
    zero-fill the tail past the per-batch ``num_keys``.
  - Build a uniform-batch-size lookup table.
  - Extract the contiguous K_rope slice from the paged storage so the
    naive MHA reference can compute against the same data the kernel
    sees.
  - Provide the runtime list of ``num_keys`` values each test binary
    iterates over (see ``num_keys_to_test``). ``page_size`` is a
    compile-time parameter (one binary per page_size); ``num_keys``
    is purely runtime, so we test the cartesian product of every
    page_size against every value in this list.

The Q/K/V (nope) ragged tensors and the kernel calls themselves stay in
the per-variant test files because they differ across the three
kernels. The naive-MHA reference call is also kept per-variant since
it needs per-variant scale/dtype handling.
"""

from std.math import align_up, ceildiv
from std.random import randn

from std.memory import alloc


# ===-----------------------------------------------------------------------===#
# Constants matching DeepSeek-V2/V3 MLA prefill shapes
# ===-----------------------------------------------------------------------===#

comptime CACHE_DEPTH = 576  # Full MLA cache depth (nope + rope)
comptime ROPE_DEPTH = 64  # Last 64 elements of cache hold rope
comptime NUM_LAYERS = 1  # Single layer per test
comptime KV_NUM_HEADS = 1  # MLA caches a single KV head

# Blockwise FP8 scale granularity: one scale value per 64-element block of
# the head_size axis. With CACHE_DEPTH=576, this yields HEAD_DIM_GRAN=9
# scale blocks per (token, head). The blockscale kernel reads the scale
# for the rope window (head_dim_idx=512..575) at block index 8.
comptime SCALE_BLOCK_SIZE = ROPE_DEPTH  # 64
comptime HEAD_DIM_GRAN = (
    CACHE_DEPTH + SCALE_BLOCK_SIZE - 1
) // SCALE_BLOCK_SIZE  # 9
comptime ROPE_SCALE_BLOCK_IDX = HEAD_DIM_GRAN - 1  # 8 — the block holding rope


# ===-----------------------------------------------------------------------===#
# Paged-KV block layout helpers
# ===-----------------------------------------------------------------------===#


@always_inline
def paged_block_elems(
    total_pages: Int, page_size: Int, head_size: Int = CACHE_DEPTH
) -> Int:
    """Number of scalar elements in the paged block array.

    Block shape (matching ``test_mla_decode_paged_variable.mojo``):
    ``[total_pages, kv_dim2=1, NUM_LAYERS=1, page_size, KV_NUM_HEADS=1,
    head_size]``.
    """
    return total_pages * NUM_LAYERS * page_size * KV_NUM_HEADS * head_size


@always_inline
def page_stride(page_size: Int, head_size: Int = CACHE_DEPTH) -> Int:
    """Per-page element stride in the paged block array."""
    return NUM_LAYERS * page_size * KV_NUM_HEADS * head_size


@always_inline
def token_stride(head_size: Int = CACHE_DEPTH) -> Int:
    """Per-token element stride within a page."""
    return KV_NUM_HEADS * head_size


@always_inline
def lut_max_pages_per_batch(num_keys: Int, page_size: Int) -> Int:
    """LUT row stride (pages-per-batch padded to multiple of 8).

    The 8-page padding matches the SIMD chunk size of the
    ``PagedKVCache.populate`` path.
    """
    return align_up(ceildiv(num_keys, page_size), 8)


# ===-----------------------------------------------------------------------===#
# Random-init + tail-zero-fill for uniform-batch paged blocks
# ===-----------------------------------------------------------------------===#


def fill_paged_blocks_uniform[
    kv_type: DType,
](
    blocks_host: UnsafePointer[Scalar[kv_type], MutAnyOrigin],
    batch_size: Int,
    num_keys: Int,
    page_size: Int,
    head_size: Int = CACHE_DEPTH,
    standard_deviation: Float64 = 0.5,
):
    """Fill ``blocks_host`` with random data (bf16 ~ N(0, σ²)) cast
    to ``kv_type``, then zero out tail slots past ``num_keys`` in the
    last page of each batch.

    The randn-then-cast roundtrip keeps the distribution well-defined
    across kv_types (including FP8). Zero-filling the tail makes
    accidental OOB reads contribute negligibly to softmax.

    Use ``standard_deviation=1.0`` (or larger) when ``kv_type`` is an
    FP8 format: smaller stddevs concentrate values near zero, where
    e4m3fn/e5m2 lose precision in the subnormal range. ``0.5`` is fine
    for bf16/half tests and matches the original generic-paged config.

    Each batch is assumed to occupy contiguous pages
    ``[b * num_pages_per_batch, (b+1) * num_pages_per_batch)``.
    """
    var num_pages_per_batch = ceildiv(num_keys, page_size)
    var total_pages = batch_size * num_pages_per_batch
    var block_elems = paged_block_elems(total_pages, page_size, head_size)

    # Random bf16 → cast to kv_type.
    var blocks_bf16 = alloc[BFloat16](block_elems)
    randn[DType.bfloat16](
        blocks_bf16,
        block_elems,
        mean=0.0,
        standard_deviation=standard_deviation,
    )
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()
    blocks_bf16.free()

    # Tail zero-fill in last page of each batch.
    var pstride = page_stride(page_size, head_size)
    var tstride = token_stride(head_size)
    for b in range(batch_size):
        var num_pages_b = num_pages_per_batch
        var valid_in_last = num_keys - (num_pages_b - 1) * page_size
        if valid_in_last == page_size:
            continue
        var last_page = b * num_pages_per_batch + (num_pages_b - 1)
        var base = last_page * pstride + valid_in_last * tstride
        var zero_count = (page_size - valid_in_last) * tstride
        for z in range(zero_count):
            blocks_host[base + z] = 0


# ===-----------------------------------------------------------------------===#
# Lookup-table population for uniform-batch paged blocks
# ===-----------------------------------------------------------------------===#


def fill_uniform_lookup_table(
    lookup_table_host: UnsafePointer[UInt32, MutAnyOrigin],
    batch_size: Int,
    num_keys: Int,
    page_size: Int,
    max_pages_per_batch: Int,
):
    """Populate the lookup table with each batch's pages contiguously.

    For batch ``b``, page ``p`` (in the batch's local page numbering),
    the LUT entry is ``b * num_pages_per_batch + p``. Padding entries
    (between ``num_pages_per_batch`` and ``max_pages_per_batch``) are
    left at zero — the kernel must not read them.
    """
    var num_pages_per_batch = ceildiv(num_keys, page_size)
    for i in range(batch_size * max_pages_per_batch):
        lookup_table_host[i] = UInt32(0)
    for b in range(batch_size):
        for p in range(num_pages_per_batch):
            lookup_table_host[b * max_pages_per_batch + p] = UInt32(
                b * num_pages_per_batch + p
            )


# ===-----------------------------------------------------------------------===#
# Reference K_rope extraction
# ===-----------------------------------------------------------------------===#


def extract_k_rope_for_batch[
    kv_type: DType,
](
    blocks_host: UnsafePointer[Scalar[kv_type], MutAnyOrigin],
    out_host: UnsafePointer[Scalar[kv_type], MutAnyOrigin],
    batch_idx: Int,
    num_keys: Int,
    page_size: Int,
    head_size: Int = CACHE_DEPTH,
):
    """Copy the rope window (last ``ROPE_DEPTH`` elements of every
    cache token) for batch ``batch_idx`` into ``out_host``.

    ``out_host`` must point to a buffer of at least ``num_keys *
    ROPE_DEPTH`` ``Scalar[kv_type]`` elements. The rope tokens are
    laid out contiguously in token order (matching the
    ``[num_keys, ROPE_DEPTH]`` shape the reference K_ref tile expects).

    Each batch is assumed to occupy contiguous pages
    ``[b * num_pages_per_batch, (b+1) * num_pages_per_batch)``.
    """
    var num_pages_per_batch = ceildiv(num_keys, page_size)
    var page_base = batch_idx * num_pages_per_batch
    var rope_offset_in_token = head_size - ROPE_DEPTH

    var pstride = page_stride(page_size, head_size)
    var tstride = token_stride(head_size)

    for tok in range(num_keys):
        var page_idx = tok // page_size
        var tok_in_page = tok % page_size
        var physical_page = page_base + page_idx

        var src_offset = (
            physical_page * pstride
            + tok_in_page * tstride
            + rope_offset_in_token
        )
        var dst_offset = tok * ROPE_DEPTH
        for d in range(ROPE_DEPTH):
            out_host[dst_offset + d] = blocks_host[src_offset + d]


# ===-----------------------------------------------------------------------===#
# Blockwise FP8 scales (paged): layout, fill, and reference dequant helpers
# ===-----------------------------------------------------------------------===#
#
# These helpers support tests of MLA prefill kernels that read the FP8
# K_rope cache via a ``PagedKVCacheCollection`` configured with
# ``scale_dtype_=DType.float32, quantization_granularity_=SCALE_BLOCK_SIZE``.
# The scales array is 6D, mirroring the FP8 blocks array but with the
# last axis replaced by ``HEAD_DIM_GRAN`` block-scales:
#
#   shape = [total_pages, kv_dim2=1, NUM_LAYERS=1, page_size,
#            KV_NUM_HEADS=1, HEAD_DIM_GRAN=9]
#
# The blockscale kernel only reads the rope-window scale (block index
# ``ROPE_SCALE_BLOCK_IDX = 8``) but the scales tensor must cover all 9
# blocks because the underlying ``KVCacheT`` operand stores them
# contiguously per-token.


@always_inline
def paged_scale_block_elems(
    total_pages: Int, page_size: Int, head_dim_gran: Int = HEAD_DIM_GRAN
) -> Int:
    """Number of FP32 scale elements in the paged scales array.

    Same shape as ``paged_block_elems`` but with the last axis replaced
    by ``head_dim_gran`` block scales rather than ``head_size``.
    """
    return total_pages * NUM_LAYERS * page_size * KV_NUM_HEADS * head_dim_gran


@always_inline
def scale_page_stride(
    page_size: Int, head_dim_gran: Int = HEAD_DIM_GRAN
) -> Int:
    """Per-page element stride in the paged scales array."""
    return NUM_LAYERS * page_size * KV_NUM_HEADS * head_dim_gran


@always_inline
def scale_token_stride(head_dim_gran: Int = HEAD_DIM_GRAN) -> Int:
    """Per-token element stride within a page in the paged scales array."""
    return KV_NUM_HEADS * head_dim_gran


@always_inline
def _palette_scale(idx: Int) -> Float32:
    """Pick a non-uniform scale from a tight 8-entry palette centered
    around 1.0.

    A tighter range (compared to the decode test's 256x range) keeps
    FP8 quantization noise within tolerance after dequantization, since
    the prefill test compares against an FP8→BF16 dequantized
    reference: ``out = fp8_val * scale``. Wild scales amplify FP8's
    ~5% mantissa error past the 2e-2 tolerance.
    """
    if idx % 8 == 0:
        return 0.5
    if idx % 8 == 1:
        return 0.625
    if idx % 8 == 2:
        return 0.75
    if idx % 8 == 3:
        return 0.875
    if idx % 8 == 4:
        return 1.0
    if idx % 8 == 5:
        return 1.125
    if idx % 8 == 6:
        return 1.25
    return 1.5


def fill_paged_block_scales(
    scales_host: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    num_keys: Int,
    page_size: Int,
    head_dim_gran: Int = HEAD_DIM_GRAN,
):
    """Fill ``scales_host`` with non-uniform per-(token, block) FP32
    scales drawn from a small palette.

    Using a small palette of order-of-magnitude-1 values (rather than
    ``randn``) keeps the dequantized FP8→BF16 result in a numerically
    well-behaved range for the reference comparison: the kernel does
    ``out = fp8_val * scale`` so wild scales would amplify FP8
    quantization noise past the test's tolerance.

    Tail slots past ``num_keys`` in the last page of each batch are
    filled with neutral 1.0 (matching the kernel's CVT consumer
    behavior, which uses scale=1 for OOB rows — see
    ``cvt_block_fp8_to_bf16_with_scale`` in ``mla_prefill_utils.mojo``).
    """
    var num_pages_per_batch = (num_keys + page_size - 1) // page_size
    var pstride = scale_page_stride(page_size, head_dim_gran)
    var tstride = scale_token_stride(head_dim_gran)

    for b in range(batch_size):
        var page_base = b * num_pages_per_batch
        for pg in range(num_pages_per_batch):
            var physical_page = page_base + pg
            for tok_in_page in range(page_size):
                var tok_global = pg * page_size + tok_in_page
                for blk in range(head_dim_gran):
                    var off = (
                        physical_page * pstride + tok_in_page * tstride + blk
                    )
                    if tok_global < num_keys:
                        # Coprime stride 7 vs palette length 8 so all
                        # entries are exercised even for small token
                        # counts. Vary by both token and block index.
                        scales_host[off] = _palette_scale(tok_global * 7 + blk)
                    else:
                        scales_host[off] = 1.0


def extract_dequantized_k_rope_for_batch[
    fp8_type: DType,
    out_type: DType,
](
    blocks_host: UnsafePointer[Scalar[fp8_type], MutAnyOrigin],
    scales_host: UnsafePointer[Float32, MutAnyOrigin],
    out_host: UnsafePointer[Scalar[out_type], MutAnyOrigin],
    batch_idx: Int,
    num_keys: Int,
    page_size: Int,
    head_size: Int = CACHE_DEPTH,
    head_dim_gran: Int = HEAD_DIM_GRAN,
):
    """Extract the rope window for ``batch_idx``, dequantizing per token
    using the matching scale at block index ``ROPE_SCALE_BLOCK_IDX``.

    Equivalent to ``extract_k_rope_for_batch`` followed by
    ``out[t, d] = fp8_val[t, d].cast[float32]() * scale[t,
    ROPE_SCALE_BLOCK_IDX]``, with the result cast to ``out_type``.
    Mirrors the dequantization the blockscale kernel applies via
    ``cvt_block_fp8_to_bf16_with_scale`` (which reads the scale at
    ``head_dim_idx=cache_depth - rope_depth``, i.e. block ``8`` for
    ``cache_depth=576, granularity=64``).

    ``out_host`` must point to a buffer of at least
    ``num_keys * ROPE_DEPTH`` ``Scalar[out_type]`` elements.
    """
    var num_pages_per_batch = (num_keys + page_size - 1) // page_size
    var page_base = batch_idx * num_pages_per_batch
    var rope_offset_in_token = head_size - ROPE_DEPTH

    var pstride = page_stride(page_size, head_size)
    var tstride = token_stride(head_size)
    var spstride = scale_page_stride(page_size, head_dim_gran)
    var ststride = scale_token_stride(head_dim_gran)

    for tok in range(num_keys):
        var page_idx = tok // page_size
        var tok_in_page = tok % page_size
        var physical_page = page_base + page_idx

        var src_offset = (
            physical_page * pstride
            + tok_in_page * tstride
            + rope_offset_in_token
        )
        var scale_offset = (
            physical_page * spstride
            + tok_in_page * ststride
            + ROPE_SCALE_BLOCK_IDX
        )
        var scale_val = scales_host[scale_offset]

        var dst_offset = tok * ROPE_DEPTH
        for d in range(ROPE_DEPTH):
            var fp8_val = blocks_host[src_offset + d].cast[DType.float32]()
            out_host[dst_offset + d] = (fp8_val * scale_val).cast[out_type]()


# ===-----------------------------------------------------------------------===#
# (page_size × num_keys) cartesian-product test configuration
# ===-----------------------------------------------------------------------===#


def num_keys_to_test() -> List[Int]:
    """Return the list of ``num_keys`` values to test against any
    compile-time ``page_size``.

    Each paged-prefill test binary is built once per ``page_size``
    (a compile-time parameter that flows into
    ``PagedKVCacheCollection[..., page_size]`` and the kernel's
    sub-tile TMA descriptors). Inside ``main`` it iterates over the
    values returned here, so the BUILD rule fans out exactly
    ``len(_MLA_PREFILL_PAGED_PAGE_SIZES)`` binaries per variant rather
    than the cartesian product (compile time dominates over runtime
    cost in these tests).

    The list is the union of every ``num_keys`` previously tested
    across the (page_size, num_keys) configs, so the cartesian product
    of {page_sizes} × {this list} is a strict superset of the
    pre-refactor coverage. It exercises:

      - Aligned baselines: ``num_keys == page_size`` (e.g. 64, 128,
        256), and ``num_keys`` that fills whole BN-sized tiles.
      - Partial-last-tile cases that previously fired the kernel-side
        ``debug_assert`` pre-fix (e.g. (16, 17), (32, 96), (64, 64),
        (16, 100)).
      - Mixed alignment: e.g. (32, 17), (128, 100) — ``num_keys`` not
        a multiple of ``page_size``.
    """
    return [17, 64, 96, 100, 128, 256]
