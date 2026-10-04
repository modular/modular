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
"""GPU test for `kv_cache_gather_rows_ragged`.

One paged leaf with a single head per slot, read by the GPU path and by the
CPU path over host views of the same buffers. Both must equal the block
buffer indexed through the lookup table by hand, bit for bit. Covers a leaf
paged by entry (`slots_per_page < page_size`), bf16 and float32, the key and
the value side of a K/V leaf, and ragged rows including a request with none.
"""

from std.math import ceildiv
from std.random import randn_float64, random_ui64, seed

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.kv_cache_gather import kv_cache_gather_rows_ragged

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable

comptime NUM_LAYERS = 3
comptime LAYER = 1


def run_case[
    dtype: DType,
    head_dim: Int,
    slots_per_page: Int,
    is_mla: Bool,
](
    ctx: DeviceContext,
    row_counts: List[Int],
    live_slots: List[Int],
    num_slots: Int,
) raises:
    seed(0x5EED)
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=head_dim, is_mla=is_mla
    )
    comptime kv_dim = 1 if is_mla else 2
    var batch_size = len(row_counts)

    # The lookup table is built from slot counts, so a leaf paged by entry
    # gets one page per `slots_per_page` entries, as the model's does.
    var zeros = List[Int](length=batch_size, fill=0)
    var lengths = CacheLengthsTable.build(live_slots, zeros, ctx)
    var num_pages = (
        ceildiv(lengths.max_full_context_length, slots_per_page) * batch_size
        + 1
    )
    var lut = PagedLookupTable[slots_per_page].build(
        live_slots, zeros, lengths.max_full_context_length, num_pages, ctx
    )
    var blocks = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_pages),
            Idx[kv_dim],
            Int64(NUM_LAYERS),
            Idx[slots_per_page],
            Idx[1],
            Idx[head_dim],
        ),
        ctx,
    )
    var blocks_host = blocks.host_tensor()
    for i in range(blocks_host.num_elements()):
        blocks_host.unsafe_ptr().unsafe_store(i, Scalar[dtype](randn_float64()))
    blocks.to_device()

    var offsets = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size + 1)), ctx
    )
    var offsets_host = offsets.host_tensor()
    var num_rows = 0
    for b in range(batch_size):
        offsets_host[b] = UInt32(num_rows)
        num_rows += row_counts[b]
    offsets_host[batch_size] = UInt32(num_rows)
    offsets.to_device()

    var slots = HostDeviceTileTensor[.int32](
        row_major(Int64(num_rows), Int64(num_slots)), ctx
    )
    var slots_host = slots.host_tensor()
    var row_batch = List[Int]()
    for b in range(batch_size):
        for _ in range(row_counts[b]):
            row_batch.append(b)
    for r in range(num_rows):
        for j in range(num_slots):
            slots_host[r, j] = Int32(
                random_ui64(0, UInt64(live_slots[row_batch[r]] - 1))
            )
    slots.to_device()

    var out_layout = row_major(Int64(num_rows), Int64(num_slots), Idx[head_dim])
    var gpu_out = HostDeviceTileTensor[dtype](out_layout, ctx)
    _ = gpu_out.host_tensor().fill(0)
    gpu_out.to_device()
    # The CPU path writes host memory only.
    var cpu_out = HostDeviceTileTensor[dtype](out_layout)
    var cpu_host = cpu_out.host_tensor().fill(0)

    comptime Collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        slots_per_page,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var device_collection = Collection(
        blocks.device_tensor().as_unsafe_any_origin(),
        lengths.cache_lengths.device_tile_tensor(),
        lut.device_tile_tensor(),
        UInt32(lengths.max_seq_length_batch),
        UInt32(lengths.max_full_context_length),
    )
    var host_collection = Collection(
        blocks.host_tensor().as_unsafe_any_origin(),
        lengths.cache_lengths.host_tile_tensor(),
        lut.host_tile_tensor(),
        UInt32(lengths.max_seq_length_batch),
        UInt32(lengths.max_full_context_length),
    )
    var lut_host = lut.host_tile_tensor()
    var cpu_ctx = DeviceContext(api="cpu")

    comptime for kv in range(kv_dim):
        comptime if kv == 0:
            kv_cache_gather_rows_ragged[target="gpu"](
                gpu_out.device_tensor(),
                slots.device_tensor(),
                offsets.device_tensor(),
                device_collection.get_key_cache(LAYER),
                ctx,
            )
            kv_cache_gather_rows_ragged[target="cpu"](
                cpu_host,
                slots_host,
                offsets_host,
                host_collection.get_key_cache(LAYER),
                cpu_ctx,
            )
        else:
            kv_cache_gather_rows_ragged[target="gpu"](
                gpu_out.device_tensor(),
                slots.device_tensor(),
                offsets.device_tensor(),
                device_collection.get_value_cache(LAYER),
                ctx,
            )
            kv_cache_gather_rows_ragged[target="cpu"](
                cpu_host,
                slots_host,
                offsets_host,
                host_collection.get_value_cache(LAYER),
                cpu_ctx,
            )
        ctx.synchronize()
        gpu_out.to_host()

        var gpu_host = gpu_out.host_tensor()
        for r in range(num_rows):
            var b = row_batch[r]
            for j in range(num_slots):
                var slot = Int(slots_host[r, j])
                var page = Int(lut_host[b, slot // slots_per_page])
                var in_page = slot % slots_per_page
                for d in range(head_dim):
                    var want = blocks_host[page, kv, LAYER, in_page, 0, d]
                    if gpu_host[r, j, d] != want or cpu_host[r, j, d] != want:
                        raise Error(
                            t"mismatch at kv {kv} row {r} slot {j} dim {d}"
                        )

    # The collections hold untracked views of `blocks`, so pin it past the
    # launches above.
    _ = blocks^


def main() raises:
    with DeviceContext() as ctx:
        # Compressed zone: 32 entries per page, addressed by entry.
        run_case[DType.bfloat16, 128, 32, True](ctx, [5, 3], [50, 10], 48)
        # Sliding-window latent: decode-like, one row per request.
        run_case[DType.bfloat16, 512, 128, True](
            ctx, [1, 1, 1], [300, 5, 130], 16
        )
        # Compressor open state: float32 K and V, a request with no rows.
        run_case[DType.float32, 256, 128, False](
            ctx, [4, 0, 2], [64, 10, 257], 7
        )
    print("OK")
