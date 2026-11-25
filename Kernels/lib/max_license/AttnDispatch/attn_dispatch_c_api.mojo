# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""C entry to the attention decode dispatch, behind `AttnDispatch.h`.

The C structs are read and written by byte offset, not through Mojo structs,
so that a caller built against an older, shorter `MAttnDispatchDesc` stays
valid: a field past the caller's `struct_size` reads as unset.
`AttnDispatch.cpp` pins the offsets used here with static assertions.
"""

from max.gpu.host import DeviceContext
from nn.attention.gpu.attn_dispatch import (
    ATTN_DISPATCH_MAX_SCALARS,
    ATTN_DISPATCH_UNSET,
    AttnDispatchDesc,
    attn_dispatch_resolve,
)

comptime _BytePtr = Pointer[UInt8, MutUntrackedOrigin]

# Offsets into `MAttnDispatchDesc`.
comptime _DESC_STRUCT_SIZE = 0
comptime _DESC_KIND = 8
comptime _DESC_NUM_Q_HEADS = 16
comptime _DESC_NUM_KV_HEADS = 24
comptime _DESC_HEAD_DIM = 32
comptime _DESC_PAGE_SIZE = 40
comptime _DESC_WINDOW_SIZE = 48
comptime _DESC_SPARSE_INDICES_STRIDE = 56
comptime _DESC_KV_DTYPE = 64
comptime _DESC_KV_FORM = 68
comptime _DESC_FLAGS = 72

# Offsets into `MAttnDispatch`.
comptime _OUT_KEY = 0
comptime _OUT_N_KEY = 64
comptime _OUT_METADATA = 72
comptime _OUT_N_METADATA = 136


def _load_i64(base: _BytePtr, offset: Int) -> Int:
    return Int(base.unsafe_offset(offset).unsafe_bitcast[Int64]()[])


def _load_i32(base: _BytePtr, offset: Int) -> Int:
    return Int(base.unsafe_offset(offset).unsafe_bitcast[Int32]()[])


def _load_u32(base: _BytePtr, offset: Int) -> Int:
    return Int(base.unsafe_offset(offset).unsafe_bitcast[UInt32]()[])


def _i64_field(base: _BytePtr, struct_size: Int, offset: Int) -> Int:
    if offset + 8 > struct_size:
        return ATTN_DISPATCH_UNSET
    return _load_i64(base, offset)


def _i32_field(base: _BytePtr, struct_size: Int, offset: Int) -> Int:
    if offset + 4 > struct_size:
        return ATTN_DISPATCH_UNSET
    return _load_i32(base, offset)


def _read_desc(base: _BytePtr) raises -> AttnDispatchDesc:
    var struct_size = _load_u32(base, _DESC_STRUCT_SIZE)
    if struct_size < _DESC_KIND + 8:
        raise Error("MAttnDispatchDesc.struct_size does not cover `kind`")
    var kind_address = _load_i64(base, _DESC_KIND)
    if kind_address == 0:
        raise Error("MAttnDispatchDesc.kind is null")
    var kind = String(
        unsafe_from_utf8_ptr=Pointer[UInt8, ImmUntrackedOrigin](
            unsafe_from_address=kind_address
        )
    )
    var flags = 0
    if _DESC_FLAGS + 4 <= struct_size:
        flags = _load_u32(base, _DESC_FLAGS)
    return AttnDispatchDesc(
        kind=kind,
        num_q_heads=_i64_field(base, struct_size, _DESC_NUM_Q_HEADS),
        num_kv_heads=_i64_field(base, struct_size, _DESC_NUM_KV_HEADS),
        head_dim=_i64_field(base, struct_size, _DESC_HEAD_DIM),
        page_size=_i64_field(base, struct_size, _DESC_PAGE_SIZE),
        window_size=_i64_field(base, struct_size, _DESC_WINDOW_SIZE),
        sparse_indices_stride=_i64_field(
            base, struct_size, _DESC_SPARSE_INDICES_STRIDE
        ),
        kv_dtype=_i32_field(base, struct_size, _DESC_KV_DTYPE),
        kv_form=_i32_field(base, struct_size, _DESC_KV_FORM),
        flags=flags,
    )


def _store_scalars(
    result: _BytePtr, values: List[Int], values_offset: Int, count_offset: Int
):
    for i in range(ATTN_DISPATCH_MAX_SCALARS):
        var value = values[i] if i < len(values) else 0
        result.unsafe_offset(values_offset + 8 * i).unsafe_bitcast[
            Int64
        ]().unsafe_write(Int64(value))
    result.unsafe_offset(count_offset).unsafe_bitcast[Int32]().unsafe_write(
        Int32(len(values))
    )


def _store_error(message: String, buffer: _BytePtr, capacity: Int):
    if capacity < 1:
        return
    var bytes = message.as_bytes()
    var n = min(len(bytes), capacity - 1)
    for i in range(n):
        buffer.unsafe_offset(i).unsafe_write(bytes[i])
    buffer.unsafe_offset(n).unsafe_write(UInt8(0))


@export
def attn_dispatch_resolve_c(
    device_ctx: OpaquePointer[MutUntrackedOrigin],
    desc: _BytePtr,
    batch_size: Int64,
    q_max_seq_len: Int64,
    max_cache_valid_length: Int64,
    result: _BytePtr,
    error_buffer: _BytePtr,
    error_capacity: Int64,
) abi("C") -> Int32:
    """Resolves one decode dispatch for `M_attnDispatchResolve`.

    The caller passes non-null pointers. On failure the message is written to
    `error_buffer`, truncated to `error_capacity` bytes with its terminator.

    Returns:
        0 on success, 1 on failure.
    """
    try:
        var resolved = attn_dispatch_resolve(
            DeviceContext(device_ctx),
            _read_desc(desc),
            Int(batch_size),
            Int(q_max_seq_len),
            Int(max_cache_valid_length),
        )
        _store_scalars(result, resolved.key, _OUT_KEY, _OUT_N_KEY)
        _store_scalars(
            result, resolved.metadata, _OUT_METADATA, _OUT_N_METADATA
        )
        return 0
    except e:
        _store_error(String(e), error_buffer, Int(error_capacity))
        return 1
