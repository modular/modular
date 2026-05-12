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

from std.gpu.host import DeviceContext
from std.testing import assert_equal

from layout import Coord, Idx, TileTensor, row_major

from linalg.matmul.gpu.amd import Shuffler


def test_preshuffle_b_round_trip[
    N: Int, K_BYTES: Int
](ctx: DeviceContext) raises:
    var src_hb = ctx.enqueue_create_host_buffer[DType.uint8](N * K_BYTES)
    var dst_hb = ctx.enqueue_create_host_buffer[DType.uint8](N * K_BYTES)
    ctx.synchronize()

    for i in range(N * K_BYTES):
        src_hb[i] = UInt8(i & 0xFF)

    var src_tt = TileTensor(
        src_hb, row_major(Coord(Idx[1](), Idx[N](), Idx[K_BYTES]()))
    )
    var dst_tt = Shuffler[1].preshuffle_b_5d[N=N, K_BYTES=K_BYTES](
        src_tt, dst_hb
    )
    _ = dst_tt

    # Inverse re-index: the output byte at b_5d_grouped_layout(0, n, k_byte)
    # must equal the input byte at [n, k_byte] (row-major).
    for n in range(N):
        for k_byte in range(K_BYTES):
            var src_idx = n * K_BYTES + k_byte
            var dst_idx = Int(
                Shuffler[1].b_5d_grouped_layout[N=N, K_BYTES=K_BYTES](
                    Coord(Idx(0), Idx(n), Idx(k_byte))
                )
            )
            assert_equal(dst_hb[dst_idx], src_hb[src_idx])

    # Permutation, not duplication: every output position written exactly once.
    var seen_hb = ctx.enqueue_create_host_buffer[DType.uint8](N * K_BYTES)
    ctx.synchronize()
    for i in range(N * K_BYTES):
        seen_hb[i] = UInt8(0)
    for n in range(N):
        for k_byte in range(K_BYTES):
            var dst_idx = Int(
                Shuffler[1].b_5d_grouped_layout[N=N, K_BYTES=K_BYTES](
                    Coord(Idx(0), Idx(n), Idx(k_byte))
                )
            )
            assert_equal(seen_hb[dst_idx], UInt8(0))
            seen_hb[dst_idx] = UInt8(1)
    for i in range(N * K_BYTES):
        assert_equal(seen_hb[i], UInt8(1))


def test_preshuffle_scale_round_trip[
    MN: Int, K_SCALES: Int
](ctx: DeviceContext) raises:
    comptime MN_padded = Shuffler[1].scale_padded_mn(MN)
    var src_hb = ctx.enqueue_create_host_buffer[DType.uint8](MN * K_SCALES)
    var dst_hb = ctx.enqueue_create_host_buffer[DType.uint8](
        MN_padded * K_SCALES
    )
    ctx.synchronize()

    for i in range(MN * K_SCALES):
        src_hb[i] = UInt8(i & 0xFF)

    var src_tt = TileTensor(
        src_hb, row_major(Coord(Idx[1](), Idx[MN](), Idx[K_SCALES]()))
    )
    var dst_tt = Shuffler[1].preshuffle_scale_4d[MN=MN, K_SCALES=K_SCALES](
        src_tt, dst_hb
    )
    _ = dst_tt

    # Valid rows: each input byte appears at the layout-computed offset.
    for mn in range(MN):
        for k_scale in range(K_SCALES):
            var src_idx = mn * K_SCALES + k_scale
            var dst_idx = Int(
                Shuffler[1].scale_4d_grouped_layout[
                    MN_padded=MN_padded, K_SCALES=K_SCALES
                ](Coord(Idx(0), Idx(mn), Idx(k_scale)))
            )
            assert_equal(dst_hb[dst_idx], src_hb[src_idx])

    # Pad rows (mn in [MN, MN_padded)) must be zero.
    for mn in range(MN, MN_padded):
        for k_scale in range(K_SCALES):
            var dst_idx = Int(
                Shuffler[1].scale_4d_grouped_layout[
                    MN_padded=MN_padded, K_SCALES=K_SCALES
                ](Coord(Idx(0), Idx(mn), Idx(k_scale)))
            )
            assert_equal(dst_hb[dst_idx], UInt8(0))


def _b_offset[N: Int, K_BYTES: Int](n: Int, k_byte: Int) -> Int:
    return Int(
        Shuffler[1].b_5d_grouped_layout[N=N, K_BYTES=K_BYTES](
            Coord(Idx(0), Idx(n), Idx(k_byte))
        )
    )


def _scale_offset[MN_padded: Int, K_SCALES: Int](mn: Int, k_scale: Int) -> Int:
    return Int(
        Shuffler[1].scale_4d_grouped_layout[
            MN_padded=MN_padded, K_SCALES=K_SCALES
        ](Coord(Idx(0), Idx(mn), Idx(k_scale)))
    )


def test_preshuffle_b_offset_modes() raises:
    # Spot-check the 5 layout modes (n0, k0, klane, nlane, kpack) decode
    # to the strides documented in the file header.
    # Strides (bytes): n0=K0*1024, k0=1024, klane=256, nlane=16, kpack=1
    # for the standard 16x16x128 MFMA layout. Verify via incremental deltas.
    comptime N = 32
    comptime K_BYTES = 128  # K0 = 2

    # Same n, same k0: incrementing k_byte within a (klane, nlane, kpack) cell
    # walks contiguous output bytes.
    assert_equal(_b_offset[N, K_BYTES](0, 1), 1)
    assert_equal(_b_offset[N, K_BYTES](0, 15), 15)

    # Crossing kpack=16 boundary while staying in same (klane, n0) bumps nlane.
    # k_byte=16 -> tempk=16 -> klane=1, kpack=0 -> offset = klane*256 = 256.
    assert_equal(_b_offset[N, K_BYTES](0, 16), 256)

    # Crossing klane*kpack=64 boundary bumps k0.
    # k_byte=64 -> k0=1, tempk=0 -> offset = k0*1024 = 1024.
    assert_equal(_b_offset[N, K_BYTES](0, 64), 1024)

    # Crossing nlane=16 boundary with same (k0,klane,kpack) bumps n0.
    # n=15 vs n=16 with k_byte=0: n=15 has nlane=15 -> offset 240;
    # n=16 has n0=1 -> offset = n0 * K0 * 1024 = 2048.
    assert_equal(_b_offset[N, K_BYTES](15, 0), 15 * 16)
    assert_equal(_b_offset[N, K_BYTES](16, 0), 2 * 1024)


def test_preshuffle_scale_offset_modes() raises:
    # Strides: n0=256*K0, k0=256, k_lane=64, mn_lane=4, k_pack=2, mn_pack=1
    # for MNXdlPack=KXdlPack=2, XdlMNThread=16, XdlKThread=4.
    comptime MN_padded = 64
    comptime K_SCALES = 16  # K0 = 2

    # mn_pack=0 mn_lane=0 k_lane=0 k_pack=0 k0=0 n0=0 -> 0.
    assert_equal(_scale_offset[MN_padded, K_SCALES](0, 0), 0)

    # mn=1: tempn=1, mn_lane=1, mn_pack=0 -> mn_lane*4 + mn_pack = 4.
    assert_equal(_scale_offset[MN_padded, K_SCALES](1, 0), 4)

    # mn=16: tempn=16, mn_lane=0, mn_pack=1 -> mn_pack = 1.
    assert_equal(_scale_offset[MN_padded, K_SCALES](16, 0), 1)

    # mn=32: n0=1, others 0 -> n0 * 256 * K0 = 512.
    assert_equal(_scale_offset[MN_padded, K_SCALES](32, 0), 512)

    # k_scale=1: tempk=1, k_lane=1, k_pack=0 -> k_lane*64 = 64.
    assert_equal(_scale_offset[MN_padded, K_SCALES](0, 1), 64)

    # k_scale=4: tempk=4, k_lane=0, k_pack=1 -> k_pack*2 = 2.
    assert_equal(_scale_offset[MN_padded, K_SCALES](0, 4), 2)

    # k_scale=8: k0=1 -> k0 * 256 = 256.
    assert_equal(_scale_offset[MN_padded, K_SCALES](0, 8), 256)


def test_preshuffle_scale_padding() raises:
    # Padded MN rounds up to multiple of 32.
    assert_equal(Shuffler[1].scale_padded_mn(1), 32)
    assert_equal(Shuffler[1].scale_padded_mn(31), 32)
    assert_equal(Shuffler[1].scale_padded_mn(32), 32)
    assert_equal(Shuffler[1].scale_padded_mn(33), 64)
    assert_equal(Shuffler[1].scale_padded_mn(127), 128)
    assert_equal(Shuffler[1].scale_padded_mn(128), 128)


def main() raises:
    var ctx = DeviceContext()

    # Stride/offset spot-checks (cheap, run first).
    test_preshuffle_b_offset_modes()
    test_preshuffle_scale_offset_modes()
    test_preshuffle_scale_padding()

    # Round-trip across realistic shapes: tile_n=128, tile_k=256 (Kimi K2-class),
    # plus a small shape and a non-padded scale shape.
    test_preshuffle_b_round_trip[N=16, K_BYTES=64](ctx)
    test_preshuffle_b_round_trip[N=32, K_BYTES=128](ctx)
    test_preshuffle_b_round_trip[N=128, K_BYTES=256](ctx)

    test_preshuffle_scale_round_trip[MN=32, K_SCALES=8](ctx)
    test_preshuffle_scale_round_trip[MN=64, K_SCALES=16](ctx)
    test_preshuffle_scale_round_trip[MN=128, K_SCALES=8](ctx)
    # Non-32-aligned MN exercises pad-row zero-fill.
    test_preshuffle_scale_round_trip[MN=17, K_SCALES=8](ctx)
    test_preshuffle_scale_round_trip[MN=33, K_SCALES=16](ctx)
