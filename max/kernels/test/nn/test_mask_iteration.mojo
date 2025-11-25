# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from nn.mha_mask import (
    CausalMask,
    SlidingWindowCausalMask,
    ChunkedCausalMask,
    MHAMask,
    TileMaskStatus,
)
from utils.index import Index
from testing import assert_equal, assert_true


fn compute_total_iters0[
    MaskType: MHAMask, //, BM: Int, BN: Int
](mask: MaskType, q_row: UInt32, end: UInt32) -> UInt32:
    var kv_row: UInt32 = 0
    while (
        mask.status(
            Index[dtype = DType.int32](Int(q_row), Int(kv_row)),
            Index[dtype = DType.int32](BM, BN),
        )
        == TileMaskStatus.FULL_MASK
    ):
        kv_row += BN
    var iter_count: UInt32 = 0
    while True:
        kv_row += BN
        if kv_row >= end:
            break
        if (
            mask.status(
                Index[dtype = DType.int32](Int(q_row), Int(kv_row)),
                Index[dtype = DType.int32](BM, BN),
            )
            == TileMaskStatus.FULL_MASK
        ):
            continue
        iter_count += 1
    return iter_count + 1


fn compute_total_iters1[
    MaskType: MHAMask, //, BM: Int, BN: Int
](mask: MaskType, q_row: UInt32, end: UInt32) -> UInt32:
    var iter_count: UInt32 = 0
    var kv_row: UInt32 = 0
    while kv_row < end:
        iter_count += Int(
            mask.status(
                Index[dtype = DType.int32](Int(q_row), Int(kv_row)),
                Index[dtype = DType.int32](BM, BN),
            )
            != TileMaskStatus.FULL_MASK
        )
        kv_row += BN
    return iter_count


fn status[
    MaskType: MHAMask, //, BM: Int, BN: Int
](mask: MaskType, q_row: UInt32, kv_row: UInt32) -> TileMaskStatus:
    return mask.status(
        Index[dtype = DType.int32](q_row, kv_row),
        Index[dtype = DType.int32](BM, BN),
    )


fn test_mask[
    MaskType: MHAMask, //, BM: Int, BN: Int, page_size: Int = 1
](mask: MaskType, q_row: UInt32, end: UInt32) raises:
    var kv_row: UInt32 = mask.start_column[BM, BN, page_size](q_row)
    comptime mask_sets = MaskType.nonfull_sets[BM, BN]()
    comptime num_sets = len(mask_sets)
    mask_ends = mask.masked_set_ends[BM=BM, BN=BN, page_size=page_size](
        q_row, end
    )

    var ref_mask: TileMaskStatus
    if kv_row > 0:
        ref_mask = status[BM, BN](mask, q_row, kv_row - BN)
        assert_equal(TileMaskStatus.FULL_MASK, ref_mask)
    var total_iters: UInt32 = 0
    for i in range(num_sets):
        mask_status = mask_sets[i]
        var iters: UInt32 = (
            mask_ends[i] if i == 0 else mask_ends[i] - mask_ends[i - 1]
        )
        total_iters += iters
        for _ in range(iters):
            if kv_row >= end:
                print(
                    MaskType.name(), ": kv_row end iters =", kv_row, end, iters
                )
            assert_true(kv_row < end)
            ref_mask = status[BM, BN](mask, q_row, kv_row)
            if mask_status != ref_mask:
                print("mask_ends = [", end="")
                for i in range(num_sets):
                    if i > 0:
                        print(", ", end="")
                    print(mask_ends[i], end="")
                print("]")
                print("q_row num_keys kv_row =", q_row, end, kv_row)
                print(
                    "mask_status, ref_mask = ",
                    mask_status,
                    ", ",
                    ref_mask,
                    sep="",
                )
            assert_equal(mask_status, ref_mask)
            kv_row += BN
    if kv_row < end:
        ref_mask = status[BM, BN](mask, q_row, kv_row)
        assert_equal(TileMaskStatus.FULL_MASK, ref_mask)

    calc_total_iter = mask.total_iters[BM, BN, page_size](q_row, end)
    if total_iters != calc_total_iter:
        print("mask_ends = [", end="")
        for i in range(num_sets):
            if i > 0:
                print(", ", end="")
            print(mask_ends[i], end="")
        print("]")
        print("q_row =", q_row)
        print("num_keys =", end)
        print("start_col =", mask.start_column[BM, BN, page_size](q_row))
        print("calc_total_iter =", calc_total_iter)
    assert_equal(total_iters, mask.total_iters[BM, BN, page_size](q_row, end))


fn main() raises:
    # alias BM = 2
    # alias BN = 2
    comptime BM = 128
    comptime BN = 128
    comptime causal_mask = CausalMask()
    comptime sliding_mask16 = SlidingWindowCausalMask[16]()
    comptime sliding_mask1024 = SlidingWindowCausalMask[1024]()
    comptime chunked_causal_mask = ChunkedCausalMask[256]()
    for num_keys in range(1, 8193):
        for q_row in range(num_keys):
            test_mask[BM=BM, BN=BN, page_size=1](causal_mask, q_row, num_keys)
            test_mask[BM=BM, BN=BN, page_size=512](causal_mask, q_row, num_keys)
            test_mask[BM=BM, BN=BN, page_size=1](
                sliding_mask16, q_row, num_keys
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                sliding_mask16, q_row, num_keys
            )
            test_mask[BM=BM, BN=BN, page_size=1](
                sliding_mask1024, q_row, num_keys
            )
            test_mask[BM=BM, BN=BN, page_size=512](
                sliding_mask1024, q_row, num_keys
            )
            count0 = compute_total_iters0[BM=BM, BN=BN](
                chunked_causal_mask, q_row, num_keys
            )
            count1 = compute_total_iters1[BM=BM, BN=BN](
                chunked_causal_mask, q_row, num_keys
            )
            count2 = chunked_causal_mask.total_iters[BM=BM, BN=BN, page_size=1](
                q_row, num_keys
            )
            count3 = chunked_causal_mask.total_iters[
                BM=BM, BN=BN, page_size=512
            ](q_row, num_keys)
            if count0 != count1 or count0 != count2 or count0 != count3:
                print("q_row, num_keys =", q_row, num_keys)
                print(
                    "count0, count1, count2, count3 =",
                    count0,
                    count1,
                    count2,
                    count3,
                )
            assert_equal(count0, count1)
            assert_equal(count0, count2)
            assert_equal(count0, count3)
