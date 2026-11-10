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

"""Test MLA decode with paged KV cache and variable sequence lengths.

This test exercises the splitK ragged layout path that was fixed:
  Bug A: o_accum_split layout mismatch (ragged vs padded)
  Bug B: LSE stride mismatch (per-batch seq_len vs q_max_seq_len)
  Bug C: Combine kernel output indexing (padded vs ragged)

The key stress scenarios:
  - Variable KV cache lengths per batch (30 to 32768 tokens), causing
    different num_partitions per batch and many empty splits
  - Extreme disparity in cache lengths within the same batch
"""

from std.random import seed

from max.gpu.host import DeviceContext
from _mla_paged_variable_shell import (
    PagedVariableTests,
    is_benchmark,
    make_uniform,
)
from _mla_test_vendors import SM100Vendor

comptime T = PagedVariableTests[SM100Vendor]


def main() raises:
    seed(42)

    with DeviceContext() as ctx:
        if is_benchmark():
            # -----------------------------------------------------------
            # Benchmark mode: time kernel execution, no verification
            # DeepSeek-V2/R1 full: 128 heads, depth=576
            # -----------------------------------------------------------
            print("=" * 72)
            print("MLA Decode Paged Variable-Length BENCHMARK (B200)")
            print("=" * 72)
            print()

            # Batch size 1: single long context
            T.run_bench_uniform_all_three[128]("bs1_32k", 1, 32768, ctx)

            # Batch size 1: medium context
            T.run_bench_uniform_all_three[128]("bs1_4k", 1, 4096, ctx)

            # Batch size 2: variable lengths
            var b2: List[Int] = [4096, 32768]
            T.run_bench_all_three_kv_types[128]("bs2_4k_32k", b2, ctx)

            # Batch size 4: mixed lengths
            var b4: List[Int] = [1024, 4096, 16384, 32768]
            T.run_bench_all_three_kv_types[128]("bs4_mixed", b4, ctx)

            # Batch size 8: mixed lengths (production-like)
            var b8: List[Int] = [
                128,
                512,
                1024,
                4096,
                8192,
                16384,
                24576,
                32768,
            ]
            T.run_bench_all_three_kv_types[128]("bs8_mixed", b8, ctx)

            # Batch size 8: all long (worst case)
            T.run_bench_uniform_all_three[128]("bs8_all32k", 8, 32768, ctx)

            print("=" * 72)
            print("BENCHMARK COMPLETE")
            print("=" * 72)

        else:
            # -----------------------------------------------------------
            # Correctness mode: numerical verification against reference
            # -----------------------------------------------------------
            print("=" * 72)
            print("MLA Decode Paged Variable-Length Test (B200)")
            print("=" * 72)
            print()

            # -----------------------------------------------------------
            # Group 1: Small caches with reference check (16 heads)
            # Both bf16 and fp8 KV types are tested to exercise both
            # mla_decode_sm100_kv_bf16 and mla_decode_sm100_kv_fp8
            # kernels.
            # -----------------------------------------------------------
            print(
                "--- Group 1: Small caches with reference check (16 heads) ---"
            )

            var cl1: List[Int] = [30, 50, 80, 100]
            T.run_both_kv_types[16]("short_uniform_q1", cl1, ctx)

            var cl2: List[Int] = [30, 256, 640, 1024]
            T.run_both_kv_types[16]("variable_cache_q1", cl2, ctx)

            # -----------------------------------------------------------
            # Group 2: Large caches with reference check (16 heads)
            # -----------------------------------------------------------
            print(
                "--- Group 2: Large caches with reference check (16 heads) ---"
            )

            var cl4: List[Int] = [30, 1024, 8192, 32768]
            T.run_both_kv_types[16]("extreme_disparity_q1", cl4, ctx)

            var cl6: List[Int] = [
                30,
                128,
                256,
                512,
                1024,
                4096,
                16384,
                32768,
            ]
            T.run_both_kv_types[16]("mixed_8batch_q1", cl6, ctx)

            # -----------------------------------------------------------
            # Group 3: Higher head counts (64, 128) with verification
            # -----------------------------------------------------------
            print("--- Group 3: Higher head counts with reference check ---")

            var cl7: List[Int] = [30, 512, 4096, 16384]
            T.run_both_kv_types[64]("64heads_disparity_q1", cl7, ctx)

            var cl8: List[Int] = [30, 1024, 8192, 32768]
            T.run_both_kv_types[128]("128heads_extreme_q1", cl8, ctx)

            # -----------------------------------------------------------
            # Group 4: Latency sensitive configs (uniform cache lengths)
            # Matches bench_mla_decode.yaml latency sensitive section
            # -----------------------------------------------------------
            print("--- Group 4: Latency sensitive configs (16 heads) ---")

            T.run_uniform_both[16]("lat_bs1_64k", 1, 65536, ctx)
            T.run_uniform_both[16]("lat_bs4_32k", 4, 32768, ctx)
            T.run_uniform_both[16]("lat_bs8_64k", 8, 65536, ctx)

            # -----------------------------------------------------------
            # Group 5: Production-representative configs
            # SGLang/Ant Group: 12-48 per GPU, vLLM TP8: up to 128
            # KV cache 4K-8K is the common production range
            # -----------------------------------------------------------
            print(
                "--- Group 5: Production-representative configs (16 heads) ---"
            )

            T.run_uniform_both[16]("prod_bs16_4k", 16, 4096, ctx)
            T.run_uniform_both[16]("prod_bs32_8k", 32, 8192, ctx)
            T.run_uniform_both[16]("prod_bs128_1k", 128, 1024, ctx)

            # -----------------------------------------------------------
            # Group 6: Extra-large batch stress test configs
            # -----------------------------------------------------------
            print(
                "--- Group 6: Extra-large batch stress test configs"
                " (16 heads) ---"
            )

            T.run_uniform_both[16]("stress_bs256_4k", 256, 4096, ctx)

            # -----------------------------------------------------------
            # Group 7: Very small cache_len (<1K)
            # Tests the kernel with very few KV pages -- potential
            # underutilization. Covers prefill-like decode scenarios
            # with tiny cache and high batch.
            # -----------------------------------------------------------
            print("--- Group 7: Very small cache_len configs (16 heads) ---")

            T.run_uniform_both[16]("tiny_bs128_64", 128, 64, ctx)
            T.run_uniform_both[16]("tiny_bs64_512", 64, 512, ctx)

            # -----------------------------------------------------------
            # Group 8: Mid-range configs (fills 8K->32K gap with 16K)
            # -----------------------------------------------------------
            print("--- Group 8: Mid-range configs (16 heads) ---")

            T.run_uniform_both[16]("mid_bs8_16k", 8, 16384, ctx)

            # -----------------------------------------------------------
            # Group 9: Large cache_len (>64K up to 163K max context)
            # Tests deep into long-context territory for DeepSeek.
            # -----------------------------------------------------------
            print("--- Group 9: Large cache_len configs (16 heads) ---")

            T.run_uniform_both[16]("large_bs1_163k", 1, 163840, ctx)

            # -----------------------------------------------------------
            # Group 10: Non-power-of-2 cache_len configs
            # Validates that kernel dispatch, page-level partitioning,
            # and split-k reduction handle arbitrary cache lengths.
            # -----------------------------------------------------------
            print(
                "--- Group 10: Non-power-of-2 cache_len configs (16 heads) ---"
            )

            T.run_uniform_both[16]("npo2_bs16_7777", 16, 7777, ctx)
            T.run_uniform_both[16]("npo2_bs2_50000", 2, 50000, ctx)

            # -----------------------------------------------------------
            # Group 11: Variable cache_len across batch (log-spaced)
            # Tests the kernel with heterogeneous cache lengths in a
            # single batch, sampled in log space from 128 to 32768.
            # This exercises the variable-length dispatch path,
            # page-level partitioning with different split counts per
            # sequence, and the combine kernel with mixed workloads.
            # -----------------------------------------------------------
            print(
                "--- Group 11: Variable cache_len across batch"
                " (log-spaced, 16 heads) ---"
            )

            # 10 sequences with log-spaced cache lengths from 128 to
            # 32768.
            var variable_logspace: List[Int] = [
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                16384,
                24576,
                32768,
            ]
            T.run_both_kv_types[16]("variable_logspace", variable_logspace, ctx)

            # -----------------------------------------------------------
            # Group 12: Multi-Q token tests (q_max_seq_len > 1)
            # Tests the ragged layout path where the grid iterates
            # over multiple query tokens per batch entry.  Verifies:
            # - Q/output row offset arithmetic for multi-token decode
            # - o_accum_split padded seq dimension in combine kernel
            # - FP8 blockwise scaling is independent of Q seq_len
            # -----------------------------------------------------------
            print("--- Group 12: Multi-Q token tests (q_max_seq_len > 1) ---")

            # seq_len=2, small batch, short cache (16 heads)
            var mq1: List[Int] = [30, 50, 80, 100]
            T.run_multiq_both_kv_types[16]("multiq2_short", mq1, 2, ctx)

            # seq_len=4, mixed cache lengths (16 heads)
            var mq2: List[Int] = [30, 256, 640, 1024]
            T.run_multiq_both_kv_types[16]("multiq4_mixed", mq2, 4, ctx)

            # seq_len=3, extreme disparity (16 heads)
            var mq3: List[Int] = [30, 1024, 8192, 32768]
            T.run_multiq_both_kv_types[16]("multiq3_extreme", mq3, 3, ctx)

            # seq_len=2, 128 heads (full DeepSeek config)
            var mq4: List[Int] = [30, 1024, 8192, 32768]
            T.run_multiq_both_kv_types[128]("multiq2_128heads", mq4, 2, ctx)

            # -----------------------------------------------------------
            # Group 13: Truly ragged Q — each batch has a DIFFERENT
            # number of query tokens.
            # This is the ONLY test that exercises the early-exit path
            # in OffsetPosition where seq_idx >= per-batch seq_len,
            # and where num_keys = cache_len + per-batch seq_len (NOT
            # + q_max_seq_len).
            # -----------------------------------------------------------
            print("--- Group 13: Ragged Q — variable per-batch Q seq_len ---")

            # Basic ragged: seq_lens=[1,3,2,4], mixed cache (16 heads)
            var rq_cl1: List[Int] = [256, 1024, 512, 2048]
            var rq_sl1: List[Int] = [1, 3, 2, 4]
            T.run_ragged_q_both_kv_types[16](
                "ragged_basic", rq_cl1, rq_sl1, ctx
            )

            # With 128 heads (full DeepSeek config)
            var rq_cl3: List[Int] = [256, 4096, 1024, 8192]
            var rq_sl3: List[Int] = [2, 1, 5, 3]
            T.run_ragged_q_both_kv_types[128](
                "ragged_128heads", rq_cl3, rq_sl3, ctx
            )

            # All-one seq_len except last (tests early-exit for most
            # CTAs in the Y dimension)
            var rq_cl4: List[Int] = [
                30,
                256,
                640,
                1024,
                4096,
                8192,
            ]
            var rq_sl4: List[Int] = [1, 1, 1, 1, 1, 6]
            T.run_ragged_q_both_kv_types[16](
                "ragged_mostly_1", rq_cl4, rq_sl4, ctx
            )

            # -----------------------------------------------------------
            # Group 14: Native FP8 (Q=FP8 + KV=FP8, output=BF16)
            # Tests the 3-WG native FP8 kernel through the paged path
            # with split-K + PDL. Q is quantized to FP8 before being
            # sent to the kernel. The reference uses BF16 Q and K.
            # -----------------------------------------------------------
            print(
                "--- Group 14: Native FP8 paged"
                " (Q=FP8, KV=FP8, output=BF16) ---"
            )

            # Small caches (16 heads)
            var nfp8_1: List[Int] = [30, 50, 80, 100]
            T.run_test_paged_variable_native_fp8[16](
                "native_fp8_short_16h", nfp8_1, ctx
            )

            # Variable cache lengths (16 heads)
            var nfp8_2: List[Int] = [30, 256, 640, 1024]
            T.run_test_paged_variable_native_fp8[16](
                "native_fp8_variable_16h", nfp8_2, ctx
            )

            # Extreme disparity (16 heads)
            var nfp8_3: List[Int] = [30, 1024, 8192, 32768]
            T.run_test_paged_variable_native_fp8[16](
                "native_fp8_extreme_16h", nfp8_3, ctx
            )

            # Full DeepSeek config (128 heads)
            var nfp8_4: List[Int] = [30, 1024, 8192, 32768]
            T.run_test_paged_variable_native_fp8[128](
                "native_fp8_extreme_128h", nfp8_4, ctx
            )

            # Production-like: moderate batch, medium cache (128 heads)
            T.run_test_paged_variable_native_fp8[128](
                "native_fp8_prod_bs8_4k",
                make_uniform(8, 4096),
                ctx,
            )

            # Large batch, short cache (128 heads)
            T.run_test_paged_variable_native_fp8[128](
                "native_fp8_bs64_1k",
                make_uniform(64, 1024),
                ctx,
            )

            print("=" * 72)
            print("ALL TESTS PASSED")
            print("=" * 72)
