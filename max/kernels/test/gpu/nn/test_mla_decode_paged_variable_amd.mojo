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
from nn.attention.mha_mask import CausalMask, NullMask
from _mla_paged_variable_shell import PagedVariableTests
from _mla_test_vendors import AMDVendor

comptime T = PagedVariableTests[AMDVendor]


def main() raises:
    seed(42)

    with DeviceContext() as ctx:
        # ---------------------------------------------------------------
        # AMD MI355 / gfx950 path. The AMD MLA decode kernel supports
        # single-token decode (S=1) for any num_heads; the MTP token fold
        # (S = q_seq_len > 1) is FP8-only and requires num_heads <= 16
        # (Kimi-K2.5 TP=4). qkv_fp8's `test_decoding_fold` only covers the
        # contiguous-K dense uniform path — this exercises the fold through
        # the production paged KV cache + ragged input-row-offsets layout.
        #
        # The helpers are the ones `test_mla_decode_paged_variable_sm100`
        # runs; only the configs differ. The SM100 groups cover num_heads up
        # to 128 with bf16 + fp8-converter KV, outside the AMD MTP fold
        # envelope (FP8-only, num_heads <= 16). The AMD-specific value-add is
        # the ragged/paged multi-Q fold (Groups B/C).
        # ---------------------------------------------------------------
        print("=" * 72)
        print("MLA Decode Paged Variable-Length Test (AMD MI355/gfx950)")
        print("=" * 72)
        print()

        # -----------------------------------------------------------
        # Group A: S=1 paged decode (single query token per batch).
        # Confirms the AMD paged decode path works through this harness
        # for both BF16 and native-FP8 (Q=FP8, KV=FP8) decode.
        # -----------------------------------------------------------
        print("--- Group A: S=1 paged decode (16 heads) ---")

        var amd_cl: List[Int] = [30, 256, 640, 1024]
        T.run_test_paged_variable[.bfloat16, DType.bfloat16, 16](
            "amd_s1_bf16", amd_cl, ctx
        )
        T.run_test_paged_variable_native_fp8[16](
            "amd_s1_native_fp8", amd_cl, ctx
        )

        # GLM-5.3 TP8 per-rank shape: 8 heads, depth 576 (kv_lora 512 + rope
        # 64), FP8 paged KV, page 128, S=1 decode; run np=1 and np=4.
        print("--- Group A: S=1 paged decode (8 heads, GLM-5.3 TP8) ---")

        var amd_cl_h8: List[Int] = [30, 256, 640, 1024]
        T.run_test_paged_variable_native_fp8[8](
            "amd_s1_native_fp8_h8_np1", amd_cl_h8, ctx
        )
        T.run_test_paged_variable_native_fp8[8](
            "amd_s1_native_fp8_h8_np4",
            amd_cl_h8,
            ctx,
            num_partitions=Optional(4),
        )

        # GLM-5.3 stages the decode Q in bf16; exercise that shape (8 heads,
        # depth 576, v 512, page 128), variable lengths, np=1 and split-K.
        T.run_test_paged_variable[.bfloat16, DType.bfloat16, 8](
            "amd_s1_bf16_h8_np1", amd_cl_h8, ctx
        )
        T.run_test_paged_variable[.bfloat16, DType.bfloat16, 8](
            "amd_s1_bf16_h8_np4",
            amd_cl_h8,
            ctx,
            num_partitions=Optional(4),
        )

        # -----------------------------------------------------------
        # Group B: uniform multi-Q FP8 fold (S>1, every sequence has
        # exactly q_max_seq_len query tokens). M = num_heads*S <= 128 →
        # supported. This is the AMD MTP fold through the paged + ragged
        # layout path. Expected to PASS.
        # -----------------------------------------------------------
        print("--- Group B: uniform multi-Q FP8 fold (16 heads) ---")

        var amd_sl_s2: List[Int] = [2, 2, 2, 2]  # M=32, warp-local (2,1)
        T.run_test_paged_variable_ragged_q_native_fp8[16](
            "amd_uniform_s2", amd_cl, amd_sl_s2, NullMask(), ctx
        )

        var amd_sl_s4: List[Int] = [4, 4, 4, 4]  # M=64, warp-local (4,1)
        T.run_test_paged_variable_ragged_q_native_fp8[16](
            "amd_uniform_s4", amd_cl, amd_sl_s4, NullMask(), ctx
        )

        # -----------------------------------------------------------
        # Group C: VARIABLE ragged-Q FP8 fold — POSITIVE correctness.
        # Per-batch query length differs while q_max_seq_len is only the
        # padding ceiling; the runtime per-sequence length drives the SRD
        # clamp, causal offset, split-K stat writer, and ragged reducer, so
        # short sequences' pad rows are dropped and live rows land at the
        # right slot. The reference (mha_gpu_naive per batch) is fully
        # variable-aware. Covers both np=1 (small caches) and split-K (np>1,
        # large caches — the reducer's ragged remap + pad-row skip).
        #
        # Both masks are run: NullMask validates the ragged remap / reduce /
        # pad-row skip, and CausalMask validates the per-token causal offset
        # `score_row = num_keys - seq_len + token`. Causal is the only mask
        # that *discriminates* the runtime per-sequence seq_len from the
        # comptime q_max_seq_len ceiling (under NullMask the offset never
        # affects output), so the variable configs below are the only place
        # that per-token offset is exercised with a non-uniform seq_len.
        # -----------------------------------------------------------
        print(
            "--- Group C: variable ragged-Q FP8 fold correctness (16 heads) ---"
        )

        # np=1: tiny caches (1 page each) so the heuristic returns
        # num_partitions=1 → the decode kernel writes the ragged output
        # directly, not via the reducer.
        var amd_cl_var_tiny: List[Int] = [30, 64, 96, 120]
        var amd_sl_var: List[Int] = [1, 3, 2, 4]
        T.run_test_paged_variable_ragged_q_native_fp8[16](
            "amd_var_np1_tiny", amd_cl_var_tiny, amd_sl_var, NullMask(), ctx
        )
        # CAUSAL np=1: per-token causal offset on the direct ragged-write
        # path (no reducer). seq_lens=[1,3,2,4] so the runtime per-sequence
        # length != q_max_seq_len=4 for 3 of 4 batches — the discriminating
        # case a comptime-seq_len regression would fail.
        T.run_test_paged_variable_ragged_q_native_fp8[16](
            "amd_var_np1_tiny_causal",
            amd_cl_var_tiny,
            amd_sl_var,
            CausalMask(),
            ctx,
        )

        # np>1 (small): small caches, variable seq_lens (M up to 64).
        var amd_cl_var_small: List[Int] = [256, 1024, 512, 2048]
        T.run_test_paged_variable_ragged_q_native_fp8[16](
            "amd_var_npk_small",
            amd_cl_var_small,
            amd_sl_var,
            NullMask(),
            ctx,
        )
        # CAUSAL np>1: same variable seq_lens through the row-keyed split-K
        # reducer's ragged remap — validates the per-token causal offset is
        # preserved across the reduce (a head-keyed or comptime-seq_len bug
        # would mismatch the per-batch causal reference).
        T.run_test_paged_variable_ragged_q_native_fp8[16](
            "amd_var_npk_small_causal",
            amd_cl_var_small,
            amd_sl_var,
            CausalMask(),
            ctx,
        )

        # np>1: large caches force split-K; variable seq_lens exercise the
        # reducer's ragged output remap + short-sequence pad-row skip.
        var amd_cl_var_big: List[Int] = [256, 65536, 1024, 32768]
        T.run_test_paged_variable_ragged_q_native_fp8[16](
            "amd_var_npk_mixed", amd_cl_var_big, amd_sl_var, NullMask(), ctx
        )

        # np>1, uniform-large cache, variable seq_lens (every batch deep
        # context → every batch split-K; ragged Q row count varies).
        var amd_cl_var_uniform_big: List[Int] = [
            65536,
            65536,
            65536,
            65536,
        ]
        var amd_sl_var_b: List[Int] = [1, 4, 2, 3]
        T.run_test_paged_variable_ragged_q_native_fp8[16](
            "amd_var_npk_uniformcache",
            amd_cl_var_uniform_big,
            amd_sl_var_b,
            NullMask(),
            ctx,
        )

        # H=8 variable case: DEFERRED. It passes under the JIT `mojo run`
        # path, but the H=8 ragged ladder instantiates all S=1..8 kernels and
        # the S=6 fold (`_48x128x128_True_nqh8`) crashes the LLVM register
        # allocator under the AOT `--config=kernels` build (a pre-existing
        # backend bug — the dense `_False_nqh8` twin compiles fine; only the
        # ragged variant tips regalloc over its VGPR threshold). Re-enable
        # once that's fixed; production fold is num_heads=16, covered above.

        print("=" * 72)
        print("ALL TESTS PASSED")
        print("=" * 72)
