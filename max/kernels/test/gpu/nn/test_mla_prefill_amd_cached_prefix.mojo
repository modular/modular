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
"""AMD/gfx950 MLA prefill against a pre-populated cache prefix (S=9).

Reuses ``run_test_paged_prefill`` (``_paged_prefill_test_utils.mojo``), whose
other callers are SM100-only and never set ``cache_length > 0``. On AMD every
bf16 S>1 MLA call routes to prefill, so a speculative-decode verify step (S=9
over a decode-written prefix) runs this kernel. A directly filled prefix
matches a decode-written one, since both write K_rope through
``fused_rope_rmsnorm_quantization_kernel``. The reference is the naive causal
MHA the SM100 variants already use.

Shapes: ``depth=192`` (nope=128, rope=64), 12 heads, batch 1. Per
``page_size``:

  - CONTROL: ``cache_length=0``; a failure here is not about the prefix.
  - TARGET: ``cache_length=503`` (``num_keys=512``).
  - EVEN_NO_SLACK: ``cache_length=448`` (``num_keys=457``). With TARGET at
    ``page_size=64`` this gives an even tile count and no LUT slack, where
    the double-buffered loop used to prefetch one tile past ``num_keys``.
    The paged-cache bounds check is a ``debug_assert``, so release builds
    read out of bounds silently.
  - SCATTERED_503 / SCATTERED_900: this sequence's pages in reversed order
    inside a 3x pool of unrelated pages (``scatter_pool_factor=3``), so a
    kernel that derives later pages from the first instead of re-reading
    the LUT fails. The reference follows the same LUT.
"""

from std.random import seed
from std.sys import get_defined_int, has_amd_gpu_accelerator

from max.gpu.host import DeviceContext

from _paged_prefill_test_utils import run_test_paged_prefill


comptime PAGE_SIZE = get_defined_int["page_size", 128]()


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_amd_gpu_accelerator():
            print("=== CONTROL fresh S=9, no cache prefix ===")
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=12,
                nope_depth=128,
                page_size=PAGE_SIZE,
                batch_size=1,
            ](9, 9, ctx)

            print("=== TARGET S=9 appended after cache_length=503 prefix ===")
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=12,
                nope_depth=128,
                page_size=PAGE_SIZE,
                batch_size=1,
                cache_length=503,
            ](9, 512, ctx)

            # See the module docstring's Shapes section for why 448 pairs
            # with 503.
            print(
                "=== EVEN_NO_SLACK S=9 appended after cache_length=448"
                " prefix ==="
            )
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=12,
                nope_depth=128,
                page_size=PAGE_SIZE,
                batch_size=1,
                cache_length=448,
            ](9, 457, ctx)

            # See "Scattered-LUT cases" in the module docstring.
            print("=== SCATTERED_503 S=9, scattered LUT, cache_length=503 ===")
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=12,
                nope_depth=128,
                page_size=PAGE_SIZE,
                batch_size=1,
                cache_length=503,
                scatter_pool_factor=3,
            ](9, 512, ctx)

            print("=== SCATTERED_900 S=9, scattered LUT, cache_length=900 ===")
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=12,
                nope_depth=128,
                page_size=PAGE_SIZE,
                batch_size=1,
                cache_length=900,
                scatter_pool_factor=3,
            ](9, 909, ctx)
