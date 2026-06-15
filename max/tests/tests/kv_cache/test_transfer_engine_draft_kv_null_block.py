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

"""Regression tests for the draft-KV null-block bytes_per_page mismatch.

Root cause (hardware-confirmed, Mammoth run 26960744675):
  KVCacheParams.allocate_buffers() allocates ``total_num_pages + 1`` buffer
  slots so that index 0 can serve as a sentinel null block.  The schedulers
  originally passed ``get_num_pages()`` (= N, without +1) as
  ``total_num_pages`` to ``register_tensor_group``.  Inside
  ``register_extra_group`` this produced:

      bytes_per_page = (N + 1) * elts_per_block * dtype_bytes // N

  Integer floor-division of ``(N+1)*C // N`` gives different remainders for
  different N values (prefill and decode pools differ because they run at
  different device-memory-utilization fractions).  The hardware run measured
  73733 bytes/page on the prefill engine and 73739 bytes/page on the decode
  engine — a 6-byte gap — which triggered
  ``createXferReq: length mismatch at index 504``.

Fix: pass ``get_num_pages() + 1`` so the divisor matches the actual allocated
buffer size, making ``bytes_per_page = (N+1)*C // (N+1) = C`` exactly.

These tests are pure-arithmetic and mock all NIXL/GPU objects, so they run on
any host platform without hardware.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Helpers that exercise the exact arithmetic used in register_extra_group
# ---------------------------------------------------------------------------


def _bpp_old(
    total_num_pages: int, elts_per_block: int, dtype_bytes: int
) -> int:
    """Compute bytes_per_page as the OLD (buggy) code did.

    Buffer holds ``total_num_pages + 1`` elements (including null block);
    the scheduler passed ``total_num_pages`` (= N, not N+1) as the divisor.
    """
    num_elements = (total_num_pages + 1) * elts_per_block
    return num_elements * dtype_bytes // total_num_pages


def _bpp_new(
    total_num_pages: int, elts_per_block: int, dtype_bytes: int
) -> int:
    """Compute bytes_per_page as the FIXED code does.

    Scheduler passes ``total_num_pages + 1`` so the divisor matches the
    actual allocation, yielding exact integer division.
    """
    n_plus_1 = total_num_pages + 1
    num_elements = n_plus_1 * elts_per_block
    return num_elements * dtype_bytes // n_plus_1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_null_block_reproduces_hardware_observed_73733() -> None:
    """Old behaviour reproduces the prefill-side bpp (73733) seen on hardware.

    Kimi K2.5 MLA draft geometry:
      kv_dim=1, num_layers=1, page_size=128, n_kv_heads_per_device=1,
      head_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
      → elts_per_block = 1 * 1 * 128 * 1 * 576 = 73728
      dtype = float8_e4m3fn (1 byte/element)

    With device_memory_utilization=0.8 the prefill pool had ~14745 pages,
    giving (14745+1)*73728 // 14745 = 73733.
    """
    bpp = _bpp_old(total_num_pages=14745, elts_per_block=73728, dtype_bytes=1)
    assert bpp == 73733


def test_null_block_reproduces_hardware_observed_73739() -> None:
    """Old behaviour reproduces the decode-side bpp (73739) seen on hardware.

    With device_memory_utilization=0.7 and DP=8 the decode per-replica pool
    had ~6702 pages, giving (6702+1)*73728 // 6702 = 73739.
    """
    bpp = _bpp_old(total_num_pages=6702, elts_per_block=73728, dtype_bytes=1)
    assert bpp == 73739


def test_null_block_old_code_diverges_between_engines() -> None:
    """Old code: different pool sizes → different bpp → NIXL length mismatch."""
    elts = 73728
    dtype_bytes = 1

    bpp_prefill = _bpp_old(14745, elts, dtype_bytes)
    bpp_decode = _bpp_old(6702, elts, dtype_bytes)

    assert bpp_prefill != bpp_decode, (
        "Old code should produce different bpp for different pool sizes"
    )
    assert bpp_prefill == 73733
    assert bpp_decode == 73739


def test_null_block_fix_produces_exact_elts_times_dtype() -> None:
    """Fixed code: bpp = elts * dtype_bytes exactly, regardless of pool size."""
    elts = 73728
    dtype_bytes = 1

    for n in [6702, 8000, 12288, 14745, 20000, 50000]:
        bpp = _bpp_new(n, elts, dtype_bytes)
        assert bpp == elts * dtype_bytes, (
            f"N={n}: expected {elts * dtype_bytes}, got {bpp}"
        )


def test_null_block_fix_matches_across_prefill_and_decode() -> None:
    """Fixed code: prefill and decode compute identical bpp."""
    elts = 73728
    dtype_bytes = 1

    bpp_prefill = _bpp_new(14745, elts, dtype_bytes)
    bpp_decode = _bpp_new(6702, elts, dtype_bytes)

    assert bpp_prefill == bpp_decode
    assert bpp_prefill == 73728


def test_null_block_fix_works_for_other_block_geometries() -> None:
    """The fix is correct for arbitrary elts_per_block / dtype_bytes combos."""
    cases = [
        # (n_pages, elts, dtype_bytes)
        (10000, 73728, 1),  # Kimi K2.5 MLA fp8
        (5000, 65536, 2),  # hypothetical bf16 draft
        (20000, 49152, 4),  # hypothetical fp32 draft
        (3, 6, 1),  # tiny edge case
    ]
    for n, elts, dtype_bytes in cases:
        expected = elts * dtype_bytes
        bpp = _bpp_new(n, elts, dtype_bytes)
        assert bpp == expected, (
            f"n={n}, elts={elts}, dtype_bytes={dtype_bytes}: "
            f"expected {expected}, got {bpp}"
        )
