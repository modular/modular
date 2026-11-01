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
"""SM100 FA4 tile configs for a 192-wide key with a 128-wide value.

The key is split MLA-style: its first 128 columns share the V-sized ring slot
and the last 64 go to the rope buffers, so `nope_depth == ov_depth` and the BN
and slot budgets are the d128 ones. The 2Q prefill tile then has 1,876 B of
shared memory to spare, so these checks are compile-time: a budget change that
drops it to a non-shared single-stage tile fails the build rather than a
dispatch at run time.
"""

from std.testing import assert_equal, assert_false, assert_true

from max.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.attention.gpu.nvidia.sm100.attention import FA4Config
from nn.attention.gpu.nvidia.sm100.mla_prefill_utils import (
    select_mla_prefill_config,
)


comptime dtype = DType.bfloat16
comptime page_size = 128


def split_config_2q[num_kv_heads: Int]() -> FA4Config[dtype]:
    return FA4Config[dtype](
        num_q_heads=16 * num_kv_heads,
        group=16,
        qk_depth=192,
        ov_depth=128,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        page_size=page_size,
        is_mla=False,
    )


def check_split_configs[num_kv_heads: Int]() raises:
    comptime cfg = split_config_2q[num_kv_heads]()
    comptime assert cfg.supported(), cfg.description()
    comptime assert cfg.k_tail(), cfg.description()
    comptime assert not cfg.can_switch_to_1q(), cfg.description()
    comptime assert not cfg.crossp_on(), cfg.description()

    assert_equal(cfg.BN, 128)
    assert_equal(cfg.num_kv_stages, 3)
    assert_equal(cfg.num_rope_buffers(), 2)
    assert_equal(cfg.num_qk_stages, 1)
    assert_equal(cfg.padded_nope_depth, 128)
    assert_equal(cfg.rope_depth(), 64)
    assert_equal(cfg.k_tma_box_cols(), 64)
    assert_equal(cfg.tmem_used, 512)
    assert_equal(cfg.smem_used, 230572)

    comptime cfg_1q = cfg.with_num_q(1)
    comptime assert cfg_1q.supported(), cfg_1q.description()
    comptime assert not cfg_1q.k_tail(), cfg_1q.description()
    assert_false(cfg_1q.use_shared_kv)
    assert_equal(cfg_1q.BN, 128)
    assert_equal(cfg_1q.num_kv_stages, 2)
    assert_equal(cfg_1q.num_qk_stages, 3)
    assert_equal(cfg_1q.BK0, 64)
    assert_equal(cfg_1q.k_tma_box_cols(), 64)
    assert_equal(cfg_1q.smem_used, 214244)

    print("2Q K-tail, kv heads", num_kv_heads, ":")
    print(cfg.description())
    print("1Q, kv heads", num_kv_heads, ":")
    print(cfg_1q.description())


def test_existing_d128_switch_unchanged() raises:
    comptime cfg = FA4Config[dtype](
        num_q_heads=16,
        group=16,
        qk_depth=128,
        ov_depth=128,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        page_size=page_size,
        is_mla=False,
    )
    comptime assert cfg.supported(), cfg.description()
    assert_false(cfg.k_tail())
    assert_equal(cfg.k_tma_box_cols(), cfg.BK0)
    assert_true(cfg.can_switch_to_1q())


def test_mla_prefill_switch_unchanged() raises:
    # MLA's 64-wide rope region is not a K tail: the K-tail refusal must leave
    # MLA prefill's in-kernel 2Q->1Q demotion (DeepSeek shape) available.
    comptime cfg = select_mla_prefill_config[
        dtype, rope_gmem_dtype=dtype, rope_mma_dtype=dtype
    ](num_q_heads=16, group=1, depth=192, page_size=page_size, v_depth=-1)
    comptime assert cfg.supported(), cfg.fa4_config.description()
    assert_equal(cfg.fa4_config.rope_depth(), 64)
    assert_false(cfg.fa4_config.k_tail())
    assert_true(cfg.can_switch_to_1q())
    assert_true(cfg.fa4_config.can_switch_to_1q())


def main() raises:
    check_split_configs[1]()
    check_split_configs[2]()
    check_split_configs[4]()
    test_existing_d128_switch_unchanged()
    test_mla_prefill_switch_unchanged()
