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

"""Config and routing checks for the fused non-dividing-group WS BM=32 route.

Target hardware family: NVIDIA SM100 (B200).

Groups 9 and 6 do not divide the 32-row MMA tile. `fuse_nondividing` fuses such
a group into one CTA, which shrinks `BM_eff()` from 32 to 3 (group 9) or 5
(group 6). The dispatch therefore instantiates a fused and an unfused BM=32
config and picks per launch from `max_prompt_len`:

- `max_prompt_len <= fused BM_eff`   -> fused BM=32 (Layout-G)
- `max_prompt_len <= 32`             -> unfused BM=32 (Layout-G)
- otherwise                          -> Layout-E / 1Q-2Q carve

Fusing unconditionally (the first version of this route) sent the middle band
to Layout-E and measured 8-106% slower there. These checks pin the three-way
choice and the config plumbing it relies on. `with_splitk` rebuilds the config
through the constructor, so a dropped `fuse_nondividing` would run an unfused
config under a grid sized for the fused one.

Host-only: no kernel is launched. The kernel-level boundary numerics live in
`test_mha_sm100_ws_bm32_fuse_gqa_tail_rows.mojo`.
"""

from std.testing import assert_equal, assert_false, assert_true

from max.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.attention.gpu.nvidia.sm100.attention import FA4Config
from nn.attention.gpu.nvidia.sm100.dispatch import (
    WS_G_FUSED,
    WS_G_NONE,
    WS_G_UNFUSED,
    _ws_g_route,
)

comptime _NUM_KV_HEADS = 8
comptime _PAGE_SIZE = 128
comptime _SPLITK_P = [1, 2, 4, 10]


def _base_config[group: Int, depth: Int = 128]() -> FA4Config[DType.bfloat16]:
    return FA4Config[DType.bfloat16](
        num_q_heads=group * _NUM_KV_HEADS,
        group=group,
        qk_depth=depth,
        ov_depth=depth,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        page_size=_PAGE_SIZE,
        is_mla=False,
    )


def _check_configs[group: Int, fused_bm_eff: Int]() raises:
    comptime base = _base_config[group]()
    comptime unfused = base.with_bm(32)
    comptime fused = base.with_bm(32, fuse_nondividing=True)

    assert_false(unfused.fuse_gqa)
    assert_false(unfused.fuse_nondividing)
    assert_equal(unfused.BM_eff(), 32)

    assert_true(fused.fuse_gqa)
    assert_true(fused.fuse_nondividing)
    assert_equal(fused.BM_eff(), fused_bm_eff)
    assert_equal(fused.gqa_pad_rows(), 32 % group)
    assert_true(fused.supported())
    assert_true(unfused.supported())

    # The knob must survive `with_splitk`: the dispatch builds the split-K
    # kernel from it, and a dropped flag changes BM_eff() and so the grid.
    comptime for p in _SPLITK_P:
        comptime fused_sk = fused.with_splitk(p)
        comptime unfused_sk = unfused.with_splitk(p)
        assert_true(fused_sk.fuse_nondividing)
        assert_true(fused_sk.fuse_gqa)
        assert_equal(fused_sk.BM_eff(), fused.BM_eff())
        assert_equal(fused_sk.BM, fused.BM)
        assert_equal(fused_sk.BN, fused.BN)
        assert_equal(fused_sk.splitk_partitions, p)
        assert_false(unfused_sk.fuse_nondividing)
        assert_false(unfused_sk.fuse_gqa)
        assert_equal(unfused_sk.BM_eff(), unfused.BM_eff())
        assert_equal(unfused_sk.splitk_partitions, p)


def _check_route[group: Int, fused_bm_eff: Int]() raises:
    comptime base = _base_config[group]()
    comptime unfused = base.with_bm(32)
    comptime fused = base.with_bm(32, fuse_nondividing=True)
    comptime layout_e = base.with_bm(64)

    for prompt in range(1, 129):
        var route = _ws_g_route(
            fused.fuse_nondividing,
            fused.BM_eff(),
            unfused.BM_eff(),
            prompt,
        )
        if prompt <= fused_bm_eff:
            assert_equal(route, WS_G_FUSED, "prompt " + String(prompt))
        elif prompt <= 32:
            assert_equal(route, WS_G_UNFUSED, "prompt " + String(prompt))
        else:
            assert_equal(route, WS_G_NONE, "prompt " + String(prompt))
            # Layout-E (BM=64, unfused for a non-dividing group) takes the
            # next band; past it the 1Q/2Q carve does.
            assert_equal(layout_e.BM_eff(), 64)


def _check_not_requested_or_not_eligible() raises:
    # A group that divides MMA_M fuses without the knob, and the knob does not
    # change it; the fused and unfused configs are then the same config, so the
    # fused route is never offered.
    comptime g4 = _base_config[4]()
    comptime g4_plain = g4.with_bm(32)
    comptime g4_knob = g4.with_bm(32, fuse_nondividing=True)
    assert_true(g4_plain.fuse_gqa)
    assert_false(g4_knob.fuse_nondividing)
    assert_equal(g4_plain.BM_eff(), g4_knob.BM_eff())
    assert_equal(g4_plain.BM_eff(), 8)
    for prompt in range(1, 65):
        assert_equal(
            _ws_g_route(
                g4_knob.fuse_nondividing,
                g4_knob.BM_eff(),
                g4_plain.BM_eff(),
                prompt,
            ),
            WS_G_UNFUSED if prompt <= 8 else WS_G_NONE,
        )

    # Only the validated (BM=32, depth 128, group 9/6) set honors the request.
    comptime g9 = _base_config[9]()
    assert_false(g9.with_bm(32).fuse_nondividing)
    assert_false(g9.with_bm(64, fuse_nondividing=True).fuse_nondividing)
    assert_false(g9.with_bm(64, fuse_nondividing=True).fuse_gqa)
    comptime g9_d64 = _base_config[9, 64]()
    assert_false(g9_d64.with_bm(32, fuse_nondividing=True).fuse_nondividing)
    comptime g5 = _base_config[5]()
    assert_false(g5.with_bm(32, fuse_nondividing=True).fuse_nondividing)
    assert_false(g5.with_bm(32, fuse_nondividing=True).fuse_gqa)


def main() raises:
    # group 9: BM_eff = 32 // 9 = 3. group 6: BM_eff = 32 // 6 = 5.
    _check_configs[9, 3]()
    _check_configs[6, 5]()
    _check_route[9, 3]()
    _check_route[6, 5]()
    _check_not_requested_or_not_eligible()
    print("test_mha_sm100_ws_bm32_fuse_nondividing_route: PASSED")
