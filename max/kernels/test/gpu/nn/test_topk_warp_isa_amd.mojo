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

"""Pins the gfx950 shape of `_topk_warp`, which no numerical test can see.

`_topk_warp` is worth having over `_topk_stage1` + `_topk_stage2` only because
of how it spends its k passes: one *warp* reduction and no barrier per
extracted element, against the two-stage path's two *block* reductions and
five barriers. Both produce the same answer, so swapping `_warp_reduce_topk`
back for `_block_reduce_topk` -- or adding a barrier to the extraction loop --
would pass every correctness test in this directory while giving the whole
optimization back.

Measured on gfx950 at K3's router instantiation (fp32, int64 indices,
`largest=True`), per extracted element:

    _topk_stage1 phase 2/3   24 ds_bpermute   2 s_barrier
    _topk_stage2             24 ds_bpermute   3 s_barrier
    _topk_warp               12 ds_bpermute   0 s_barrier

The bounds below are those counts with a little slack, not aspirations.

The spill assertions record a refuted hypothesis as much as they guard one.
None of these kernels spills on gfx950 even though `.max_flat_workgroup_size`
defaults to 1024 -- which budgets 128 VGPRs per thread -- because their live
state is small (57, 28 and 20 VGPRs respectively). So none of them wants a
`MAX_THREADS_PER_BLOCK_METADATA` annotation, and the two-stage kernels could
not safely take one anyway: `block_size` reaches them as a runtime argument
with no enforced bound. If a future change grows the per-thread state past the
budget, this test is what reports it.

AMD-only: `.vgpr_spill_count` and `ds_bpermute` are AMDGPU spellings, and the
128-VGPR budget this guards against is an AMDGPU backend behavior. NVIDIA's
ptxas budgets registers independently of the declared block size, so the
equivalent guard there would assert different things.
"""

from max.gpu.host.compile import _compile_code
from nn.topk import _topk_stage1, _topk_stage2, _topk_warp
from std._gpu.host.info import get_gpu_target
from std.testing import assert_equal, assert_true

comptime GFX950 = get_gpu_target["mi355x"]()

# One `_warp_reduce_topk` over a 64-lane wavefront: 6 butterfly steps, each
# shuffling the value and the index. A block reduction would be twice this.
comptime MAX_BPERMUTE = 12

# Staging the row into shared memory needs one barrier. The extraction loop
# needs none, because each lane owns its own slice of the row outright.
comptime MAX_BARRIER = 1


def assert_no_scratch(asm: String, name: String) raises:
    assert_true(
        ".vgpr_spill_count: 0" in asm,
        String(name, " spills VGPRs to scratch on gfx950"),
    )
    assert_true(
        ".private_segment_fixed_size: 0" in asm,
        String(name, " reserved a private (scratch) segment on gfx950"),
    )
    assert_equal(asm.count("scratch_store"), 0, String(name, " scratch_store"))
    assert_equal(asm.count("scratch_load"), 0, String(name, " scratch_load"))


def test_warp_kernel_stays_warp_scoped() raises:
    var asm = String(
        _compile_code[
            _topk_warp[DType.float32, DType.int64, True],
            emission_kind="asm",
            target=GFX950,
        ]()
    )
    assert_no_scratch(asm, "_topk_warp")

    var bpermute = asm.count("ds_bpermute")
    assert_true(
        bpermute <= MAX_BPERMUTE,
        String(
            "_topk_warp emits ",
            bpermute,
            " ds_bpermute, over the ",
            MAX_BPERMUTE,
            (
                " of a single warp reduction -- has the extraction loop grown a"
                " block-wide reduction?"
            ),
        ),
    )

    var barriers = asm.count("s_barrier")
    assert_true(
        barriers <= MAX_BARRIER,
        String(
            "_topk_warp emits ",
            barriers,
            " s_barrier, over the ",
            MAX_BARRIER,
            (
                " the shared-memory staging needs -- a barrier in the"
                " extraction loop costs one block-wide sync per selected"
                " element"
            ),
        ),
    )


def test_two_stage_kernels_do_not_spill() raises:
    assert_no_scratch(
        String(
            _compile_code[
                _topk_stage1[DType.float32, DType.int64, True],
                emission_kind="asm",
                target=GFX950,
            ]()
        ),
        "_topk_stage1",
    )
    assert_no_scratch(
        String(
            _compile_code[
                _topk_stage2[DType.float32, DType.int64, False, True],
                emission_kind="asm",
                target=GFX950,
            ]()
        ),
        "_topk_stage2[sampling=False]",
    )
    assert_no_scratch(
        String(
            _compile_code[
                _topk_stage2[DType.float32, DType.int64, True, True],
                emission_kind="asm",
                target=GFX950,
            ]()
        ),
        "_topk_stage2[sampling=True]",
    )


def main() raises:
    test_warp_kernel_stays_warp_scoped()
    test_two_stage_kernels_do_not_spill()
