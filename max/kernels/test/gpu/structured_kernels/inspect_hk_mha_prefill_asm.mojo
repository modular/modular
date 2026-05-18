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
"""Compile HKMhaPrefill to GCN asm and dump for register/spill analysis."""

from std.gpu import MAX_THREADS_PER_BLOCK_METADATA
from std.gpu.host.compile import _compile_code
from std.gpu.host import get_gpu_target
from std.utils import StaticTuple

from layout.coord import Coord, Idx, RuntimeInt
from layout.tile_layout import row_major

from nn.attention.gpu.amd_structured.hk_mha_prefill import (
    HKMhaConfig,
    HKMhaPrefill,
)


comptime MI355X_TARGET = get_gpu_target["mi355x"]()

# Bench config: 16 heads, d=128, seq=8192. seq_len / num_keys are
# runtime in production, so the asm dump uses `RuntimeInt` for those
# layout dims to match the codegen the bench actually sees.
comptime NUM_HEADS = 16
comptime NUM_KV_HEADS = 16
comptime DEPTH = 128

comptime CONFIG = HKMhaConfig(
    q_block_size=32,
    kv_block=64,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    num_warps=8,
    causal=True,
)


def main() raises:
    print("=== HKMhaPrefill.run GCN asm dump (MI355X, bench config) ===")
    comptime q_layout = row_major(
        Coord(
            RuntimeInt[DType.int32](Int32(1)),  # batch (placeholder)
            RuntimeInt[DType.int32](Int32(8192)),  # seq_len (placeholder)
            Idx[NUM_HEADS](),
            Idx[DEPTH](),
        )
    )
    comptime kv_layout = row_major(
        Coord(
            RuntimeInt[DType.int32](Int32(1)),
            RuntimeInt[DType.int32](Int32(8192)),
            Idx[NUM_KV_HEADS](),
            Idx[DEPTH](),
        )
    )
    comptime l_vec_layout = row_major(
        Coord(
            RuntimeInt[DType.int32](Int32(1)),
            Idx[NUM_HEADS](),
            RuntimeInt[DType.int32](Int32(8192)),
        )
    )
    comptime kernel = HKMhaPrefill[CONFIG].run[
        type_of(q_layout),
        type_of(kv_layout),
        type_of(kv_layout),
        type_of(q_layout),
        type_of(l_vec_layout),
    ]
    comptime _PREFILL_IGLP_OPTS: StaticString = (
        "amdgpu-igrouplp-exact-solver=true,"
        "amdgpu-igrouplp-exact-solver-max-branches=10000,"
        "amdgpu-igrouplp-exact-solver-cost-heur=false"
    )
    var asm = _compile_code[
        kernel,
        target=MI355X_TARGET,
        emission_kind="asm",
        compile_options=_PREFILL_IGLP_OPTS,
    ]()
    print(asm.asm)
