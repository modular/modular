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
"""AMDGCN lowerings for the `SIMD.cast` conversions `HIPPlugin` overrides."""

from std.collections import OptionalReg
from std.memory import bitcast
from std.simd import _CastPluginHookFnType, _simd_apply
from std.sys.info import CompilationTarget


comptime _CastFn[dst: DType] = OptionalReg[_CastPluginHookFnType[dst]]


def _amdgcn_native_float8_cast[
    src: DType, dst: DType, target: CompilationTarget
]() -> Bool:
    """Returns whether `target` converts between `src` and `dst` natively:
    float32 to and from the fnuz float8 types on MI300X (gfx942), and the
    e4m3fn and e5m2 types on CDNA4 and newer (gfx950, gfx1250)."""
    comptime fnuz = (DType.float8_e4m3fnuz, DType.float8_e5m2fnuz)
    comptime ocp = (DType.float8_e4m3fn, DType.float8_e5m2)
    comptime if (src in fnuz and dst == DType.float32) or (
        src == DType.float32 and dst in fnuz
    ):
        return target.is_amd_gpu["gfx942"]()
    elif (src in ocp and dst == DType.float32) or (
        src == DType.float32 and dst in ocp
    ):
        return target.is_amd_gpu["gfx950"]() or target.is_amd_gpu["gfx1250"]()
    return False


def _amdgcn_cast_fn[src: DType, dst: DType]() -> _CastFn[dst]:
    """Returns the AMDGCN lowering of a cast from `src` to `dst`, or `None` to
    keep the stdlib's own conversion."""
    # A cast to bool is a comparison with zero, which bfloat16 handles.
    comptime if src == DType.bfloat16 and dst != DType.bool:
        return comptime (_CastFn[dst](_from_bfloat16[dst]))
    return None


@inline(.always)
def _from_bfloat16[
    dtype: DType, width: SIMDLength, //, dst: DType
](x: SIMD[dtype, width]) -> SIMD[dst, width]:
    """Converts bfloat16 to `dst` by widening it to float32 first."""

    @inline(.always)
    def to_float32[
        input_dtype: DType, result_dtype: DType
    ](val: Scalar[input_dtype]) -> Scalar[result_dtype]:
        # bfloat16 is the high half of a float32 with the same value.
        return bitcast[result_dtype, 1](SIMD[input_dtype, 2](0, val))

    return _simd_apply[to_float32, result_dtype=DType.float32](x).cast[dst]()
