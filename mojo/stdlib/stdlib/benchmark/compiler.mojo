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

from sys._assembly import inlined_assembly

# ===-----------------------------------------------------------------------===#
# keep
# ===-----------------------------------------------------------------------===#


@always_inline
fn keep[dtype: AnyTrivialRegType](val: dtype):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
      dtype: The type of the input.

    Args:
      val: The value to not optimize away.
    """
    var tmp = val
    var tmp_ptr = UnsafePointer(to=tmp)
    inlined_assembly[
        "",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](tmp_ptr)
