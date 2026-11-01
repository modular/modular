# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import extensibility
from extensibility import OutputTensor
from std.collections.string.string_span import _get_kgen_string
from std.collections.string import StaticString


# Custom version of constrained so we get the note in this file.
@inline(.nodebug)
def constrained[cond: Bool, msg: StaticString]():
    __mlir_op.`kgen.param.assert`[
        cond=cond.__mlir_bool__(),
        message=_get_kgen_string[msg](),
    ]()  # expected-note {{constraint failed}}


@extensibility.register("fails_to_elaborate")
struct InvalidOwnedArgConvention:
    @staticmethod
    def execute[
        target: StaticString, dtype: DType, rank: Int
    ](output: OutputTensor[dtype=dtype, rank=rank, ...]):
        comptime assert (
            False
        ), "oops"  # expected-error {{call expansion failed}}
