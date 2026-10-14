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


struct TString(Stringable):
    def __init__(out self):
        pass

    def __str__(self) -> String:
        return {}


struct _FormatArgument[origin: ImmOrigin]:
    def __init__[T: AnyType](out self, ref[Self.origin] writable: T):
        pass


def __make_tstring[
    format_string: __mlir_type.`!kgen.string`,
    origins: ImmOrigin,
](ref array: Array[_FormatArgument[origins], _]) -> TString:
    return {}
