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
# RUN: kgen %s -elaborate -verify-diagnostics %s

from std.collections.string.string_span import StaticString, _get_kgen_string


# expected-warning @+4 {{duplicate LLVM metadata attribute for "rocdl.waves_per_eu". Value of the last occurrence will be used}}
@__llvm_metadata(`rocdl.waves_per_eu`=SIMDLength(3))
@__llvm_metadata(`rocdl.flat_work_group_size`=SIMDLength(128))
@__llvm_metadata(`rocdl.waves_per_eu`=SIMDLength(4))
def my_kernel_multiple():
    pass


def alias_parametric_fn() -> String:
    comptime if False:
        return "nvvm.maxntid"
    else:
        return "rocdl.flat_work_group_size"


comptime mname = _get_kgen_string[alias_parametric_fn()]()


# expected-warning @+4 {{duplicate LLVM metadata attribute for "rocdl.flat_work_group_size" : !kgen.string. Value of the last occurrence will be used}}
@__llvm_metadata(mname=SIMDLength(3))
@__llvm_metadata(`rocdl.waves_per_eu`=SIMDLength(128))
@__llvm_metadata(mname=SIMDLength(4))
def my_kernel_multiple_alias():
    pass


@export
def top() abi("Mojo"):
    my_kernel_multiple()
    my_kernel_multiple_alias()
