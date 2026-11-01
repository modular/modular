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

from std.collections import OptionalReg

from .shape import Shape

# `Shape` is only reachable through the function-type bound on `FnType`, which
# is stored as a parameter of the universal closure trait.
comptime _HookFnType = def[FnType: ImplicitlyCopyable & def(Shape) -> None](
    func: FnType
) thin


trait Hooks:
    comptime hook_fn: OptionalReg[_HookFnType] = None


struct DefaultHooks(Hooks):
    pass


def use_hooks():
    _ = Bool(DefaultHooks.hook_fn)
