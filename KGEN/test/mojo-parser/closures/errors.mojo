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
# RUN: %parse-mojo-isolated -verify-diagnostics %s


# expected-error @+1 {{the 'escaping' function effect is no longer supported}}
def escaping_effect_is_rejected(closure: def() escaping -> None):
    pass


def escaping_on_nested_decl_is_rejected():
    # expected-error @below {{the 'escaping' function effect is no longer supported}}
    def myclosure() escaping:
        pass


# COM: https://github.com/modular/mojo/issues/1223
# COM: When a runtime argument has incorrect type, nested function bodies may
# COM: still be resolved. Ensure that we don't crash when the arg is used.
struct Parametric[a: Int](Movable where False):
    pass


def test_suppressed_dyn_binding_error[
    x: Int
    # expected-error @below {{parametric functions must not be used as arguments; pass as a parameter instead}}
    # expected-note @below {{alternatively, bind its type parameters to create a concrete function}}
](pval: Parametric[x], func: def[y: Int](p: Parametric[y]) thin -> None):
    def nested():
        func(pval)
