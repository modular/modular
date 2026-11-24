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

# Legacy closure syntax is rejected with a pointer to capture lists.


# expected-error @+1 {{the 'capturing' function effect is no longer supported; use a closure capture list like '{var}' or '{mut x}' instead}}
def takes_capturing[f: def() capturing -> Int]():
    pass


def parameter_decorator():
    # expected-error @+1 {{'@__parameter' closures are no longer supported; remove the decorator and declare the captures in a capture list like '{var}' or '{mut x}'}}
    @__parameter
    def f() -> Int:
        return 1


def deprecated_parameter_decorator():
    # expected-error @+1 {{'@parameter' closures are no longer supported; remove the decorator and declare the captures in a capture list like '{var}' or '{mut x}'}}
    @parameter
    def f() -> Int:
        return 1


def copy_capture(x: Int):
    # expected-error @+1 {{'@__copy_capture' is no longer supported; list the captures in a capture list like '{var x}' instead}}
    @__copy_capture(x)
    def f() {imm} -> Int:
        return 1


def move_capture(var x: String):
    # expected-error @+1 {{'@__move_capture' is no longer supported; list the captures in a capture list like '{var x}' instead}}
    @__move_capture(x)
    def f() {imm} -> Int:
        return 1
