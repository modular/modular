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

# Reading a field of a field of a value whose type is `A if c else B` inside a
# `comptime if c:` branch must keep the lifetime checker working: the nested
# read goes through the refined value.
#
# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics


struct Box(Movable):
    var v: Int

    def __init__(out self, v: Int):
        self.v = v

    def get(self) raises -> Int:
        return self.v


struct Ctx[t: Int](Movable):
    var context: Box

    def __init__(out self, v: Int):
        self.context = Box(v)


struct A[d: Int](Movable):
    var ctx: Ctx[0]
    var inner: Box

    def __init__(out self, v: Int):
        self.ctx = Ctx[0](v)
        self.inner = Box(v)


struct B[d: Int](Movable):
    var x: Int

    def __init__(out self, v: Int):
        self.x = v


def use_a() -> Bool:
    return True


struct Wrapper[d: Int]:
    comptime _Impl = A[Self.d] if use_a() else B[Self.d]
    var _impl: Self._Impl

    def __init__(out self, var impl: Self._Impl):
        self._impl = impl^

    def borrowed(self) raises -> Int:
        comptime if use_a():
            return self._impl.ctx.context.get() + self._impl.inner.get()
        else:
            return self._impl.x

    def owned(var self) raises -> Int:
        comptime if use_a():
            return self._impl.ctx.context.get() + self._impl.inner.get()
        else:
            return self._impl.x

    def owned_try(var self) -> Int:
        comptime if use_a():
            try:
                var x = self._impl.ctx.context.get()
                return x + self._impl.inner.get()
            except:
                return 0
        else:
            return self._impl.x


def main():
    pass
