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

# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s


def make_closure(x: Int):
    # CHECK: Could not infer capture convention of the captured value x
    def my_closure(y: Int) -> Int:
        return x + y


# // -----

# COM: Verify Capture Rules Are Enforced

def immByMut(byRefMut: String):
    # CHECK: error: Cannot capture byRefMut by mut because it could be immutable
    def myclosure() {mut byRefMut}:
        pass


struct DoNotMoveMe(Movable where False):
    pass

def notMovable(var byMove: DoNotMoveMe):
    # CHECK: error: Cannot capture byMove by move because the type is not movable
    def myclosure() {var byMove^}:
        pass

struct MoveMe(Movable where False):
    pass

def immByMov(var byMove: MoveMe):
    # CHECK: error: Cannot capture byMove by move because the type is not movable
    def myclosure() {var byMove^}:
        pass

def paramNotAllowed[X: Int, Y: Int]():
    # CHECK: error: value X is a parameter and does not need a capture convention
    # CHECK: error: value Y is a parameter and does not need a capture convention
    def myclosure() {var X, read Y}:
        pass


def doesNotExist():
    # CHECK: error: reference to an unknown value: What
    def myclosure() {var What}:
        pass


# // -----


def mutateMe(mut str: String):
    pass


def illegal(mut byRefMut: String):
    def myclosure() {read byRefMut}:
        # CHECK: invalid call to 'mutateMe': value passed to mutable argument 'str' must be mutable
        mutateMe(byRefMut)


# // -----


def toy(rogue: String):
    # CHECK: error: Could not infer capture convention of the captured value rogue
    def myclosure() {} -> String:
        return rogue
