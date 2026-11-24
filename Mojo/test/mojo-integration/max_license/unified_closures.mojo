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
# RUN: env MOJO_ENABLE_PARAMETRIC_CLOSURE_TRAIT=1 %mojo %s 3 4 2 4 1 3 hello | FileCheck %s
# RUN: env MOJO_ENABLE_PARAMETRIC_CLOSURE_TRAIT=1 kgen %s -elaborate -S -o - | FileCheck %s -check-prefix=CHECK-RP

# RUN: %mojo %s 3 4 2 4 1 3 hello | FileCheck %s
# RUN: kgen %s -elaborate -S -o - | FileCheck %s -check-prefix=CHECK-RP

# COM: Check that the argument is augmented at the definition site.

from std.sys import argv


def takeIt[T: def(sdfa: String) -> String, //](state: T, x: String):
    print(state(x))


def takeIt[T: def(asdf: Int) -> Int, //](state: T, x: Int):
    print(state(x))


def inLoops(list: List[Int], w: Int):
    for item in list:

        @always_inline
        def closure(z: Int) {imm item, var w} -> Int:
            var sum = item
            for j in range(w):
                sum = j + sum * sum
            return z + sum

        takeIt(closure, w)


@no_inline
def takeIt[func: def() -> Int](impl: func):
    print(impl())


def printItThrows[
    kernel: def(x: Int) raises -> Int
](func: kernel, y: Int) raises:
    print(func(y))


def aThingThrows(y: Int) raises:
    def myclosure(x: Int) raises {var} -> Int:
        if x == 3:
            raise Error("NOT THREE!")
        return y + x

    printItThrows(myclosure, y)


@no_inline
def captureParameters[XX: Int, YY: Int](x: Int):
    @no_inline
    def myclosure(z: Int) {var x} -> Int:
        return XX + YY + x

    takeIt(myclosure, x)


def thin[XX: Int, YY: Int]():
    @no_inline
    def myclosure(z: Int) {} -> Int:
        return XX + YY

    takeIt(myclosure, 0)


@no_inline
def takesThinPromotedParam[
    FuncType: def[U: AnyType, //](impl: U) thin -> Int
]():
    print(FuncType("here"))


def thinWithPromotedParam[XX: Int]():
    @no_inline
    def myclosure[U: AnyType, //](impl: U) {} -> Int:
        return XX

    takesThinPromotedParam[myclosure]()


@no_inline
def regPass[T: def(zz: Int) -> Int](impl: T, y: Int):
    print(impl(y))


def takeItMaybe[
    T: Copyable & def(x: Int) -> Int
](x: Int, state: Optional[T] = None):
    if case .Some(state) = state:
        print(state(x))
    else:
        print("none")


struct CheckForTrivialAliases[F: Copyable & def(z: Int) -> Int]:
    comptime DEL_IS_TRIVIAL = Self.F.__del__is_trivial
    comptime MOVE_IS_TRIVIAL = Self.F.__move_ctor_is_trivial
    comptime COPY_IS_TRIVIAL = Self.F.__copy_ctor_is_trivial


struct NonTrivialCopy(ImplicitlyCopyable, RegisterPassable):
    var x: Int

    def __init__(out self, y: Int):
        self.x = y

    def __init__(out self, *, copy: Self):
        self.x = copy.x + 1


trait HasString(Deinitable, ImplicitlyCopyable):
    def get(self) -> String:
        ...


@fieldwise_init
struct HasStringImpl(HasString):
    var s: String

    def get(self) -> String:
        return self.s


@no_inline
def takeGenericClosure[F: def(Int) -> Int](var impl: F) -> Int:
    return impl(0)


def captureGeneric[T: HasString](x: T) -> Int:
    var copy = x.copy()

    def closure(y: Int) {var^} -> Int:
        print(copy.get())
        return y + 1

    return takeGenericClosure(closure)


# ===----------------------------------------------------------------------=== #
# Foldable Closure Where Clauses
# ===----------------------------------------------------------------------=== #
trait Coord(Deinitable, ImplicitlyCopyable):
    def prettyPrint(self):
        pass


@fieldwise_init
struct Cartesian(Coord):
    var x: Int
    var y: Int

    def prettyPrint(self):
        print(self.x, self.y)


struct Sphere(Coord):
    var theta: Int
    var phi: Int


def hasWhereClause[T: Coord, C: def() -> T](impl: C):
    impl().prettyPrint()


def hasWhereClauseTraitUpcast[
    TT: ImplicitlyCopyable & Deinitable, C: def(x: TT) -> TT
](impl: C, x: TT):
    _ = impl(x)


def exerciseWhereClauseTraitUpcast[T: Coord](x: T):
    def stronger(y: T) {var} -> T:
        return x

    hasWhereClauseTraitUpcast[T, type_of(stronger)](stronger, x)


# ===----------------------------------------------------------------------=== #
# Parametric Ref Closures
# ===----------------------------------------------------------------------=== #


def take_parametric_ref(ref x: Int):
    def bar() {ref x}:
        comptime if origin_of(x).mut:
            print("mut")
        else:
            print("immut")

    bar()
    return


def take_immut_ref(imm x: Int):
    return take_parametric_ref(x)


# An explicit `imm` argument behaves the same as the explicit argument above.
def take_imm_ref(imm x: Int):
    return take_parametric_ref(x)


def take_mut_ref(mut x: Int):
    return take_parametric_ref(x)


def testMovedCaptureClosure():
    var list2 = [1, 2, 3]

    def test() {var list2^}:
        print(len(list2))
        for i in range(len(list2)):
            print(list2[i])

    test()


def lifetimeMarkersRemoved():
    var s = "Hello, world!"

    def test() {var s}:
        s = s + "!"
        print(s)

    test()


def main() raises:
    var str = argv()[1]
    var x = atol(str)
    var y = atol(argv()[2])
    var greeting = argv()[7]

    def captureMe() {var} -> Int:
        return y

    # COM: Nested capture instances are emitted correctly
    def aFinalThing(z: Int) {imm} -> Int:
        def aNestedThing(u: Int) {var} -> Int:
            return captureMe() + z

        # CHECK: 7
        takeIt(aNestedThing, captureMe())
        return x

    # CHECK: 3
    takeIt(aFinalThing, x)

    # COM: Check that non trivial copies of RP types are handled. `y` is 4;
    # capturing it by `var` runs the non-trivial copy ctor once (+1), so the
    # closure observes 5.
    var nonTrivialCopy = NonTrivialCopy(y)

    def captureRPNonTrivialCopy(z: Int) {var nonTrivialCopy} -> Int:
        return nonTrivialCopy.x

    # CHECK: 5
    takeIt(captureRPNonTrivialCopy, x)

    # COM: Check that nested closures that capture closures in parent are lifted properly
    def level00() {var} -> Int:
        return y

    def level01() {var level00, var y} -> Int:
        def level10() {var y} -> Int:
            return y

        def level11() {var level00, var level10} -> Int:
            def level20() {var level10} -> Int:
                return level10()

            # CHECK: 4
            takeIt(level20)
            return level00()

        # CHECK: 4
        takeIt(level11)
        return level00()

    # CHECK: 4
    takeIt(level01)

    var nontrivial = NonTrivialCopy(0)

    @no_inline
    def myclosure(z: Int) {var x, var y, var nontrivial} -> Int:
        _ = nontrivial.x
        return x + y + z

    # CHECK: False
    print(CheckForTrivialAliases[type_of(myclosure)].DEL_IS_TRIVIAL)

    # CHECK: False
    print(CheckForTrivialAliases[type_of(myclosure)].MOVE_IS_TRIVIAL)

    # CHECK: False
    print(CheckForTrivialAliases[type_of(myclosure)].COPY_IS_TRIVIAL)

    # CHECK: 10
    print(myclosure(x))

    # CHECK: 10
    captureParameters[3, 4](x)
    # CHECK: 17
    captureParameters[13, 1](x)

    # CHECK: 8
    thin[1, 7]()

    # COM: Promoted stateless closures can bind captured params and satisfy
    # COM: thin generic constraints.
    # CHECK: 4
    thinWithPromotedParam[4]()

    try:
        # CHECK: NOT THREE!
        aThingThrows(x)
    except e:
        print(e)

    # COM: Nested closures
    var two = atol(argv()[3])
    var four = atol(argv()[4])
    var one = atol(argv()[5])

    def twoPlusOnePlusFour(z: Int) {var} -> Int:
        var three = z + two
        var str = String(three)

        def child(s: String) {mut str} -> String:
            str += s
            return str

        # CHECK: 3s
        takeIt(child, "s")

        # COM: Inner closure capturing a local declared in a nested scope of
        # COM: this enclosing closure (MOCO-4652).
        if z:
            var kg = three

            def nested_scope_child(s: String) {imm kg} -> String:
                return String(kg) + s

            # CHECK: 3n
            takeIt(nested_scope_child, "n")

        return two + four + z

    # CHECK: 7
    takeIt(twoPlusOnePlusFour, one)
    # COM: Ensure closures nested in lexical scopes work.
    if one == 1:
        var three = one + two
        var str = String(three)

        def child(s: String) {mut str} -> String:
            str += s
            return str

        # CHECK: 3s
        takeIt(child, "s")

    # CHECK-RP: kgen.func @"{{.*}}my_rp_closure{{.*}}"(%arg0: !kgen.struct<(scalar<index>, scalar<index>)>
    @no_inline
    def my_rp_closure(zz: Int) {var} -> Int:
        return x + y + zz

    # CHECK: 10
    regPass(my_rp_closure, x)

    # CHECK: 10
    takeItMaybe(x, my_rp_closure)

    # CHECK: none
    takeItMaybe[type_of(my_rp_closure)](x)

    # COM: parent scopes are searched.
    if x > 1:
        var three = atol(argv()[6])
        if three > 2:

            def aThing(z: Int) {var x} -> Int:
                return z + x

            # CHECK: 7
            takeIt(aThing, four)

    # COM: invoking parametric closures
    def paramclosure[a: Int](b: Int) {var x} -> Int:
        return a + b + x

    # CHECK: 8
    print(paramclosure[2](3))

    # COM: Lexical scopes are preserved
    var list = List[Int]()
    list.append(two)
    list.append(four)
    # CHECK: 5
    # CHECK: 17
    inLoops(list, one)

    # COM: Where clauses over closures are foldable
    var coordinate = Cartesian(two, four)

    def givesCart() {var} -> Cartesian:
        return coordinate

    # COM: Trait upcasts in closure alias witnesses are foldable.
    exerciseWhereClauseTraitUpcast(coordinate)

    # CHECK: 2 4
    hasWhereClause[Cartesian, type_of(givesCart)](givesCart)

    # COM: Generic captures preserve copy/move/del information.
    var copiedString = HasStringImpl(greeting)
    # CHECK: hello
    _ = captureGeneric(copiedString)

    var some_thing_to_ref = 1
    # CHECK: immut
    take_immut_ref(some_thing_to_ref)
    # CHECK: immut
    take_imm_ref(some_thing_to_ref)
    # CHECK: mut
    take_mut_ref(some_thing_to_ref)

    # COM: Closures consume the moved captures
    # CHECK: 3
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    testMovedCaptureClosure()

    # CHECK: Hello, world!!
    lifetimeMarkersRemoved()
