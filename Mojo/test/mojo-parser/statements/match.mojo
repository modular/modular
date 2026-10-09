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

# RUN: %parse-mojo-isolated %s -verify-diagnostics | FileCheck %s
# RUN: %parse-mojo-isolated %s -verify-diagnostics | kgen-opt -verify-parameters -lower-semantic-cf -check-lifetimes -verify-parameters

# A function we can call with minimal IR gruff but still verify the right
# code is put out in the right place.
def case_callee[p: Int](): pass

struct InitPattern(ImplicitlyCopyable):
    var value: Int

    def __init__(out self):
        self.value = 0

    def __init__(out self, value: Int, *, offset: Int = 0):
        self.value = value + offset

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value


# CHECK-LABEL: lit.fn @"match_trivial
def match_trivial(i: Int):
    # Irrefutable `case _` is not a match: the body is emitted inline.
    # CHECK-NEXT:    lit.call {{.*}}@"case_callee{{.*}}<index> 0
    # CHECK-NOT:     hlcf.match
    __match i:
    case _:
        case_callee[0]()

    # Guarded `case _` is an `hlcf.if` on the guard; failure falls through.
    # CHECK:       [[ZERO:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
    # CHECK:       [[GT:%.*]] = lit.call {{.*}}@"__gt__({{.*}}(%i, [[ZERO]])
    # CHECK:       [[B:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[GT]])
    # CHECK:       hlcf.if [[B]] {
    # CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
    # CHECK:         hlcf.yield
    # CHECK:       } else {
    # CHECK:         hlcf.yield
    # CHECK:       }
    __match i:
    case _ if i > 0:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_bindings
def match_bindings(i: Int, var s: String):
    # Bare bindings are "imm" bindings: register values become `bound`,
    # memory values become an immutable `ref` (muttoimm).
    # CHECK:       [[BX:%.*]] = lit.var.decl "x" bound
    # CHECK:       lit.ref.store %i, [[BX]]
    # CHECK:       lit.call {{.*}}@"__add__(
    __match i:
    case x:
        _ = x+1

    # CHECK:       [[SX:%.*]] = lit.var.decl "x" ref
    # CHECK:       lit.ref.store {{.*}}, [[SX]]
    # CHECK:       lit.ref.load [[SX]]
    # CHECK:       lit.call {{.*}}@"__len__(::String)"{{.*}}muttoimm
    __match s:
    case x:
        _ = x.__len__()

    # 'var' bindings make a copy.
    # CHECK:       [[VX:%.*]] = lit.var.decl "x" var
    # CHECK:       lit.ref.store %i, [[VX]]
    # CHECK:       lit.call {{.*}}@"__iadd__({{.*}}([[VX]],
    __match i:
    case var x:
        x += 1

    # CHECK:       [[VS:%.*]] = lit.var.decl "x" var
    # CHECK:       lit.call {{.*}}@"__init__(copy:::String)"{{.*}}[[VS]]
    # CHECK:       lit.call {{.*}}@"__iadd__{{.*}}([[VS]],
    __match s:
    case var x:
        x += "x"

    # Ref bindings work with mutable references to the original.
    # CHECK:       [[RX:%.*]] = lit.var.decl "x" ref
    # CHECK:       lit.ref.store %s, [[RX]]
    # CHECK:       lit.call {{.*}}@"__iadd__{{.*}}[mut *"s`
    __match s:
    case ref x:
        x += "x"


# CHECK-LABEL: lit.fn @"match_same_indent
# `0` and `1` share one equality test; trailing `_` is the exclusive complement
# (elif else), so no outer `hlcf.match` is needed.
# CHECK-NEXT:    [[L0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:    [[EQ0:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L0]])
# CHECK-NEXT:    [[B0:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ0]])
# CHECK-NEXT:    hlcf.if [[B0]] {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      [[L1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
# CHECK-NEXT:      [[EQ1:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L1]])
# CHECK-NEXT:      [[B1:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ1]])
# CHECK-NEXT:      hlcf.if.elifcond.yield [[B1]]
# CHECK-NEXT:    } then {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 2
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    }
def match_same_indent(x: Int):
    __match x:
    case 0:
        case_callee[0]()
    case 1:
        case_callee[1]()
    case _:
        case_callee[2]()


# CHECK-LABEL: lit.fn @"match_indented_cases
# CHECK:       kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK:       lit.call {{.*}}@"__eq__(
# CHECK:       lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.if %{{.*}} {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         hlcf.yield
# CHECK:       }
def match_indented_cases(x: Int):
    __match x:
        case 0:
            case_callee[0]()
        case _:
            case_callee[1]()


# CHECK-LABEL: lit.fn @"match_tuple_subject
# CHECK:       hlcf.match {
# CHECK:         lit.call {{.*}}@"__getitem_param__
# CHECK:         lit.ref.load
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.if %{{.*}} {
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         hlcf.match.next
# CHECK:       }
# CHECK:         lit.call {{.*}}@"__getitem_param__
# CHECK:         lit.ref.load
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.if %{{.*}} {
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         hlcf.match.next
# CHECK:       }
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.match.complete
def match_tuple_subject(point: Tuple[Int, Int]):
    __match point:
    case (0, 0):
        case_callee[0]()
    case _:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_initializer_list
# CHECK:       lit.call {{.*}}@InitPattern::@"__init__()"
# CHECK:       lit.call {{.*}}@"__eq__({{.*}}InitPattern
# CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:       lit.call {{.*}}@InitPattern::@"__init__(
# CHECK:       lit.call {{.*}}@"__eq__({{.*}}InitPattern
# CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 1
def match_initializer_list(value: InitPattern):
    __match value:
    case {}:
        case_callee[0]()
    case {1, offset=2}:
        case_callee[1]()


# Unparenthesized commas form tuples in both the subject and case patterns.
# CHECK-LABEL: lit.fn @"match_comma_subject
# CHECK:       lit.ref.pack.create
# CHECK:       lit.call {{.*}}@"__init__[LITImmOrigin,::Origin[False, $2]](*$0)"
# CHECK:       hlcf.match {
# CHECK:         lit.call {{.*}}@"__getitem_param__
# CHECK:         lit.call {{.*}}@"__eq__(::Bool,::Bool)"
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 2
def match_comma_subject(a: Bool, b: Bool):
    __match a, b:
    case True, True:
        case_callee[0]()
    case False, True:
        case_callee[1]()
    case _, _:
        case_callee[2]()


# CHECK-LABEL: lit.fn @"match_with_guard
# CHECK-NEXT:    hlcf.match {
# CHECK-NEXT:      [[Z0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:      [[NE0:%.*]] = lit.call {{.*}}@"__ne__({{.*}}(%c, [[Z0]])
# CHECK-NEXT:      [[B0:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[NE0]])
# CHECK-NEXT:      hlcf.if [[B0]] {
# CHECK-NEXT:        hlcf.yield
# CHECK-NEXT:      } else {
# CHECK-NEXT:        hlcf.match.next
# CHECK-NEXT:      }
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK-NEXT:      hlcf.match.complete
# CHECK-NEXT:    }
# CHECK-NEXT:    case {
# CHECK-NEXT:      [[L1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:      [[EQ1:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L1]])
# CHECK-NEXT:      [[B1:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ1]])
# CHECK-NEXT:      hlcf.if [[B1]] {
# CHECK-NEXT:        hlcf.yield
# CHECK-NEXT:      } else {
# CHECK-NEXT:        hlcf.match.next
# CHECK-NEXT:      }
# CHECK-NEXT:      [[Z1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:      [[NE1:%.*]] = lit.call {{.*}}@"__ne__({{.*}}(%c, [[Z1]])
# CHECK-NEXT:      [[GB1:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[NE1]])
# CHECK-NEXT:      hlcf.if [[GB1]] {
# CHECK-NEXT:        hlcf.yield
# CHECK-NEXT:      } else {
# CHECK-NEXT:        hlcf.match.next
# CHECK-NEXT:      }
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK-NEXT:      hlcf.match.complete
# CHECK-NEXT:    } else {
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    }
def match_with_guard(x: Int, c: Int):
    __match x:
    case _ if c != 0:
        case_callee[0]()
    case 0 if c != 0:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_case_body
# CHECK:       kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK:       lit.call {{.*}}@"__eq__(
# CHECK:       lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.if %{{.*}} {
# CHECK:         %inside_case = lit.var.decl "inside_case"
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       }
def match_case_body(x: Int):
    __match x:
    case 0:
        var inside_case: Int
    case _:
        case_callee[0]()


# CHECK-LABEL: lit.fn @"match_float
# CHECK:       lit.call {{.*}}@"__eq__(
# CHECK:       lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.if %{{.*}} {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         hlcf.yield
# CHECK:       }
def match_float(x: Float64):
    __match x:
    case 0.0:
        case_callee[0]()
    case _:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_string
# CHECK:       lit.call {{.*}}@"__eq__(
# CHECK:       lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.if %{{.*}} {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         hlcf.yield
# CHECK:       }
def match_string(x: String):
    __match x:
    case "a":
        case_callee[0]()
    case _:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_bool
# `False` then `True` (sorted by spelling); trailing `_` is unreachable once
# both Bool constructors are covered, so the elif else is `hlcf.unreachable`.
# CHECK:       lit.call {{.*}}@"__eq__(
# CHECK:       lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.if %{{.*}} {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.if.elifcond.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         hlcf.unreachable
# CHECK:       }
def match_bool(x: Bool):
    __match x:
    case True:
        case_callee[0]()
    case False:
        case_callee[1]()
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        case_callee[2]()


# EnumLike Color with inferred-member patterns (e.g. `case .red`).
struct Color(ImplicitlyCopyable, EnumLike):
    comptime _enum_case_names = ParameterList.of[
        "red".value, "green".value, "blue".value
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, NoneType, NoneType
    ].values

    def __init__(out self):
        pass

    def _get_enum_discriminant(self) -> Int:
        return 0

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Self._enum_case_types]()[id]:
        # No payload. The body only exists so the trait method is implemented.
        while True:
            pass


struct OpenColor(ImplicitlyCopyable, EnumLike):
    comptime _enum_case_names = ParameterList.of[
        "red".value, "green".value
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, NoneType
    ].values
    comptime _enum_is_exhaustive = False

    def __init__(out self):
        pass

    def _get_enum_discriminant(self) -> Int:
        return 0

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Self._enum_case_types]()[id]:
        while True:
            pass


# An open enum does not require every known case to appear.
# CHECK-LABEL: lit.fn @"match_open_enum_partial
# CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 0
def match_open_enum_partial(color: OpenColor):
    __match color:
    case .red:
        case_callee[0]()


# Covering every known case does not make the catch-all unreachable.
# CHECK-LABEL: lit.fn @"match_open_enum_catch_all
# CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 2
def match_open_enum_catch_all(color: OpenColor):
    __match color:
    case .red:
        case_callee[1]()
    case .green:
        case_callee[1]()
    case _:
        case_callee[2]()


# CHECK-LABEL: lit.fn @"match_color
# The three color tags share one discriminant; trailing `_` is unreachable
# after red/green/blue, so the elif else is `hlcf.unreachable`.
# CHECK:       [[DISC:%.*]] = lit.call {{.*}}@"_get_enum_discriminant{{.*}}
# CHECK:       [[TAG0:%.*]] = kgen.rebind [[DISC]]
# CHECK:       lit.call {{.*}}@"__eq__({{.*}}([[TAG0]],
# CHECK:       lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:       hlcf.if %{{.*}} {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         [[TAG1:%.*]] = kgen.rebind [[DISC]]
# CHECK:         lit.call {{.*}}@"__eq__({{.*}}([[TAG1]],
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.if.elifcond.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         [[TAG2:%.*]] = kgen.rebind [[DISC]]
# CHECK:         lit.call {{.*}}@"__eq__({{.*}}([[TAG2]],
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.if.elifcond.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 2
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         hlcf.unreachable
# CHECK:       }
def match_color(c: Color):
    __match c:
    case Color.red:  # explicit enum case.
        case_callee[0]()
    case .green:     # inferred case.
        case_callee[1]()
    case .blue:     # inferred case.
        case_callee[2]()
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        case_callee[3]()


# CHECK-LABEL: lit.fn @"match_color_complex
# One discriminant for `tc[0]`. Same Color tags are pulled together so `red`
# is tested once; its Int residuals (0 and 4) share a nested elif.
# CHECK:       lit.call {{.*}}@"__getitem_param__{{.*}}(%tc)
# CHECK:       [[DISC:%.*]] = lit.call {{.*}}@"_get_enum_discriminant{{.*}}
# CHECK:       [[TAG0:%.*]] = kgen.rebind [[DISC]]
# CHECK:       lit.call {{.*}}@"__eq__({{.*}}([[TAG0]],
# CHECK:       hlcf.if %{{.*}} {
# CHECK:         lit.call {{.*}}@"__getitem_param__{{.*}}(%tc)
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         hlcf.if %{{.*}} {
# CHECK:           lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:           hlcf.yield
# CHECK:         } else {
# CHECK:           lit.call {{.*}}@"__eq__(
# CHECK:           hlcf.if.elifcond.yield
# CHECK:         } then {
# CHECK:           lit.call {{.*}}@"case_callee{{.*}}<index> 2
# CHECK:           hlcf.yield
# CHECK:         } else {
# CHECK:           hlcf.yield
# CHECK:         }
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         [[TAG1:%.*]] = kgen.rebind [[DISC]]
# CHECK:         lit.call {{.*}}@"__eq__({{.*}}([[TAG1]],
# CHECK:         hlcf.if.elifcond.yield
# CHECK:       } then {
# CHECK:         hlcf.match {
# CHECK:           lit.call {{.*}}@"__getitem_param__{{.*}}(%tc)
# CHECK:           lit.call {{.*}}@"__eq__(
# CHECK:           hlcf.if %{{.*}} {
# CHECK:             hlcf.yield
# CHECK:           } else {
# CHECK:             hlcf.match.next
# CHECK:           }
# CHECK:           lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:           hlcf.match.complete
# CHECK:         }
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         [[TAG2:%.*]] = kgen.rebind [[DISC]]
# CHECK:         lit.call {{.*}}@"__eq__({{.*}}([[TAG2]],
# CHECK:         hlcf.if.elifcond.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 3
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         hlcf.yield
# CHECK:       }
def match_color_complex(tc: Tuple[Color, Int]):
    __match tc:
    case (Color.red, 0):
        case_callee[0]()
    case (Color.green, 0):
        case_callee[1]()
    case (Color.red, 4):
        case_callee[2]()
    case (Color.blue, _):
        case_callee[3]()

# CHECK-LABEL: lit.fn @"match_var_and_ref_binding
def match_var_and_ref_binding(value: String, mut mutString: String):
    # `var x` copies the borrowed subject into an owned binding that can be mutated.
    # CHECK:       [[X:%.*]] = lit.var.decl "x" var
    # CHECK:       lit.call {{.*}}@"__init__(copy:::String)"{{.*}}(%value, [[X]])
    # CHECK:       lit.call {{.*}}@"__iadd__{{.*}}([[X]],
    # CHECK:       lit.call {{.*}}@"byte_length(
    __match value:
    case var x:
        x += "x"
        _ = x.byte_length()

    # `ref y` stores a reference to the subject; no copy.
    # CHECK:       [[Y:%.*]] = lit.var.decl "y" ref
    # CHECK:       lit.ref.store %value, [[Y]]
    # CHECK:       lit.call {{.*}}@"byte_length(
    __match value:
    case ref y:
        _ = y.byte_length()

    # `ref z` maintains the mutability of the subject.
    # CHECK:       [[Z:%.*]] = lit.var.decl "z" ref
    # CHECK:       lit.ref.store %mutString, [[Z]]
    # CHECK:       lit.call {{.*}}@"__iadd__{{.*}}(
    __match mutString:
    case ref z:
        z += "x"


# Tuple patterns with nested var bindings (from the pattern-matching proposal).
def match_inspect_point(point: Tuple[Int, Int]):
    __match point:
    case (0, 0):
        case_callee[0]()
    case (var x, 0):
        case_callee[1]()
    case (0, var y):
        case_callee[2]()
    case (var x, var y):
        case_callee[3]()


# CHECK-LABEL: lit.fn @"match_as_pattern
def match_as_pattern(value: String):
    # `as` binds a ref to the whole subject, never a copy.
    # CHECK:       [[S:%.*]] = lit.var.decl "s" ref
    # CHECK:       lit.ref.store %value, [[S]]
    # CHECK-NOT:   lit.call {{.*}}@"__init__(copy:::String)"
    # CHECK:       lit.call {{.*}}@"byte_length(
    __match value:
    case _ as s:
        _ = s.byte_length()

    # Combined with a value pattern: equality runs first; binding is
    # materialized after the pattern succeeds.
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       [[T:%.*]] = lit.var.decl "t" ref
    # CHECK:       lit.ref.store %value, [[T]]
    __match value:
    case "hello" as t:
        _ = t.byte_length()

    # Register-passable subjects cannot be `ref`; `as` uses `bind` instead.
    # CHECK:       [[X:%.*]] = lit.var.decl "x" bound
    # CHECK:       lit.ref.store {{.*}}, [[X]]
    # CHECK:       lit.call {{.*}}@"__add__(
    __match 42:
    case _ as x:
        _ = x + 1

    # Bindings are materialized after pattern tests in each case region.
    # CHECK:       hlcf.match {
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       [[ORIGIN:%.*]] = lit.var.decl "origin" ref
    # CHECK:       lit.ref.store %point, [[ORIGIN]]
    # CHECK:       case {
    # CHECK:       hlcf.if %{{.*}} {
    # CHECK:       [[P:%.*]] = lit.var.decl "p" ref
    # CHECK:       lit.ref.store %point, [[P]]
    # CHECK:       lit.var.decl "x" var
    var point: Tuple[Int, Int] = (0, 0)
    __match point:
    case (0, 0) as origin:
        _ = origin
        case_callee[0]()
    case (var x, 0) as p:
        _ = x
        _ = p
        case_callee[1]()

    # `as` can be combined with a guard. The guard itself can use the binding.
    # CHECK:       [[S:%.*]] = lit.var.decl "s" ref
    # CHECK:       lit.ref.store %value, [[S]]
    # CHECK:       [[R:%.*]] = lit.ref.load [[S]]
    # CHECK:       [[L0:%.*]] = lit.call {{.*}}@"byte_length({{.*}}([[R]])
    # CHECK:       [[L1:%.*]] = kgen.rebind [[L0]]
    # CHECK:       [[Z0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
    # CHECK:       [[NE0:%.*]] = lit.call {{.*}}@"__ne__({{.*}}([[L1]], [[Z0]])
    # CHECK:       [[B0:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[NE0]])
    # CHECK:       hlcf.if [[B0]] {
    # CHECK:       lit.call {{.*}}@"byte_length(
    __match value:
    case _ as s if s.byte_length() != 0:
        _ = s.byte_length()


# CHECK-LABEL: lit.fn @"match_or_pattern_int
# CHECK-NEXT:    hlcf.match {
# CHECK-NEXT:      hlcf.match {
# CHECK-NEXT:        [[L0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:        [[EQ0:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L0]])
# CHECK-NEXT:        [[B0:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ0]])
# CHECK-NEXT:        hlcf.if [[B0]] {
# CHECK-NEXT:          hlcf.yield
# CHECK-NEXT:        } else {
# CHECK-NEXT:          hlcf.match.next
# CHECK-NEXT:        }
# CHECK-NEXT:        hlcf.match.complete
# CHECK-NEXT:      }
# CHECK-NEXT:      case {
# CHECK-NEXT:        [[L1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
# CHECK-NEXT:        [[EQ1:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L1]])
# CHECK-NEXT:        [[B1:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ1]])
# CHECK-NEXT:        hlcf.if [[B1]] {
# CHECK-NEXT:          hlcf.yield
# CHECK-NEXT:        } else {
# CHECK-NEXT:          hlcf.match.next
# CHECK-NEXT:        }
# CHECK-NEXT:        hlcf.match.complete
# CHECK-NEXT:      } else {
# CHECK-NEXT:        hlcf.match.next
# CHECK-NEXT:      }
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK-NEXT:      hlcf.match.complete
# CHECK-NEXT:    }
# CHECK-NEXT:    case {
# Or-alternatives are flattened into one nested match (not recursively nested).
# CHECK:         hlcf.match {
# CHECK:           kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK:           hlcf.if %{{.*}} {
# CHECK:             hlcf.yield
# CHECK:           } else {
# CHECK:             hlcf.match.next
# CHECK:           }
# CHECK:           hlcf.match.complete
# CHECK:         case {
# CHECK:           kgen.param.constant: !Int = <{:scalar<index> 1}>
# CHECK:           hlcf.if %{{.*}} {
# CHECK:             hlcf.yield
# CHECK:           } else {
# CHECK:             hlcf.match.next
# CHECK:           }
# CHECK:           hlcf.match.complete
# CHECK:         case {
# CHECK:           kgen.param.constant: !Int = <{:scalar<index> 2}>
# CHECK:           hlcf.if %{{.*}} {
# CHECK:             hlcf.yield
# CHECK:           } else {
# CHECK:             hlcf.match.next
# CHECK:           }
# CHECK:           hlcf.match.complete
# CHECK:         } else {
# CHECK:           hlcf.match.next
# CHECK:         }
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         hlcf.match.complete
def match_or_pattern_int(x: Int):
    __match x:
    case 0 | 1:
        case_callee[0]()
    case 0 | 1 | 2:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_or_pattern_tup
# CHECK:       hlcf.match {
# CHECK:         hlcf.match {
# CHECK:           lit.call {{.*}}@"__getitem_param__
# CHECK:           lit.call {{.*}}@"__eq__(
# CHECK:         hlcf.if %{{.*}} {
# CHECK:           hlcf.yield
# CHECK:         } else {
# CHECK:           hlcf.match.next
# CHECK:         }
# CHECK:           lit.call {{.*}}@"__getitem_param__
# CHECK:           lit.call {{.*}}@"__eq__(
# CHECK:         hlcf.if %{{.*}} {
# CHECK:           hlcf.yield
# CHECK:         } else {
# CHECK:           hlcf.match.next
# CHECK:         }
# CHECK:           hlcf.match.complete
# CHECK:         case {
# CHECK:           lit.call {{.*}}@"__getitem_param__
# CHECK:           lit.call {{.*}}@"__eq__(
# CHECK:         hlcf.if %{{.*}} {
# CHECK:           hlcf.yield
# CHECK:         } else {
# CHECK:           hlcf.match.next
# CHECK:         }
# CHECK:           lit.call {{.*}}@"__getitem_param__
# CHECK:           lit.call {{.*}}@"__eq__(
# CHECK:         hlcf.if %{{.*}} {
# CHECK:           hlcf.yield
# CHECK:         } else {
# CHECK:           hlcf.match.next
# CHECK:         }
# CHECK:           hlcf.match.complete
# CHECK:         } else {
# CHECK:           hlcf.match.next
# CHECK:         }
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 2
# CHECK:         hlcf.match.complete
def match_or_pattern_tup(point: Tuple[Int, Int]):
    __match point:
    case (0, 0) | (1, 1):
        case_callee[2]()


# CHECK-LABEL: lit.fn @"match_or_pattern_bind
def match_or_pattern_bind(var point: Tuple[Int, Int]):
    # Shared temp VarDecl before the nested or-match; each arm stores into it.
    # CHECK:       [[X:%.*]] = lit.var.decl "x" var
    # CHECK:       hlcf.match {
    # CHECK:       lit.ref.store {{.*}}, [[X]]
    # CHECK:       lit.ref.store {{.*}}, [[X]]
    # CHECK:       lit.var.decl "x" var
    # CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 3
    __match point:
    case (0, var x) | (var x, 1):
        _ = x
        case_callee[3]()

    # Same with `ref` bindings.
    # CHECK:       [[RX:%.*]] = lit.var.decl "x" ref
    # CHECK:       hlcf.match {
    # CHECK:       lit.ref.store {{.*}}, [[RX]]
    # CHECK:       lit.ref.store {{.*}}, [[RX]]
    # CHECK:       lit.var.decl "x" ref
    # CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 4
    __match point:
    case (2, ref x) | (ref x, 3):
        _ = x
        case_callee[4]()


# CHECK-LABEL: lit.fn @"test_or_MOCO_4953
def test_or_MOCO_4953(var a: Optional[Int]):
    # The subject is materialized once before the alternatives, so both
    # payload references have the same live origin and initialize one binding.
    # CHECK:       [[X:%.*]] = lit.var.decl "x" ref : !lit.ref<!lit.ref<!Int, [[COMMON_ORIGIN:.*]]>, mut
    # CHECK:       lit.ref.store {{.*}}, [[X]] : <!lit.ref<!Int, [[COMMON_ORIGIN]]>, mut
    # CHECK:       lit.ref.store {{.*}}, [[X]] : <!lit.ref<!Int, [[COMMON_ORIGIN]]>, mut
    # CHECK:       lit.var.decl "x" ref
    __match (a, a):
    case (.Some(x), .None) | (.None, .Some(x)):
        _ = x
    case _:
        pass


@fieldwise_init
struct Vec3:
    var x: Int
    var y: Int
    var z: Int


# CHECK-LABEL: lit.fn @"match_vec3
def match_vec3(v: Vec3):
    # Keyword field patterns: project each named field and match it.
    # CHECK:       lit.ref.struct.ger {{.*}}[x]
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       lit.ref.struct.ger {{.*}}[y]
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       lit.ref.struct.ger {{.*}}[z]
    # CHECK:       lit.call {{.*}}@"__eq__(
    __match v:
    case Vec3(x=0, y=0, z=0):
        case_callee[0]()

    # Bindings are materialized after field tests. `_` does not project.
    # CHECK:       lit.ref.struct.ger {{.*}}[x]
    # CHECK:       lit.ref.struct.ger {{.*}}[y]
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       [[X:%.*]] = lit.var.decl "x" var
    # CHECK:       lit.ref.store {{.*}}, [[X]]
    __match v:
    case Vec3(x=var x, y=0, z=_):
        _ = x
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_optional
def match_optional(opt: Optional[Int], mut mut_opt: Optional[Int]):
    # Immutable Optional: None then Some (sorted by discriminant). Some's
    # then-region projects the payload.
    # CHECK:       lit.call {{.*}}@"_get_enum_discriminant{{.*}}[imm *"opt`
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       hlcf.if %{{.*}} {
    # CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
    # CHECK:       } else {
    # CHECK:         hlcf.if.elifcond.yield
    # CHECK:       } then {
    # CHECK:         lit.call {{.*}}@"_unsafe_get_enum_payload{{.*}}(%opt)
    # CHECK:         [[ELT:%.*]] = lit.var.decl "elt" ref
    # CHECK:         lit.ref.store {{.*}}, [[ELT]]
    # CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
    __match opt:
    case Optional.Some(ref elt):
        _ = elt
        case_callee[0]()
    case Optional.None:
        case_callee[1]()

    # Mutable Optional: discriminant is read immutably (muttoimm); payload
    # projection keeps the mut origin so `ref` bindings can mutate.
    # CHECK:       lit.ref.immut %mut_opt
    # CHECK:       lit.call {{.*}}@"_get_enum_discriminant{{.*}}[muttoimm *"mut_opt`
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       hlcf.if %{{.*}} {
    # CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 3
    # CHECK:       } else {
    # CHECK:         hlcf.if.elifcond.yield
    # CHECK:       } then {
    # CHECK:         lit.call {{.*}}@"_unsafe_get_enum_payload{{.*}}(%mut_opt)
    # CHECK:         [[MELT:%.*]] = lit.var.decl "elt" ref
    # CHECK:         lit.ref.store {{.*}}, [[MELT]]
    # CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 2
    __match mut_opt:
    case Optional.Some(ref elt):
        _ = elt
        case_callee[2]()
    case Optional.None:
        case_callee[3]()

    # Matching with inferred base also work.
    __match opt:
    case .Some(ref elt):
        case_callee[0]()
    case ((.None)):  # Extra parens are fine of course.
        case_callee[1]()
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case .Some:  # just check the tag, don't bind the value.
        case_callee[2]()


# Owned Optional: `ref` into the payload keeps the subject's mut origin, so
# mutating through the binding writes back into `a`.
# CHECK-LABEL: lit.fn @"testLValueMutableMatch
def testLValueMutableMatch(var a: Optional[Int]):
    # Discriminant is read immutably; payload projection stays mut on `"a`.
    # CHECK:       lit.ref.immut %a
    # CHECK:       lit.call {{.*}}@"_get_enum_discriminant{{.*}}[muttoimm *"a`
    # CHECK:       lit.call {{.*}}@"__eq__(
    # CHECK:       hlcf.if %{{.*}} {
    # CHECK:         hlcf.yield
    # CHECK:       } else {
    # CHECK:         hlcf.match.next
    # CHECK:       }
    # CHECK:       lit.call {{.*}}@"_unsafe_get_enum_payload{{.*}}(%a)
    # CHECK:       [[VALUE:%.*]] = lit.var.decl "value" ref
    # CHECK:       lit.ref.store {{.*}}, [[VALUE]]
    # CHECK:       lit.call {{.*}}@"__iadd__{{.*}}[mut *"a`
    __match a: # expected-warning {{'match' is not exhaustive; missing case for 'None'}}
    case .Some(ref value):
        value += 1

# Bool ladder is one `hlcf.if` (no enclosing match: nothing follows it).
# Arms are ordered by spelling (`False` then `True`).
# The tuple ladder folds `(_, y_elt)` into the elif else of `y[0] == True`.
# CHECK-LABEL: lit.fn @"testMatchLadder
def testMatchLadder(x: Bool, y: Tuple[Bool, Int]):
    # CHECK:       lit.call {{.*}}@"__eq__(::Bool,::Bool)"(%x,
    # CHECK:       hlcf.if %{{.*}} {
    # CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
    # CHECK:         hlcf.yield
    # CHECK:       } else {
    # CHECK:         lit.call {{.*}}@"__eq__(::Bool,::Bool)"(%x,
    # CHECK:         hlcf.if.elifcond.yield
    # CHECK:       } then {
    # CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
    # CHECK:         hlcf.yield
    # CHECK:       } else {
    # CHECK:         hlcf.unreachable
    # CHECK:       }
    __match x:
    case True:
        case_callee[0]()
    case False:
        case_callee[1]()

    # `(True, _)` is decided by y[0]; `(_, y_elt)` is the exclusive else.
    # The catch-all is concluding, so the nested match else is unreachable.
    # CHECK:       lit.call {{.*}}@"__getitem_param__{{.*}}(%y)
    # CHECK:       lit.call {{.*}}@"__eq__(::Bool,::Bool)"(
    # CHECK:       hlcf.if %{{.*}} {
    # CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
    # CHECK:         hlcf.yield
    # CHECK:       } else {
    # CHECK:         hlcf.match {
    # CHECK:           lit.call {{.*}}@"__getitem_param__{{.*}}(%y)
    # CHECK:           [[Y_ELT:%.*]] = lit.var.decl "y_elt" ref
    # CHECK:           lit.ref.store {{.*}}, [[Y_ELT]]
    # CHECK:           lit.call {{.*}}@"case_callee{{.*}}<index> 1
    # CHECK:           hlcf.match.complete
    # CHECK:         } else {
    # CHECK:           hlcf.unreachable
    # CHECK:         }
    # CHECK:         hlcf.yield
    # CHECK:       }
    __match y:
    case (True, _):
        case_callee[0]()
    case (_, y_elt):
        case_callee[1]()


# ===----------------------------------------------------------------------=== #
# comptime match
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.fn @"comptime_match_wildcard
def comptime_match_wildcard[x: Int]():
    # Irrefutable `case _` emits the body with no comptime.if.
    # CHECK-NOT: hlcf.comptime.if
    # CHECK:     lit.call {{.*}}@"case_callee{{.*}}<index> 0
    comptime __match x:
    case _:
        case_callee[0]()


# CHECK-LABEL: lit.fn @"comptime_match_literals
def comptime_match_literals[x: Int]():
    # CHECK:      hlcf.comptime.if {{.*}}identical{{.*}}#lit.struct.extract<:!Int x, "_mlir_value">{{.*}}0{{.*}} {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 0
    # CHECK:        hlcf.comptime.yield
    # CHECK:      } elif {{.*}}identical{{.*}}#lit.struct.extract<:!Int x, "_mlir_value">{{.*}}1{{.*}} {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 1
    # CHECK:        hlcf.comptime.yield
    # CHECK:      } else {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 2
    # CHECK:        hlcf.comptime.yield
    # CHECK:      }
    comptime __match x:
    case 0:
        case_callee[0]()
    case 1:
        case_callee[1]()
    case _:
        case_callee[2]()


# CHECK-LABEL: lit.fn @"comptime_match_bool
def comptime_match_bool[flag: Bool]():
    # True/False cover Bool, so the else is sealed with `hlcf.unreachable`.
    # CHECK:      hlcf.comptime.if {{.*}}__eq__{{.*}}flag{{.*}}{:scalar<bool> true}{{.*}} {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 0
    # CHECK:        hlcf.comptime.yield
    # CHECK:      } elif {{.*}}__eq__{{.*}}flag{{.*}}{:scalar<bool> false}{{.*}} {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 1
    # CHECK:        hlcf.comptime.yield
    # CHECK:      } else {
    # CHECK:        hlcf.unreachable
    # CHECK:      }
    comptime __match flag:
    case True:
        case_callee[0]()
    case False:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"comptime_match_bool_unreachable_wildcard
def comptime_match_bool_unreachable_wildcard[flag: Bool]():
    # Trailing `_` is unreachable after True/False; body is parsed then sealed.
    # CHECK:      hlcf.comptime.if {{.*}} {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 0
    # CHECK:        hlcf.comptime.yield
    # CHECK:      } elif {{.*}} {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 1
    # CHECK:        hlcf.comptime.yield
    # CHECK:      } else {
    # CHECK:        hlcf.unreachable
    # CHECK:      }
    comptime __match flag:
    case True:
        case_callee[0]()
    case False:
        case_callee[1]()
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        case_callee[2]()


# CHECK-LABEL: lit.fn @"comptime_match_or
def comptime_match_or[x: Int]():
    # Or-pattern lowers to cond(eq0, eq0, eq1) — same as comptime `or`.
    # CHECK:      hlcf.comptime.if cond({{.*}}identical{{.*}}x{{.*}}0{{.*}}, {{.*}}identical{{.*}}x{{.*}}0{{.*}}, {{.*}}identical{{.*}}x{{.*}}1{{.*}}) {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 0
    # CHECK:        hlcf.comptime.yield
    # CHECK:      } else {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 1
    # CHECK:        hlcf.comptime.yield
    # CHECK:      }
    comptime __match x:
    case 0 | 1:
        case_callee[0]()
    case _:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"comptime_match_or_binding
def comptime_match_or_binding[x: Tuple[Int, Int]]():
    # Binding values are Cond-selected with the same cond as the or:
    #   or(c0, c1) = cond(c0, c0, c1)
    #   y          = cond(c0, y0, y1)  (different paths → Cond does not fold)
    # CHECK:      hlcf.comptime.if cond({{.*}}identical{{.*}}0{{.*}}, {{.*}}identical{{.*}}0{{.*}}, {{.*}}identical{{.*}}1{{.*}}) {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}cond(
    # CHECK:        hlcf.comptime.yield
    # CHECK:      } else {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 0
    # CHECK:        hlcf.comptime.yield
    # CHECK:      }
    comptime __match x:
    case (0, y) | (y, 1):
        case_callee[y]()
    case _:
        case_callee[0]()


# CHECK-LABEL: lit.fn @"comptime_match_binding
def comptime_match_binding[x: Int]():
    # Bare binding is irrefutable: no comptime.if; `y` is a PValue alias of `x`.
    # CHECK-NOT: hlcf.comptime.if
    # CHECK:     lit.call {{.*}}@"case_callee{{.*}}<:!Int x>
    comptime __match x:
    case y:
        case_callee[y]()


# CHECK-LABEL: lit.fn @"comptime_match_binding_guard
def comptime_match_binding_guard[x: Int]():
    # Binding is visible to the guard; bind-only pattern + guard → just guard.
    # CHECK:      hlcf.comptime.if {{.*}}__gt__{{.*}}x{{.*}}0{{.*}} {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<:!Int x>
    # CHECK:        hlcf.comptime.yield
    # CHECK:      } else {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 1
    # CHECK:        hlcf.comptime.yield
    # CHECK:      }
    comptime __match x:
    case y if y > 0:
        case_callee[y]()
    case _:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"comptime_match_literal_guard
def comptime_match_literal_guard[x: Int, flag: Bool]():
    # Value test AND guard.
    # CHECK:      hlcf.comptime.if cond({{.*}}identical{{.*}}x{{.*}}0{{.*}}, {{.*}}flag{{.*}}, {{.*}}identical{{.*}}x{{.*}}0{{.*}}) {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 0
    # CHECK:        hlcf.comptime.yield
    # CHECK:      } else {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 1
    # CHECK:        hlcf.comptime.yield
    # CHECK:      }
    comptime __match x:
    case 0 if flag:
        case_callee[0]()
    case _:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"comptime_match_literal_binding
def comptime_match_literal_binding[x: Int]():
    # `as` binding after a value test; binding aliases the subject in the body.
    # CHECK:      hlcf.comptime.if {{.*}}identical{{.*}}x{{.*}}0{{.*}} {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<:!Int x>
    # CHECK:        hlcf.comptime.yield
    # CHECK:      } else {
    # CHECK:        lit.call {{.*}}@"case_callee{{.*}}<index> 1
    # CHECK:        hlcf.comptime.yield
    # CHECK:      }
    comptime __match x:
    case 0 as y:
        case_callee[y]()
    case _:
        case_callee[1]()
