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

def match_bad_subject_skips_same_indent_cases():
    # expected-error @+1 {{use of unknown declaration 'no_such_subject'}}
    __match no_such_subject:
    case 0:
        pass
    case 1 if True:
        pass
    # Recovery must resume here, not leave a dangling 'case' for the outer suite.
    # expected-error @+1 {{use of unknown declaration 'also_missing'}}
    _ = also_missing


def match_bad_subject_skips_indented_cases():
    # expected-error @+1 {{use of unknown declaration 'no_such_subject'}}
    __match no_such_subject:
        case 0:
            pass
        case _:
            pass
    # expected-error @+1 {{use of unknown declaration 'also_missing'}}
    _ = also_missing


def match_missing_cases(x: Int):
    # expected-error @+1 {{'match' statement must have at least one 'case' block}}
    __match x:

    __match x:
    case foo(): # expected-error {{use of unknown declaration 'foo'}}
        pass

# expected-error @+1 {{'__match' must be contained in a function}}
__match 1:
    case 0:
        pass

def various_match_issues(a: Int, point: Tuple[Int, Int], value: String):
    __match a:
    case 0:
        var x = 42
    case _:
        # expected-error @+1 {{use of unknown declaration 'x'}}
        _ = x

    __match point:
    # expected-error @+1 {{cannot match value of 'Tuple[Int, Int]' of 2 elements against a pattern with 3 elements}}
    case (0, 0, 0):
        pass
    case _:
        pass

    __match point:
    # expected-error @+2 {{invalid redefinition of 'x'}}
    # expected-note @+1 {{previous definition here}}
    case (var x, var x):
        pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        pass

    # Nested var/ref patterns should warn.
    __match value:
    # expected-warning @+1 {{nested 'var' or 'ref' patterns are redundant, remove the outer pattern}}
    case var ref x:
        _ = x.byte_length()

    __match value:
    # expected-warning @+1 {{nested 'var' or 'ref' patterns are redundant}}
    case ref var y:
        _ = y.byte_length()

    __match a:
    # expected-error @+1 {{expected a name after 'as'}}
    case 0 as 1:
        pass
    case _:
        pass

    __match point:
    # expected-error @+1 {{expected a name after 'as'}}
    case (0 as 1, 2):
        pass
    case _:
        pass

    __match Int():
    # expected-error @+1 {{value of type 'Int' doesn't have a memory origin in 'ref' binding}}
    case ref z:
        pass

    __match a:
    # expected-error @+1 {{expected a tuple type to match against, got 'Int'}}
    case 0 | (1, 2):
        pass
    case _:
        pass


def match_or_pattern_binding_diags(var point: Tuple[Int, Int],
                                   mixed: Tuple[Int, String]):
    # Binding only on the left alternative.
    __match point:
    # expected-error @+1 {{or-pattern alternatives must bind the same names; 'x' is bound in one alternative but not the other}}
    case (0, var x) | (1, 2):
        pass
    case _:
        pass

    # Binding only on the right alternative.
    __match point:
    # expected-error @+1 {{or-pattern alternatives must bind the same names; 'x' is bound in one alternative but not the other}}
    case (0, 1) | (var x, 2):
        pass
    case _:
        pass

    # Same name, but `var` vs `ref`.
    __match point:
    # expected-error @+1 {{or-pattern binding 'x' must use the same 'var'/'ref' kind in each alternative}}
    case (0, var x) | (ref x, 1):
        pass
    case _:
        pass

    # Different binding names across alternatives.
    __match point:
    # expected-error @+1 {{or-pattern alternatives must bind the same names; 'x' is bound in one alternative but not the other}}
    case (var x, 0) | (var y, 1):
        pass
    case _:
        pass

    # Different number of bindings. Leading literals keep both arms live so
    # the or does not constant-fold away the RHS. The RHS is irrefutable, so
    # the trailing `_` is unreachable.
    __match point:
    # expected-error @+1 {{or-pattern alternatives must bind the same names}}
    case (0, var x) | (var x, var y):
        pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        pass

    # Same name and kind, but incompatible types (String vs Int).
    # `(var x, _)` is irrefutable, so the trailing `_` is unreachable.
    __match mixed:
    # expected-error @+2 {{or-pattern binding 'x' has incompatible types across alternatives}}
    # expected-note @+1 {{first alternative has type 'String', this alternative has type 'Int'}}
    case (0, var x) | (var x, _):
        pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        pass


@fieldwise_init
struct Vec3:
    var x: Int
    var y: Int
    var z: Int


def match_struct_pattern_diags(v: Vec3):
    __match v:
    # expected-error @+1 {{cannot match value of type 'Vec3' against pattern type 'Int'}}
    case Int(x=0):
        pass
    case _:
        pass

    __match v:
    # expected-error @+1 {{'w' is not a field of 'Vec3'}}
    case Vec3(w=0):
        pass
    case _:
        pass

    __match v:
    # expected-error @+2 {{keyword argument 'x' was already used; remove the duplicate}}
    # expected-note @+1 {{previously specified here}}
    case Vec3(x=0, x=1):
        pass
    case _:
        pass

    __match v:
    # expected-error @+1 {{struct patterns do not support positional or unpacked arguments}}
    case Vec3(0, y=1):
        pass
    case _:
        pass

    __match v:
    # expected-error @+1 {{struct patterns do not support positional or unpacked arguments}}
    case Vec3(x=0, *y):
        pass
    case _:
        pass

    __match v:
    # expected-error @+1 {{struct patterns do not support positional or unpacked arguments}}
    case Vec3(x=0, **y):
        pass
    case _:
        pass


def match_enum_pattern_diags(opt: Optional[Int], someEnum: Some[EnumLike]):
    # No-payload cases must be written without parentheses.
    __match opt:
    # expected-error @+1 {{enum case 'None' has no associated value}}
    case Optional.None():
        pass
    case _:
        pass

    # Cannot destructure a case whose payload is NoneType.
    __match opt:
    # expected-error @+1 {{enum case 'None' has no associated value}}
    case Optional.None(value):
        pass
    case _:
        pass

    # Payload cases require a subpattern inside the parentheses.
    __match opt:
    # expected-error @+1 {{enum case 'Some' requires a payload pattern inside the parentheses}}
    case Optional.Some():
        pass
    case _:
        pass

    # Unknown case name (attribute form).
    __match opt:
    # expected-error @+1 {{'Nope' is not a case of 'Optional[Int]'}}
    case Optional.Nope:
        pass
    case _:
        pass

    # Unknown case name (call form).
    __match opt:
    # expected-error @+1 {{'Nope' is not a case of 'Optional[Int]'}}
    case Optional.Nope(ref x):
        pass
    case _:
        pass

    # Can only pattern match on concrete types.
    __match someEnum:
    case .What: # expected-error {{cannot match on a parametric enum type}}
        pass


@fieldwise_init
struct BoolPair:
    var a: Bool
    var b: Bool


def match_exhaustivity_diags(flag: Bool, opt: Optional[Int], pair: BoolPair):
    # Bool is EnumLike: True/False must both be covered (or a catch-all used).
    __match flag: # expected-warning {{'match' is not exhaustive; missing case for 'False'}}
    case True: pass

    # Duplicate Bool case after the type is fully covered.
    __match flag:
    case True: pass
    case False: pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        pass

    # Unreachable arms are still type-checked (body IR is emitted, then
    # discarded and replaced with `hlcf.unreachable`).
    __match flag:
    case True: pass
    case False: pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        # expected-error @+1 {{use of unknown declaration 'not_a_real_name'}}
        _ = not_a_real_name

    # Exhaustive Bool×Bool via correlated wildcards.
    __match flag, flag:
    case True, True:  pass
    case False, True: pass
    case _, False: pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        pass

    # Non-exhaustive product: (False, False) is missing.
    __match flag, flag: # expected-warning {{'match' is not exhaustive; missing case for 'False, False'}}
    case True, True: pass
    case False, True: pass
    case True, False: pass

    # Parenthesized tuple subject is the same product space of course.
    __match (flag, ((((flag))))):
    case (True, _): pass
    case (False, True): pass
    case (False, False): pass

    # Struct of EnumLike fields: same flattened product; omitted fields are
    # wildcards (`b=False` covers both values of `a`).
    __match pair:
    case BoolPair(a=True, b=True): pass
    case BoolPair(a=False, b=True): pass
    case BoolPair(b=False): pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        pass

    # Or of root constructors covers each alternative (union).
    __match flag:
    case True | False: pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        pass

    # Or that only covers one ctor is not exhaustive.
    __match flag: # expected-warning {{'match' is not exhaustive; missing case for 'False'}}
    case True | True: pass

    # Or in a product: each alternative covers its cells.
    __match flag, flag:
    case (True, _) | (False, False): pass
    case (False, True): pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        pass

    # Or with a catch-all arm closes the space.
    __match flag:
    case True | _: pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case False:
        pass

    # Redundant Or after the subject is already covered.
    __match flag:
    case True: pass
    case False: pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case True | False:
        pass

    # Various enum cases.
    __match opt: # expected-warning {{'match' is not exhaustive; missing case for 'None'}}
    case Optional.Some(_):
        pass

    __match opt: # expected-warning {{'match' is not exhaustive; missing case for 'Some'}}
    case Optional.None:
        pass

    # Duplicate Optional case while another constructor is still open.
    __match opt:
    case Optional.Some(_):
        pass
    # expected-warning @+1 {{'Some' is already covered by a previous case}}
    case Optional.Some(x):
        _ = x
    case Optional.None:
        pass

    # Payload refinements still credit the enum tag: `Some(0)` covers `Some`.
    __match opt:
    case Optional.Some(0):
        pass
    case Optional.None:
        pass

    # After a refined `Some` arm, a later `Some(_)` is unreachable at tag level.
    __match opt:
    case Optional.Some(0):
        pass
    # expected-warning @+1 {{'Some' is already covered by a previous case}}
    case Optional.Some(_):
        pass
    case Optional.None:
        pass

    # Products larger than kMaxProductCells (1024) cannot be tracked cell-by-cell
    # (Bool^11 == 2048); require a catch-all instead of silent Opaque.
    # expected-warning @+1 {{'match' is too complex to check for exhaustivity; add a '_' case}}
    __match (
        flag,
        flag,
        flag,
        flag,
        flag,
        flag,
        flag,
        flag,
        flag,
        flag,
        flag,
    ):
    case True, True, True, True, True, True, True, True, True, True, True:
        pass

    # A catch-all satisfies the TooComplex requirement.
    __match (
        flag,
        flag,
        flag,
        flag,
        flag,
        flag,
        flag,
        flag,
        flag,
        flag,
        flag,
    ):
    case True, True, True, True, True, True, True, True, True, True, True:
        pass
    case _:
        pass

def match_exhaustivity_literal_subjects(n: Int, s: String):
    # Int/String are open universes: covering literals never proves exhaustivity,
    # but duplicate / already-covered literal cases are diagnosed.
    __match n:
    case 0:
        pass
    case 1:
        pass

    __match s:
    case "a":
        pass
    case "b":
        pass

    __match s:
    case "a":
        pass
    # expected-warning @+1 {{case is unreachable; '"a"' is already covered by a previous case}}
    case "a":
        pass

    __match n:
    case 0:
        pass
    # expected-warning @+1 {{case is unreachable; '0' is already covered by a previous case}}
    case 0:
        pass
    case 1:
        pass

    # Or of literals covers each alternative; a later duplicate is unreachable.
    __match n:
    case 0 | 1 | 2:
        pass
    # expected-warning @+1 {{case is unreachable; '1' is already covered by a previous case}}
    case 1:
        pass

    # After a catch-all, everything else is unreachable (still no exhaustivity
    # warning when only literals are present).
    __match n:
    case 0:
        pass
    case _:
        pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case 1:
        pass


def comptime_match_or_binding_name_mismatch[x: Tuple[Int, Int]]():
    comptime __match x:
    # expected-error @+1 {{or-pattern alternatives must bind the same names; 'y' is bound in one alternative but not the other}}
    case (0, y) | (1, 2):
        pass


def comptime_match_var_binding_rejected[x: Int]():
    comptime __match x:
    case var y: # expected-error {{'var' bindings are not supported in 'comptime match'}}
        pass


def comptime_match_ref_binding_rejected[x: Int]():
    comptime __match x:
    case ref y: # expected-error {{'ref' bindings are not supported in 'comptime match'}}
        pass


def comptime_match_unreachable_body_still_diagnosed[flag: Bool]():
    # Unreachable arms are still parsed so body errors are reported.
    comptime __match flag:
    case True:
        pass
    case False:
        pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        # expected-error @+1 {{use of unknown declaration 'no_such_name'}}
        _ = no_such_name


def comptime_match_unreachable_after_catchall[x: Int]():
    comptime __match x:
    case _:
        pass
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case 1:
        # expected-error @+1 {{use of unknown declaration 'also_missing'}}
        _ = also_missing
