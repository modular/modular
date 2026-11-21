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

# A union case's pattern takes the case's parentheses, and its subpatterns
# match the elements of the case's payload, a `Tuple`.

def case_callee[p: Int](): pass


__union Tok(ImplicitlyCopyable):
    case eof()
    case word(String)
    case span(Int, Int)


# `.eof()` tests the tag alone. `.word(w)` binds element 0 of the payload;
# `.span(0, b)` tests element 0 and binds element 1, and shares the payload
# with `Tok.span(a, _)`, which binds element 0.
# CHECK-LABEL: lit.fn @"match_tok(
# CHECK:       [[DISC:%.*]] = lit.call {{.*}}@"_get_enum_discriminant{{.*}}
# CHECK:       hlcf.if %{{.*}} {
# CHECK-NEXT:    lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:       } then {
# CHECK-NEXT:    [[WORD:%.*]] = lit.call {{.*}}@"_unsafe_get_enum_payload{{.*}}<:!Int {:scalar<index> 1}
# CHECK:      [[W_STORAGE:%.*]] = lit.ref.struct.ger [[WORD]][_mlir_value]
# CHECK-NEXT: [[W_PTR:%.*]] = lit.ref.to_pointer [[W_STORAGE]]
# CHECK-NEXT: [[W_ELT:%.*]] = kgen.struct.gep [[W_PTR]][0]
# CHECK-NEXT: [[W:%.*]] = lit.ref.from_kgen_ptr [[W_ELT]]
# CHECK-NEXT:    %w = lit.var.decl "w" ref
# CHECK-NEXT:    lit.ref.store [[W]], %w
# CHECK-NEXT:    lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:       } else {
# CHECK-NEXT:    [[SPAN:%.*]] = lit.call {{.*}}@"_unsafe_get_enum_payload{{.*}}<:!Int {:scalar<index> 2}
# CHECK:      [[FIRST_STORAGE:%.*]] = lit.ref.struct.ger [[SPAN]][_mlir_value]
# CHECK-NEXT: [[FIRST_PTR:%.*]] = lit.ref.to_pointer [[FIRST_STORAGE]]
# CHECK-NEXT: [[FIRST_ELT:%.*]] = kgen.struct.gep [[FIRST_PTR]][0]
# CHECK-NEXT: [[FIRST:%.*]] = lit.ref.from_kgen_ptr [[FIRST_ELT]]
# CHECK:         lit.call {{.*}}@"__eq__
# CHECK:         hlcf.if %{{.*}} {
# CHECK:      [[SECOND_STORAGE:%.*]] = lit.ref.struct.ger [[SPAN]][_mlir_value]
# CHECK-NEXT: [[SECOND_PTR:%.*]] = lit.ref.to_pointer [[SECOND_STORAGE]]
# CHECK-NEXT: [[SECOND_ELT:%.*]] = kgen.struct.gep [[SECOND_PTR]][1]
# CHECK-NEXT: [[SECOND:%.*]] = lit.ref.from_kgen_ptr [[SECOND_ELT]]
# CHECK-NEXT:      %b = lit.var.decl "b" ref
# CHECK-NEXT:      lit.ref.store [[SECOND]], %b
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 2
# CHECK:         } else {
# CHECK-NEXT:      %a = lit.var.decl "a" ref
# CHECK-NEXT:      lit.ref.store [[FIRST]], %a
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 3
def match_tok(t: Tok):
    match t:
    case .eof():
        case_callee[0]()
    case .word(w):
        case_callee[1]()
    case .span(0, b):
        case_callee[2]()
    case Tok.span(a, _):
        case_callee[3]()


# A subpattern for every element of every case covers the union.
# CHECK-LABEL: lit.fn @"match_covered(
# CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:       hlcf.unreachable
def match_covered(t: Tok):
    match t:
    case .eof() | .word(_):
        case_callee[0]()
    case .span(_, _):
        case_callee[1]()
    # expected-warning @+1 {{case is unreachable; previous cases cover every value of the match subject}}
    case _:
        case_callee[2]()


def match_partial(t: Tok):
    # expected-warning @+1 {{'match' is not exhaustive; missing cases for 'word' and 'span'}}
    match t:
    case .eof():
        case_callee[0]()


# A refutable subpattern does not cover its case.
def match_refutable(t: Tok):
    # expected-warning @+1 {{'match' is not exhaustive; missing case for 'span'}}
    match t:
    case .eof() | .word(_):
        case_callee[0]()
    case .span(0, _):
        case_callee[1]()


def match_pair(a: Tok, b: Tok):
    # expected-warning @+1 {{'match' is not exhaustive; missing cases for 'eof, word', 'eof, span', 'word, eof', 'word, word', 'word, span', 'span, eof', 'span, word', and 'span, span'}}
    match a, b:
    case .eof(), .eof():
        case_callee[0]()


# CHECK-LABEL: lit.fn @"if_case(
# CHECK:       lit.call {{.*}}@"_unsafe_get_enum_payload{{.*}}<:!Int {:scalar<index> 2}
# CHECK:       lit.call {{.*}}@"case_callee{{.*}}<index> 0
def if_case(t: Tok):
    if case .span(a, b) = t:
        case_callee[0]()
