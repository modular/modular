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


# expected-error @+1 {{@__annotation requires at least one value}}
@__annotation
struct BareOnStruct:
    var value: Int


# expected-error @+1 {{@__annotation does not accept keyword arguments}}
@__annotation(value=1)
struct KeywordOnStruct:
    var value: Int


# Unpacking is reported separately: a splat is not a keyword argument, and
# expanding one would mean splicing a comptime variadic into the list.
comptime pack = (1, 2)


# expected-error @+1 {{@__annotation does not accept unpacked arguments}}
@__annotation(*pack)
struct StarOnStruct:
    var value: Int


# expected-error @+1 {{@__annotation does not accept unpacked arguments}}
@__annotation(**pack)
struct StarStarOnStruct:
    var value: Int


# expected-error @+1 {{@__annotation is only supported on structs and struct fields}}
@__annotation(1)
def annotated_function():
    pass


# Traits and comptime aliases have nowhere to store annotations either, and
# reject the decorator through their own decorator handling.
# expected-error @+1 {{unrecognized body decorators}}
@__annotation(1)
trait AnnotatedTrait:
    pass


trait AnnotatedTraitMember:
    # expected-error @+1 {{decorator on this statement is unsupported}}
    @__annotation(1)
    comptime member: Int


# expected-error @+1 {{decorator on this statement is unsupported}}
@__annotation(1)
comptime annotated_alias = 3


# A reader has to be able to move a value out of the annotation list and
# destroy it, so every value must be `Movable & Deinitable`.
@explicit_destroy("must be consumed")
struct Linear(Deinitable where False):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x

    def consume(deinit self):
        pass


# expected-error @+1 {{@__annotation value of type 'Linear' does not conform to 'Deinitable'}}
@__annotation(Linear(1))
struct NotDeinitable:
    var value: Int


# expected-error @+1 {{@__annotation value must be a value, not a type}}
@__annotation(Int)
struct TypeValued:
    var value: Int


# A trait bound has to provide `Deinitable` for a value of that type to be
# stored; `Copyable` alone does not.
struct UnboundedTraitValue[T: Copyable, v: T]:
    # expected-error @+1 {{@__annotation value of type 'T' does not conform to 'Deinitable'}}
    @__annotation(Self.v)
    var value: Int
