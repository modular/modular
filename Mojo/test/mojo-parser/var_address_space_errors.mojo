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


def bad_decorator_arguments():
    @__address_space  # expected-error {{@__address_space takes a single positional 'AddressSpace' argument}}
    var a: Int = 0

    @__address_space(.SHARED, .LOCAL)  # expected-error {{@__address_space takes a single positional 'AddressSpace' argument}}
    var b: Int = 0

    @__address_space(space=.SHARED)  # expected-error {{@__address_space takes a single positional 'AddressSpace' argument}}
    var c: Int = 0

    @__address_space(.SHARED)
    @__address_space(.LOCAL)  # expected-error {{@__address_space is specified more than once; remove the duplicate}}
    var d: Int = 0


def not_an_address_space():
    @__address_space("shared")  # expected-error {{cannot implicitly convert}}
    var a: Int = 0


def unsupported_statement_shape():
    @__address_space(.SHARED)
    var a, b = 1, 2  # expected-error {{@__address_space only applies to a 'var' that declares a single name}}


def other_decorators():
    @__address_space(.SHARED)
    @not_good
    var a: Int = 0  # expected-error {{'var' statement in function body does not support decorators; remove the decorator}}


struct HasField:
    @__address_space(.SHARED)  # expected-error {{struct fields can't have an address space; put @__address_space on the 'var' that holds the struct}}
    var x: Int


def use_field(h: HasField):
    pass


@__address_space(.SHARED)  # expected-error {{@__address_space only applies to a 'var' in a function body}}
def decorated_function():
    @__address_space(.SHARED)  # expected-error {{@__address_space only applies to a 'var' in a function body}}
    def nested():
        pass


@__address_space(.SHARED)  # expected-error {{@__address_space only applies to a 'var' in a function body}}
struct DecoratedStruct:
    var x: Int


def use_decorated_struct(s: DecoratedStruct):
    pass


# expected-note @below {{candidate not viable: missing required argument: 'move'}}
# expected-note @below {{def __init__(out self, *, deinit move: Self)    # note - generated function}}
struct SharedOnly:
    var x: Int

    # expected-note @below {{candidate not viable: cannot initialize a value in address space '0' with an initializer for address space '2'}}
    # expected-note @below {{candidate not viable: cannot initialize a value in address space '4' with an initializer for address space '2'}}
    # expected-note @below {{candidate not viable: cannot initialize a value in address space 'addr_space' with an initializer for address space '2'}}
    def __init__(out[AddressSpace.SHARED] self):
        self.x = 0


def wrong_address_space_for_initializer[addr_space: AddressSpace]():
    var a = SharedOnly()  # expected-error {{no matching function in initialization}}

    @__address_space(.LOCAL)
    var b = SharedOnly()  # expected-error {{no matching function in initialization}}

    @__address_space(addr_space)
    var c = SharedOnly()  # expected-error {{no matching function in initialization}}


def match_on_constructed_subject():
    # The subject is a temporary in the default address space.
    match SharedOnly():  # expected-error {{no matching function in initialization}}
        case var y:
            _ = y.x


def match_copies_out_of_shared():
    @__address_space(.SHARED)
    var s = SharedOnly()
    match s:
        case var y:  # expected-error {{value of type 'SharedOnly' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}
            _ = y.x


struct CopyableAnywhere(ImplicitlyCopyable):
    var x: Int

    def __init__[addr_space: AddressSpace](out[addr_space] self):
        self.x = 0

    def __init__(out self, *, copy: Self):
        self.x = copy.x


def implicit_copy_out_of_shared():
    @__address_space(.SHARED)
    var s = CopyableAnywhere()
    match s:
        case var y:  # expected-error {{non-implicitly trivially copyable value cannot be copied from a non-default address space}}
            _ = y.x

    var z = s  # expected-error {{non-implicitly trivially copyable value cannot be copied from a non-default address space}}
    _ = z.x


struct CrossAddressSpaceCopy(ImplicitlyCopyable):
    var x: Int

    def __init__[addr_space: AddressSpace](out[addr_space] self):
        self.x = 0

    def __init__(out self, *, copy: Self):
        self.x = copy.x

    def __init__[
        from_as: AddressSpace, to_as: AddressSpace
    ](out[to_as] self, *, ref[from_as] copy: Self):
        self.x = copy.x


# expected-note @below {{function declared here}}
def takes_local(ref[_, AddressSpace.LOCAL] x: CopyableAnywhere):
    pass


def wrong_address_space_for_argument[addr_space: AddressSpace]():
    @__address_space(.SHARED)
    var s = CopyableAnywhere()
    # expected-error @below {{invalid call to 'takes_local': value passed to 'x' cannot be converted from 'CopyableAnywhere' to ref 'CopyableAnywhere'}}
    # expected-note @below {{operand address space '2' doesn't match expected address space '4'}}
    takes_local(s)

    @__address_space(addr_space)
    var p = CopyableAnywhere()
    # expected-error @below {{invalid call to 'takes_local': value passed to 'x' cannot be converted from 'CopyableAnywhere' to ref 'CopyableAnywhere'}}
    # expected-note @below {{operand address space 'addr_space' doesn't match expected address space '4'}}
    takes_local(p)


def implicit_copy_ignores_cross_address_space_initializer():
    @__address_space(.SHARED)
    var s = CrossAddressSpaceCopy()
    match s:
        case var y:  # expected-error {{non-implicitly trivially copyable value cannot be copied from a non-default address space}}
            _ = y.x
