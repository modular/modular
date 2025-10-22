# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Implements type rebind/trait downcast

These are Mojo built-ins, so you don't need to import them.
"""


@always_inline("nodebug")
fn rebind[
    src_type: AnyTrivialRegType, //,
    dest_type: AnyTrivialRegType,
](src: src_type) -> dest_type:
    """Statically assert that a parameter input type `src_type` resolves to the
    same type as a parameter result type `dest_type` after function
    instantiation and "rebind" the input to the result type.

    This function is meant to be used in uncommon cases where a parametric type
    depends on the value of a constrained parameter in order to manually refine
    the type with the constrained parameter value.

    Parameters:
        src_type: The original type.
        dest_type: The type to rebind to.

    Args:
        src: The value to rebind.

    Returns:
        The rebound value of `dest_type`.
    """
    return __mlir_op.`kgen.rebind`[_type=dest_type](src)


@always_inline("nodebug")
fn rebind[
    src_type: AnyType, //,
    dest_type: AnyType,
](ref src: src_type) -> ref [src] dest_type:
    """Statically assert that a parameter input type `src_type` resolves to the
    same type as a parameter result type `dest_type` after function
    instantiation and "rebind" the input to the result type, returning a
    reference to the input value with an adjusted type.

    This function is meant to be used in uncommon cases where a parametric type
    depends on the value of a constrained parameter in order to manually refine
    the type with the constrained parameter value.

    Parameters:
        src_type: The original type.
        dest_type: The type to rebind to.

    Args:
        src: The value to rebind.

    Returns:
        A reference to the value rebound as `dest_type`.
    """
    lit = __get_mvalue_as_litref(src)
    rebound = rebind[Pointer[dest_type, origin_of(src)]._mlir_type](lit)
    return __get_litref_as_mvalue(rebound)


@always_inline("nodebug")
fn rebind_var[
    src_type: Movable, //,
    dest_type: Movable,
](var src: src_type, out dest: dest_type):
    """Statically assert that a parameter input type `src_type` resolves to the
    same type as a parameter result type `dest_type` after function
    instantiation and "rebind" the input to the result type, returning a
    owned variable with an adjusted type.

    Unlike `rebind`, this function takes an owned variable and returns an owned
    variable via moving the value from the input to the output.

    This function is meant to be used in uncommon cases where a parametric type
    depends on the value of a constrained parameter in order to manually refine
    the type with the constrained parameter value.

    Parameters:
        src_type: The original type.
        dest_type: The type to rebind to.

    Args:
        src: The value to rebind.

    Returns:
        An owned value rebound as `dest_type`.
    """
    ref dest_ref = rebind[dest_type](src)
    dest = UnsafePointer(to=dest_ref).take_pointee()
    __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(src))


alias downcast[_Trait: type_of(AnyType), T: AnyType] = __mlir_attr[
    `#kgen.downcast<`, T, `> : `, _Trait
]


@always_inline
fn trait_downcast[
    T: AnyTrivialRegType, //, Trait: type_of(AnyType)
](var src: T) -> downcast[Trait, T]:
    """Downcast a parameter input type `T` and rebind the type such that the
    return value's type conforms the provided `Trait`. If `T`, after resolving
    to a concrete type, does not actually conform to `Trait`, a compilation
    error would occur.

    Parameters:
        T: The original type.
        Trait: The trait to downcast into.

    Args:
        src: The value to downcast.

    Returns:
        The downcasted value.
    """
    return rebind[downcast[Trait, T]](src)


@always_inline
fn trait_downcast[
    T: AnyType, //, Trait: type_of(AnyType)
](ref src: T) -> ref [src] downcast[Trait, T]:
    """Downcast a parameter input type `T` and rebind the type such that the
    return value's type conforms the provided `Trait`. If `T`, after resolving
    to a concrete type, does not actually conform to `Trait`, a compilation
    error would occur.

    Parameters:
        T: The original type.
        Trait: The trait to downcast into.

    Args:
        src: The value to downcast.

    Returns:
        The downcasted value.
    """
    return rebind[downcast[Trait, T]](src)
