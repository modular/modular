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
"""Decorators for registering MAX Graph kernels.

`@register` is the public decorator for DPS-style custom kernels.
`@register_internal` is used by built-in MAX Graph operations.
"""

from .platform_predicates import PlatformExpr, PlatformPredicate


def register_internal(name: StaticString):
    """
    Registers the given mojo function as an implementation of an `mo` op or a
    `mo.custom` op. Used by built-in
    [MAX Graph operations](/api/python/graph.ops).

    For registering [custom operations](/develop/custom-ops/), use the
    [@extensibility.register](/api/mojo/extensibility/decorators/register/) decorator,
    instead.

    For instance:

    ```mojo
    @register_internal("mo.add")
    def my_op[...](...):
      ...
    ```

    Registers `my_op` as an implementation of `mo.add`.

    Args:
      name: The name of the op to register.
    """
    return


def register(name: StaticString):
    """Registers a struct as a kernel implementation of a MAX Graph op for all
    devices.

    ```mojo
    @extensibility.register("mo.matmul")
    struct Matmul:
        ...
    ```

    To restrict a registration to some devices, use the `cond` overload of
    `register` with a platform.

    Args:
        name: The MAX Graph op name to register (e.g. `"mo.matmul"`).
    """
    pass


def register[
    expr: PlatformExpr, //
](name: StaticString, cond: PlatformPredicate[expr], /):
    """Registers a struct as a kernel implementation for the devices that
    `cond` selects.

    Build `cond` from the aliases in `extensibility.platforms` and the atoms in
    `extensibility.platform_predicates`. Give each platform that uses the
    `type`, `api`, `arch`, or `model` atoms a name with `comptime`, and pass
    that name as `cond`. In the alias, state every level of the device above
    the narrowest one that it tests, even when a helper already implies it:

    ```mojo
    comptime TCGEN05 = ACCELERATOR & CUDA & (arch("sm_100a") | arch("sm_103a"))

    @extensibility.register("mo.matmul", CPU)
    struct CpuMatmul:
        ...

    @extensibility.register("mo.matmul", ACCELERATOR)
    struct AcceleratorMatmul:
        ...

    @extensibility.register("mo.matmul", TCGEN05)
    struct TcgenMatmul:
        ...
    ```

    The Graph Compiler selects a kernel for each op and device as follows:

    1. It keeps the registrations whose platform holds for the device.
    2. If a user-defined registration remains, it keeps only the user-defined
       ones. Otherwise it keeps the built-in ones.
    3. It selects the registration whose set of devices is a strict subset of
       the set of every other remaining registration. In the example, a B200
       matches `AcceleratorMatmul` and `TcgenMatmul`, and `TcgenMatmul` wins.

    If no registration is a strict subset of all the others, the selection is
    ambiguous and the Graph Compiler reports an error. To resolve it, narrow a
    platform or register a kernel for exactly the devices that overlap. Two
    registrations of the same op for the same set of devices, both built-in or
    both user-defined, are an error, because neither can ever be selected.

    Parameters:
        expr: The expression of the platform, inferred from `cond`.

    Args:
        name: The MAX Graph op name to register (e.g. `"mo.matmul"`).
        cond: The devices that the kernel supports.
    """
    pass


def register_shape_function(name: StaticString):
    """Registers a standalone shape function for a MAX Graph op.

    The decorated function computes the output shapes of the named op on the
    host, without executing the op itself. Tensor arguments must use tensor
    traits: `Some[Tensor]` for shapes, `Some[TileTensorable]` for element values,
    or `TensorTuple` for variadic inputs. `ManagedTensorSlice` and
    `VariadicTensors` aliases are not accepted.

    Args:
        name: The name of the op whose output shapes the function computes.
    """
    pass
