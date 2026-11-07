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


def register(
    name: StaticString,
    type: StaticString = "",
    api: StaticString = "",
    arch: StaticString = "",
    model: StaticString = "",
):
    """Registers a struct as a kernel implementation for a MAX Graph op.

    The four device fields restrict the devices that the kernel supports:

    - `type`: broad device category: `"cpu"`, `"gpu"`, or a custom label.
    - `api`: compute backend: `"cuda"`, `"hip"`, `"metal"`, etc.
    - `arch`: microarchitecture: `"sm_100a"`, `"sm_90a"`, `"gfx942"`, etc.
    - `model`: exact hardware model: `"NVIDIA B200"`, `"AMD Instinct MI355X"`, etc.

    An empty string for any field acts as a wildcard: it matches any device
    value for that field. A registration applies to the devices whose fields
    equal every field that it sets.

    The Graph Compiler selects a kernel for each op and device as follows:

    1. It keeps the registrations that apply to the device.
    2. If a user-defined registration remains, it keeps only the user-defined
       ones. Otherwise it keeps the built-in ones.
    3. It selects the registration whose set of devices is a strict subset of
       the set of every other remaining registration.

    If no registration is a strict subset of all the others, the selection is
    ambiguous and the Graph Compiler reports an error. For example,
    `type="gpu", api="cuda"` and `arch="sm_90a"` both apply to an H100, but
    neither contains the other: `arch="sm_90a"` alone also applies to a device
    of any type and API with that architecture. Set every field above the
    narrowest one that you test, as in `type="gpu", api="cuda", arch="sm_90a"`.
    Two registrations of the same op with identical device fields, both
    built-in or both user-defined, are an error, because neither can ever be
    selected.

    Example: device field combinations and their reach:

    ```mojo
    # Matches all devices (backward-compatible default).
    @extensibility.register("mo.matmul")

    # Matches all CPU devices.
    @extensibility.register("mo.matmul", type="cpu")

    # Matches all GPU devices regardless of vendor or architecture.
    @extensibility.register("mo.matmul", type="gpu")

    # Matches all CUDA GPUs (any architecture).
    @extensibility.register("mo.matmul", type="gpu", api="cuda")

    # Matches only the NVIDIA SM100A architecture.
    @extensibility.register("mo.matmul", type="gpu", api="cuda", arch="sm_100a")

    # Matches only the NVIDIA B200.
    @extensibility.register("mo.matmul", type="gpu", api="cuda", arch="sm_100a", model="NVIDIA B200")
    ```

    Example: selection with multiple registrations in scope:

    ```mojo
    @extensibility.register("mo.matmul")                                          # all devices
    @extensibility.register("mo.matmul", type="cpu")                              # all CPUs
    @extensibility.register("mo.matmul", type="gpu")                              # all GPUs
    @extensibility.register("mo.matmul", type="gpu", api="cuda", arch="sm_100a")  # SM100A CUDA GPUs
    ```

    - SM100A CUDA GPU → the wildcard, `type="gpu"`, and `arch` registrations
      match; the `arch` registration wins, because its devices are a strict
      subset of the other two.
    - SM90A CUDA GPU  → wildcard and `type="gpu"` match; `type="gpu"` wins.
    - CPU             → wildcard and `type="cpu"` match; `type="cpu"` wins.
    - Some NPU        → only the wildcard matches; wildcard wins.

    Args:
        name: The MAX Graph op name to register (e.g. `"mo.matmul"`).
        type: Broad device category: `"cpu"`, `"gpu"`, or a custom accelerator
            label such as `"npu-xxx"`. Corresponds to the label used in
            `DeviceRef` (e.g. `DeviceRef.GPU()`). Empty matches all.
        api: Programming API or compute backend for the device, for example
            `"cuda"` (NVIDIA CUDA), `"hip"` (AMD ROCm/HIP), or `"metal"`
            (Apple Metal). Empty matches all.
        arch: Microarchitecture identifier, for example `"sm_90a"` (Hopper),
            `"sm_100a"` (Blackwell), or `"gfx942"` (AMD CDNA3). Empty matches
            all.
        model: Specific device model, for example `"NVIDIA B200"` or
            `"AMD Instinct MI355X"`. Empty matches all.
    """
    pass


def register[
    expr: PlatformExpr, //
](name: StaticString, cond: PlatformPredicate[expr], /):
    """Registers a struct as a kernel implementation for the devices that
    `cond` selects.

    The Graph Compiler does not accept this overload yet: importing a package
    that uses it fails.

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
