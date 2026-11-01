# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
#
# Trait-bound kernel stubs whose signatures are INVALID -- each violates exactly
# one tensor-argument validation rule. They confirm that the import-time
# validation rejects every bad composition (and reports them all, rather than
# aborting on the first). Kept in a separate package from the valid stubs
# because an invalid composition fails the package import.
#
from std.collections.string import StaticString
from max.gpu.host import DeviceContext
from std.utils import IndexList

from layout import coord_to_index_list

import extensibility
from extensibility import (
    Fused,
    Input,
    InputTensor,
    InputVariadicTensors,
    ManagedTensorSlice,
    StableInputTensor,
    StableInputVariadicTensors,
    VariadicTensors,
    _FusedInputVariadicTensors as FusedInputVariadicTensors,
    _FusedOutputVariadicTensors as FusedOutputVariadicTensors,
    LoadAccess,
    Loadable,
    MutableInput,
    Output,
    Stable,
    StoreAccess,
    Storable,
    TensorTuple,
    TileTensorable,
    _LegacyComputable,
    _LegacyComputeOutput,
)


# No role trait (none of Input / Output / MutableInput).
@extensibility.register("trait_no_role")
struct TraitNoRole:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[TileTensorable], ctx: DeviceContext) raises:
        pass


# More than one role trait.
@extensibility.register("trait_two_roles")
struct TraitTwoRoles:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Input & Output & TileTensorable,
    ](z: Z, ctx: DeviceContext) raises:
        pass


# Fused but with no access form.
@extensibility.register("trait_fused_no_access")
struct TraitFusedNoAccess:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[Output & Fused & TileTensorable], ctx: DeviceContext) raises:
        pass


# A foreign (non-tensor-argument) trait in the bound -- `Copyable` is a stdlib
# trait, not part of extensibility.tensor_arg_traits.
@extensibility.register("trait_foreign")
struct TraitForeign:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable & Copyable,
    ](z: Z, ctx: DeviceContext) raises:
        pass


# A `TensorTuple` whose list bound carries `Movable`. Nothing requires it of the
# tuple's elements, so it is a foreign trait.
@extensibility.register("trait_tensor_tuple_movable")
struct TraitTensorTupleMovable:
    @staticmethod
    def execute[
        Ts: TypeList[Trait=Input & TileTensorable & Movable, ...],
        //,
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        xs: TensorTuple[*Ts],
        ctx: DeviceContext,
    ) raises:
        pass


# A variadic pack of tensor arguments, which `TensorTuple` replaces.
@extensibility.register("trait_variadic_pack")
struct TraitVariadicPack:
    @staticmethod
    def execute[
        target: StaticString,
        *Ts: Input & TileTensorable,
    ](z: Some[Output & TileTensorable], *xs: *Ts, ctx: DeviceContext) raises:
        pass


# Store access on a read-only Input.
@extensibility.register("trait_store_on_input")
struct TraitStoreOnInput:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[Input & Fused & Storable], ctx: DeviceContext) raises:
        pass


# Two role traits, specifically the `Input & MutableInput` pair (exact-leaf
# matching means `MutableInput` does not subsume `Input`; both roles count).
@extensibility.register("trait_input_and_mutable_input")
struct TraitInputAndMutableInput:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Input & MutableInput & TileTensorable,
    ](z: Z, ctx: DeviceContext) raises:
        pass


# A fused arg with more than one access form: a fused MutableInput carrying BOTH
# `Loadable` and `Storable` capability leaves. Multiple access forms are allowed
# on a NON-fused arg, but a fused arg must declare exactly one, so this is
# rejected.
@extensibility.register("trait_multi_access_forms")
struct TraitMultiAccessForms:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[MutableInput & Fused & Loadable & Storable],
        ctx: DeviceContext,
    ) raises:
        pass


# Two access MARKERS on a NON-fused input (`LoadAccess & StoreAccess`).
# Multiple access forms are now allowed, but `store` is still invalid on a
# read-only Input, so the per-form role/access check rejects it.
@extensibility.register("trait_multi_access_markers_nonfused")
struct TraitMultiAccessMarkersNonFused:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[Input & LoadAccess & StoreAccess], ctx: DeviceContext) raises:
        pass


# Load access on an Output: `Output & Fused & Loadable` -- load is not valid on
# an output.
@extensibility.register("trait_load_on_output")
struct TraitLoadOnOutput:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[Output & Fused & Loadable], ctx: DeviceContext) raises:
        pass


# Transform access on a read-only Input: `Input & Fused & _LegacyComputeOutput`
# -- transform requires an Output.
@extensibility.register("trait_transform_on_input")
struct TraitTransformOnInput:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[Input & Fused & _LegacyComputeOutput], ctx: DeviceContext) raises:
        pass


# Transform access on a MutableInput: `MutableInput & Fused &
# _LegacyComputeOutput` -- transform requires an Output, so a mutable input is
# also rejected.
@extensibility.register("trait_transform_on_mutable_input")
struct TraitTransformOnMutableInput:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[MutableInput & Fused & _LegacyComputeOutput],
        ctx: DeviceContext,
    ) raises:
        pass


# `_LegacyComputable` on the signature: `Output & Fused & _LegacyComputable`
# demands the compute capability, but the graph compiler may deliver the slot
# as a store fusion, so the kernel must declare `_LegacyComputeOutput` and
# refine in its body.
@extensibility.register("trait_computable_on_signature")
struct TraitComputableOnSignature:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[Output & Fused & _LegacyComputable], ctx: DeviceContext) raises:
        pass


# `TileTensorable` on a `_LegacyComputeOutput` signature:
# `Output & Fused & _LegacyComputeOutput & TileTensorable`. The store-fusion
# form the graph compiler may deliver doesn't conform to it, so the kernel
# must refine in its body.
@extensibility.register("trait_compute_output_tile_on_signature")
struct TraitComputeOutputTileOnSignature:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & _LegacyComputeOutput & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        pass


# A shape-function tensor argument carrying an io-role marker. Shape-fn args
# are read-only and must carry no role (conform to Tensor or TileTensorable
# instead), so an `Input` (or any io-role) is rejected in the
# shape-fn distill lane.
@extensibility.register_shape_function("trait_shape_fn_with_role")
def trait_shape_fn_with_role[
    T: Input & TileTensorable
](x: T) -> IndexList[T.rank]:
    return rebind[IndexList[T.rank]](coord_to_index_list(x.shape()))


# A shape-function tensor argument carrying `Fused`. Fusion is an `execute`-only
# concept; rejected on a shape-fn arg.
@extensibility.register_shape_function("trait_shape_fn_fused")
def trait_shape_fn_fused[T: Fused & TileTensorable](x: T) -> IndexList[T.rank]:
    return rebind[IndexList[T.rank]](coord_to_index_list(x.shape()))


# A shape-function tensor argument carrying `Loadable` (a fusion access
# capability). Fusion access is `execute`-only machinery; a shape-fn argument
# is limited to Tensor / TileTensorable, so this is rejected.
@extensibility.register_shape_function("trait_shape_fn_loadable")
def trait_shape_fn_loadable[
    T: Loadable & TileTensorable
](x: T) -> IndexList[T.rank]:
    return rebind[IndexList[T.rank]](coord_to_index_list(x.shape()))


# ========================================================================== #
# Stable tensor marker                                                       #
# ========================================================================== #


# `Output & TileTensorable & Stable`: outputs are written through a live
# pointer, never through a slot.
@extensibility.register("trait_stable_output")
struct TraitStableOutput:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[Output & TileTensorable & Stable], ctx: DeviceContext) raises:
        pass


# `Input & Fused & Loadable & Stable`: a slot is read through a `TileTensor`
# projection, not a fusion functor.
@extensibility.register("trait_stable_fused")
struct TraitStableFused:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & Fused & Loadable & Stable],
        ctx: DeviceContext,
    ) raises:
        pass


# A shape-function tensor argument carrying `Stable`: shape functions run on
# the host, where a slot cannot be read.
@extensibility.register_shape_function("trait_shape_fn_stable")
def trait_shape_fn_stable[
    T: Stable & TileTensorable
](x: T) -> IndexList[T.rank]:
    return rebind[IndexList[T.rank]](coord_to_index_list(x.shape()))


# Unbound IO parameters must not hide a legacy shape-function tensor argument.
@extensibility.register_shape_function("legacy_shape_tensor")
def legacy_shape_tensor(x: ManagedTensorSlice, y: InputTensor) -> IndexList[1]:
    return IndexList[1]()


@extensibility.register_shape_function("legacy_shape_variadic")
def legacy_shape_variadic(
    x: VariadicTensors, y: InputVariadicTensors
) -> IndexList[1]:
    return IndexList[1]()


@extensibility.register_shape_function("legacy_shape_fused_variadic")
def legacy_shape_fused_variadic(
    x: FusedInputVariadicTensors, y: FusedOutputVariadicTensors
) -> IndexList[1]:
    return IndexList[1]()


@extensibility.register_shape_function("legacy_shape_stable")
def legacy_shape_stable(
    x: StableInputTensor, y: StableInputVariadicTensors
) -> IndexList[1]:
    return IndexList[1]()
