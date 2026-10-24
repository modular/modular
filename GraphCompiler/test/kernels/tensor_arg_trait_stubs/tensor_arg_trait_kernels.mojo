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
# Trait-bound kernel stubs for testing the tensor-argument trait import path
# (distilling a written trait composition into a `{role, fused, access,
# capabilities}` contract). Kept separate from testing_kernels, which feeds
# the full emit/lowering/execution paths -- these stubs only exercise import and
# have no-op bodies, so they must not be mixed into the general kernel package.
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
    LoadAccess,
    Loadable,
    MutableInput,
    Output,
    OutputTensor,
    Stable,
    StoreAccess,
    Storable,
    Tensor,
    TensorTuple,
    TileTensorable,
    _LegacyComputeOutput,
    _MutableInputTensor as MutableInputTensor,
)


# Dense add, written with explicit trait-bound generic type parameters.
@extensibility.register("trait_add_explicit")
struct TraitAddExplicit:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, y: X, ctx: DeviceContext) raises:
        pass


# Same contract as `TraitAddExplicit`, written with the `Some[...]` infer-only
# sugar. Must distill to an identical contract per argument.
@extensibility.register("trait_add_some")
struct TraitAddSome:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & TileTensorable],
        y: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        pass


# A fused, loadable input alongside a dense output -- exercises the fused-load
# access form (`Input & Fused & Loadable`).
@extensibility.register("trait_fused_load")
struct TraitFusedLoad:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & Fused & Loadable],
        ctx: DeviceContext,
    ) raises:
        pass


# Negative fixture: a trait-bound tensor arg passed with a non-borrowed (`mut`)
# convention. A tensor arg's mutability belongs in its contract (the role), not
# the calling convention, so VerifyKernels must reject this. It imports fine;
# only `mogg-verify-kernels` flags it.
@extensibility.register("trait_bad_convention")
struct TraitBadConvention:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
    ](mut z: Z, ctx: DeviceContext) raises:
        pass


# Negative fixture (off-by-one guard for the trait exclusion in the legacy
# borrow-check): a signature with a trait arg at position 0 (borrowed, fine) AND
# a legacy `ManagedTensorSlice` (`OutputTensor`) at position 1 passed with a
# non-borrowed (`mut`) convention. The legacy borrow-check must STILL fire for
# the legacy arg at position 1; the `!getTensorArgTraitType(...)` exclusion must
# skip ONLY the trait arg at position 0. An off-by-one there would silently
# suppress the legacy diagnostic.
@extensibility.register("trait_legacy_bad_convention")
struct TraitLegacyBadConvention:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        x: Some[Input & TileTensorable],
        mut z: OutputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


# Two dense trait outputs followed by a dense trait input -- exercises the
# multi-output DPS split: both leading outputs must be classified as DPS
# outputs (not folded into the inputs).
@extensibility.register("trait_multi_output")
struct TraitMultiOutput:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        a: Some[Output & TileTensorable],
        b: Some[Output & TileTensorable],
        x: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        pass


# A dense output and a `MutableInput` trait argument -- the mutable-input role
# must classify as a mutable (writable, non-DPS-output) tensor, distinct from a
# read-only `Input`.
@extensibility.register("trait_mutable_input")
struct TraitMutableInput:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[MutableInput & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        pass


# Negative fixture: two `TensorTuple`s on the input side. Each distills its own
# contract from its `TypeList` bound, so import bakes all three; only
# `mogg-verify-kernels` flags it, because a graph op's flat operand list cannot
# carry two runs.
@extensibility.register("trait_tensor_tuple")
struct TraitTensorTuple:
    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & TileTensorable, ...],
        As: TypeList[Trait=Input & TileTensorable, ...],
        Bs: TypeList[Trait=Input & TileTensorable, ...],
        //,
        target: StaticString,
    ](
        outputs: TensorTuple[*Outs],
        a: TensorTuple[*As],
        b: TensorTuple[*Bs],
        ctx: DeviceContext,
    ) raises:
        pass


# A fused `TensorTuple`: the tuple argument distills one fused contract from its
# list's bound.
@extensibility.register("trait_tensor_tuple_fused")
struct TraitTensorTupleFused:
    @staticmethod
    def execute[
        Ins: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        inputs: TensorTuple[*Ins],
        ctx: DeviceContext,
    ) raises:
        pass


# Mixed lane: a trait-bound dense output alongside legacy `ManagedTensorSlice`
# (`InputTensor`) inputs. The trait output and the legacy inputs must classify
# independently and coexist -- the legacy lane is unchanged.
@extensibility.register("trait_mixed_legacy")
struct TraitMixedLegacy:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: InputTensor,
        y: InputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


# Mirror of `trait_mixed_legacy` with a legacy `OutputTensor` and trait inputs:
# the legacy output classifies as the DPS output, the trait inputs as borrowed
# inputs.
@extensibility.register("trait_mixed_legacy_out")
struct TraitMixedLegacyOut:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: Some[Input & TileTensorable],
        y: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        pass


# D6 parity: explicit-generic spellings of the multi-output / mutable-input /
# fused-load roles. Each must classify identically to its `Some[...]` twin
# above, proving the explicit and inferred spellings agree for ALL roles, not
# just the dense add pair.
@extensibility.register("trait_multi_output_explicit")
struct TraitMultiOutputExplicit:
    @staticmethod
    def execute[
        target: StaticString,
        A: Output & TileTensorable,
        X: Input & TileTensorable,
    ](a: A, b: A, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("trait_mutable_input_explicit")
struct TraitMutableInputExplicit:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: MutableInput & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("trait_fused_load_explicit")
struct TraitFusedLoadExplicit:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & Fused & Loadable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


# Leading mutable_input: the FIRST tensor arg is a (writable, non-output) input,
# so `countDPSOutputs` must break at position 0 and yield ZERO DPS outputs --
# directly exercising the "no leading output" boundary for trait args.
@extensibility.register("trait_leading_mutable")
struct TraitLeadingMutable:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        x: Some[MutableInput & TileTensorable],
        y: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        pass


# All-outputs: three leading trait outputs and nothing else, so the leading-
# output loop runs to `n` without ever breaking -- covers count > 2 and the
# run-to-end path.
@extensibility.register("trait_all_outputs")
struct TraitAllOutputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        a: Some[Output & TileTensorable],
        b: Some[Output & TileTensorable],
        c: Some[Output & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        pass


# Non-leading output: input, output, input. `countDPSOutputs` must break at the
# FIRST input (position 0) and NOT count the later output as a DPS output --
# mirroring the legacy struct lane's "break at first non-output" rule.
@extensibility.register("trait_non_leading_output")
struct TraitNonLeadingOutput:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        x: Some[Input & TileTensorable],
        z: Some[Output & TileTensorable],
        y: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        pass


# Mixed lane with a legacy MUTABLE arg: a legacy `MutableInputTensor`
# (`mut=true`, non-DPS-output input) coexists with a trait output. The legacy
# mutable arg must still render `[in] mut=true` and the trait output `[out]`.
@extensibility.register("trait_mixed_legacy_mut")
struct TraitMixedLegacyMut:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: MutableInputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


# Mixed lane with a FUSED trait arg: a legacy `InputTensor` coexists with a
# trait `Input & Fused & Loadable`. Confirms the two lanes coexist when a fused
# trait arg (not just a dense one) is present.
@extensibility.register("trait_mixed_legacy_fused")
struct TraitMixedLegacyFused:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: InputTensor,
        y: Some[Input & Fused & Loadable],
        ctx: DeviceContext,
    ) raises:
        pass


# A trait-bound fused output with a legacy input. This combination exercises
# contract conflicts when epilogue fusion captures the input.
@extensibility.register("trait_fused_store_legacy_input")
struct TraitFusedStoreLegacyInput:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & StoreAccess],
        x: InputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("trait_dense_and_fused_inputs")
struct TraitDenseAndFusedInputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & TileTensorable],
        y: Some[Input & Fused & Loadable],
        ctx: DeviceContext,
    ) raises:
        pass


# Pure-legacy control: no trait args at all. A regression guard that this change
# leaves the legacy `[out]`/`[in]`/`mut` split byte-for-byte unchanged. The
# `MutableInputTensor` here pins `mut=true` on the legacy lane too.
@extensibility.register("legacy_only_control")
struct LegacyOnlyControl:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: InputTensor,
        w: MutableInputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


# ===----------------------------------------------------------------------=== #
# Distillation matrix (IMPORT-only): one kernel per remaining valid
# `{role, fused, access, capabilities}` contract the distiller produces. These
# back the `IMPORT-DAG` baked-contract assertions in infra-lower-tensor-arg-traits.mlir
# and replace the deleted gtest distill cases. The fused-store / fused-transform
# / mutable-input-fused kernels import (distill) cleanly even though dense-only
# lowering does not yet handle them, so they appear ONLY on the IMPORT run line,
# never as graph cases.
# ===----------------------------------------------------------------------=== #


# Plain `Tensor` metadata (no `TileTensorable`): an `Input` bound with only
# the base `Tensor` trait still distills to input/none -- the
# `Tensor`-vs-`TileTensorable` distinction is invisible to the contract.
@extensibility.register("trait_plain_tensor_input")
struct TraitPlainTensorInput:
    @staticmethod
    def execute[
        target: StaticString,
    ](x: Some[Input & Tensor], ctx: DeviceContext) raises:
        pass


# Fused load via the bare `LoadAccess` MARKER (no `Loadable` capability leaf):
# distills to input/load/fused with capabilities = 0 -- distinct from the
# `trait_fused_load` capability case.
@extensibility.register("trait_fused_load_marker")
struct TraitFusedLoadMarker:
    @staticmethod
    def execute[
        target: StaticString,
    ](x: Some[Input & Fused & LoadAccess], ctx: DeviceContext) raises:
        pass


# Fused store via the bare `StoreAccess` MARKER on an Output: output/store/fused,
# capabilities = 0.
@extensibility.register("trait_fused_store_marker")
struct TraitFusedStoreMarker:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[Output & Fused & StoreAccess], ctx: DeviceContext) raises:
        pass


# Fused store via the `Storable` CAPABILITY leaf on an Output:
# output/store/fused, capabilities = store.
@extensibility.register("trait_fused_store_capability")
struct TraitFusedStoreCapability:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[Output & Fused & Storable], ctx: DeviceContext) raises:
        pass


# Fused transform via the `_LegacyComputeOutput` slot on an Output:
# output/transform/fused, capabilities = 0.
@extensibility.register("trait_fused_transform_marker")
struct TraitFusedTransformMarker:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & _LegacyComputeOutput], ctx: DeviceContext
    ) raises:
        pass


# MutableInput can store (one access form at a time): MutableInput & Fused &
# Storable distills to mutable_input/store/fused/capability.
@extensibility.register("trait_mutable_fused_store")
struct TraitMutableFusedStore:
    @staticmethod
    def execute[
        target: StaticString,
    ](x: Some[MutableInput & Fused & Storable], ctx: DeviceContext) raises:
        pass


# MutableInput can also load: MutableInput & Fused & Loadable distills to
# mutable_input/load/fused/capability.
@extensibility.register("trait_mutable_fused_load")
struct TraitMutableFusedLoad:
    @staticmethod
    def execute[
        target: StaticString,
    ](x: Some[MutableInput & Fused & Loadable], ctx: DeviceContext) raises:
        pass


# Non-fused dense input that still advertises a load access form: Input &
# TileTensorable & Loadable distills to input/load with capabilities = load but
# fused = false (the access form is independent of the fusion marker).
@extensibility.register("trait_nonfused_load_capability")
struct TraitNonFusedLoadCapability:
    @staticmethod
    def execute[
        target: StaticString,
    ](x: Some[Input & TileTensorable & Loadable], ctx: DeviceContext) raises:
        pass


# Non-fused dense MutableInput advertising BOTH access forms at once:
# MutableInput & TileTensorable & Loadable & Storable distills to mutable_input
# with access = {load, store} and capabilities = {load, store}. This is the positive
# proof that a contract carries the full access SET (multiple forms on one
# argument), independent of any emit -- `load` on a read-only Input or `store` on
# an Output would be rejected by the per-form role/access compatibility check;
# MutableInput permits both. Emitting the dense element-access conformances is a
# later slice; here we only assert the distilled contract.
@extensibility.register("trait_nonfused_load_store_capability")
struct TraitNonFusedLoadStoreCapability:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        x: Some[MutableInput & TileTensorable & Loadable & Storable],
        ctx: DeviceContext,
    ) raises:
        pass


# The bare-`Tensor` bound: shape only, no layout/alignment. Distills to the
# no-role contract `{role: none, fused: false, access: none, bound: tensor}`.
# This is the positive counterpart to the shape-fn disallows in
# tensor_arg_trait_invalid_stubs (io-role / Fused / Loadable).
@extensibility.register_shape_function("trait_tensor_shape_fn")
def trait_tensor_shape_fn[T: Tensor](x: T) -> IndexList[T.rank]:
    return rebind[IndexList[T.rank]](coord_to_index_list(x.shape()))


# The `TileTensorable` bound: a shape function may read data as well as
# metadata, since it always executes on the host. Distills to the no-role
# contract with `bound: tile`.
@extensibility.register_shape_function("trait_tile_tensorable_shape_fn")
def trait_tile_tensorable_shape_fn[
    T: TileTensorable
](x: T) -> IndexList[T.rank]:
    _ = x.to_tile_tensor()
    return rebind[IndexList[T.rank]](coord_to_index_list(x.shape()))


# `Input & TileTensorable & Stable`: a slot-backed read-only input. Distills to
# the input contract with `stable = true`; the `Stable` storage marker is the
# only vocabulary row that sets it.
@extensibility.register("trait_stable_input")
struct TraitStableInput:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & TileTensorable & Stable],
        ctx: DeviceContext,
    ) raises:
        pass


# `MutableInput & TileTensorable & Stable`: a slot-backed in-place argument.
@extensibility.register("trait_stable_mutable_input")
struct TraitStableMutableInput:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        x: Some[MutableInput & TileTensorable & Stable], ctx: DeviceContext
    ) raises:
        pass


# `Input & Tensor & Stable`: the bound needn't name `TileTensorable`. The
# emitted stable argument conforms to it regardless, so the kernel refines in
# its body to project through the slot.
@extensibility.register("trait_stable_tensor_input")
struct TraitStableTensorInput:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & Tensor & Stable],
        ctx: DeviceContext,
    ) raises:
        pass
