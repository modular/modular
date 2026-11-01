//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This defines attributes in the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGENUMS_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGENUMS_TD

include "mlir/IR/EnumAttr.td"

def MOGG_BindLambdaKind : I32EnumAttr<"BindLambdaKind", "lambda kind", [
  I32EnumAttrCase<"InputLambda", 0, "input_lambda">,
  I32EnumAttrCase<"OutputLambda", 1, "output_lambda">,
  I32EnumAttrCase<"ElemwiseLambda", 2, "elemwise_lambda">
]> {
  let cppNamespace = "::M::MOGG";
}

// How a run of variadic tensor operands packed into a `mogg.tuple.create`
// must be materialized in Mojo. `ConformCallOperands` sets this from the
// callee's variadic parameter type, so `emitMojo` chooses the aggregate
// (`VariadicTensors` / `_FusedInputVariadicTensors` /
// `_FusedOutputVariadicTensors`, or a `TensorTuple` of trait arguments)
// without inspecting the producing ops.
def MOGG_VariadicABI
    : I32EnumAttr<"VariadicABI", "variadic tensor packing ABI", [
  I32EnumAttrCase<"Plain", 0, "plain">,
  I32EnumAttrCase<"FusedInput", 1, "fused_input">,
  I32EnumAttrCase<"FusedOutput", 2, "fused_output">,
  // A `TensorTuple` (`xs: TensorTuple[*Ts]`): the elements are already trait
  // arguments, and emit builds a `TensorTuple` holding them.
  I32EnumAttrCase<"TensorTuple", 3, "tensor_tuple">,
  // A slot-backed (`Stable`) run: emit packs `StableTensor` elements.
  I32EnumAttrCase<"Stable", 4, "stable">,
]> {
  let cppNamespace = "::M::MOGG";
  // Use `MOGG_VariadicABIAttr` (MOGGAttrs.td) rather than a bare `i32`.
  let genSpecializedAttr = 0;
}

// How the runtime schedules a routine. `mgp_generic_execute` reads it as the
// `eIsAsync` bool of `mgp.generic.execute`.
def MOGG_DispatchKind
    : I32EnumAttr<"DispatchKind", "how the runtime schedules a routine", [
  // Always runs out of line, as its own AsyncRT task.
  I32EnumAttrCase<"Deferred", 0, "deferred">,
  // Runs on the dispatching worker, unless task affinity moves it.
  I32EnumAttrCase<"Immediate", 1, "immediate">,
]> {
  let cppNamespace = "::M::MOGG";
  // Use `MOGG_DispatchKindAttr` (MOGGAttrs.td) rather than a bare `i32`.
  let genSpecializedAttr = 0;
}

def MOGG_ABIKind : I32EnumAttr<"ABIKind", "ABI kind", [
  I32EnumAttrCase<"MGP", 1, "MGP">,
  // A `mogg.func` outlined from a `mogg.routine` (advanced fusion), so
  // `EmitMojo` can branch on it separately from MGP emission.
  I32EnumAttrCase<"MOGG", 2, "MOGG">,
]> {
  let cppNamespace = "::M::MOGG";
  // Use `MOGG_ABIKindAttr` (MOGGAttrs.td) rather than a bare `i32`.
  let genSpecializedAttr = 0;
}

// Role a trait-bound tensor argument plays in the graph signature. Kept in
// lockstep with `TensorArgRole` in KernelDeclAdaptor.h (named `*Kind` to avoid
// colliding with it; `tensorArgContractToAttr` maps between them) and the role
// traits in `extensibility.tensor_arg_traits`. An argument is mutable iff its
// role is not `input`. `none` is for shape-function tensor arguments, which
// read only metadata.
def MOGG_TensorArgRole
    : I32EnumAttr<"TensorArgRoleKind", "tensor-argument role", [
        I32EnumAttrCase<"input", 0, "input">,
        I32EnumAttrCase<"output", 1, "output">,
        I32EnumAttrCase<"mutable_input", 2, "mutable_input">,
        I32EnumAttrCase<"none", 3, "none">,
      ]> {
  let cppNamespace = "::M::MOGG";
}

// Set of element-access forms a trait-bound tensor argument exposes (`none`
// for a plain dense argument). Mirrors the access/capability bitmasks of
// `TensorArgContract` in KernelDeclAdaptor.h. A non-fused dense argument can
// carry several (e.g. `load | store` for read-modify-write); a fused argument
// carries exactly one.
def MOGG_TensorArgAccess
    : I32BitEnumAttr<"TensorArgAccessKind", "tensor-argument access form set", [
        I32BitEnumAttrCaseNone<"none">,
        I32BitEnumAttrCaseBit<"load", 0, "load">,
        I32BitEnumAttrCaseBit<"store", 1, "store">,
        I32BitEnumAttrCaseBit<"transform", 2, "transform">,
      ]> {
  let cppNamespace = "::M::MOGG";
  let genSpecializedAttr = 0;
}

// The deepest trait bound written on a shape-function tensor argument (see
// `TensorArgBound` in KernelDeclAdaptor.h). Execute-lane contracts always
// carry the default `unspecified`, so printing and the baked
// `tensor_arg_contracts` dict elide it.
def MOGG_TensorArgBound
    : I32EnumAttr<"TensorArgBoundKind", "tensor-argument written bound", [
        I32EnumAttrCase<"unspecified", 0, "unspecified">,
        I32EnumAttrCase<"tensor", 1, "tensor">,
        I32EnumAttrCase<"tile", 2, "tile">,
      ]> {
  let cppNamespace = "::M::MOGG";
}

// Which tile fusion trait a `mogg.tile.create.epilogue` body implements. The
// kernel's declared output kind fixes it: a `_FusedComputeOutputTileTensor`
// takes a compute lane, which yields the tile for the kernel to store, and a
// `_FusedOutputTileTensor` takes a store lane, which owns the store.
def MOGG_TileFusionLane
    : I32EnumAttr<"TileFusionLane", "tile fusion lane", [
  I32EnumAttrCase<"Compute", 0, "compute">,
  I32EnumAttrCase<"Store", 1, "store">,
]> {
  let cppNamespace = "::M::MOGG";
  // Use `MOGG_TileFusionLaneAttr` (MOGGAttrs.td) rather than a bare `i32`.
  let genSpecializedAttr = 0;
}

// A function a Mojo kernel registers: one of its struct's methods, or its
// shape function. Each name is the Mojo method's, except that the TileTensor
// `elementwise` overload is "tile_elementwise" to tell the two apart.
def MOGG_KernelFunctionKind
    : I32Enum<"KernelFunctionKind", "kernel function kind", [
        I32EnumCase<"Execute", 0, "execute">,
        I32EnumCase<"Shape", 1, "shape">,
        I32EnumCase<"Elementwise", 2, "elementwise">,
        I32EnumCase<"TileElementwise", 3, "tile_elementwise">,
        I32EnumCase<"BuildGraph", 4, "build_graph">,
        I32EnumCase<"ExecuteDevice", 5, "execute_device">,
        I32EnumCase<"ExecuteConfiguration", 6, "execute_configuration">,
      ]> {
  let cppNamespace = "::M::MOGG";
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGENUMS_TD
