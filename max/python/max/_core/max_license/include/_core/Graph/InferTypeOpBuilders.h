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

#ifndef SDK_GRAPHAPI_PYTHON_INFER_TYPE_OP_BUILDERS_H
#define SDK_GRAPHAPI_PYTHON_INFER_TYPE_OP_BUILDERS_H

#include "_core/Graph/Checked.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "nanobind/nanobind.h"
#include "llvm/ADT/SmallVector.h"

// C++ machinery for binding the InferTypeOp builder variants that
// `mlir-tblgen` emits for ops with `InferTypeOpAdaptor`. There are two
// shapes — *separate-arg* (one parameter per ODS arg) and *collective*
// (bundled operands + attributes/properties) — and each comes in two
// flavours:
//
// clang-format off
//   separate args, wrapped attributes         → bind_infer_type_op<Op>(pop)
//   separate args, unwrapped scalar attrs     → bind_infer_type_op_unwrapped<Op>(pop)
//   collective (operands, attributes)         → create_collective_attrdict<Op>
//   collective (operands, props, discardable) → create_collective_propstruct<Op>
// clang-format on
//
// Python codegen in `nanobind.py` references these helpers by name; the
// per-op work is limited to the separate-arg state-population body — a
// small lambda passed to `bind_infer_type_op{,_unwrapped}`. The
// collective helpers are op-agnostic.
//
// All four end at `finalize_checked<Op>`, which runs `Op::inferReturnTypes`
// and raises a Python `ValueError` on failure (instead of the
// `report_fatal_error` the default ODS-generated overload calls).

namespace M::Graph::Python {

/// Tail half of an `InferTypeOp::create` call: runs `inferReturnTypes`
/// against the populated `OperationState`, raises a Python `ValueError`
/// on failure (instead of `report_fatal_error`), and creates the op.
///
/// The caller (codegen'd in `nanobind.py`, or one of the helpers
/// below) is responsible for populating `state` exactly the way the
/// ODS-generated build would: adding operands, setting Properties
/// fields, etc. — mirroring `genCodeForAddingArgAndRegionForBuilder`
/// in `mlir-tblgen`. We then run the same `inferReturnTypes` call the
/// ODS InferTypeOp builder uses (reading attributes through
/// `state.getRawProperties()`), so any op that has `InferTypeOpAdaptor`
/// is supported without per-op C++ glue.
template <typename Op>
Op finalize_checked(mlir::OpBuilder &builder, mlir::OperationState &state) {
  auto *ctx = builder.getContext();
  llvm::SmallVector<mlir::Type, 2> inferredReturnTypes;
  if (mlir::failed(Op::inferReturnTypes(ctx, state.location, state.operands,
                                        state.attributes.getDictionary(ctx),
                                        state.getRawProperties(), state.regions,
                                        inferredReturnTypes))) {
    PyErr_Format(PyExc_ValueError, "Failed to infer result type(s) for '%s'",
                 Op::getOperationName().str().c_str());
    throw nanobind::python_error();
  }
  state.addTypes(inferredReturnTypes);
  return mlir::cast<Op>(builder.create(state));
}

/// Tail for the **collective InferTypeOp** builder
/// (`(ValueRange operands, DictionaryAttr attributes)`): syncs the
/// state's attribute dict into the op's `Properties` so
/// `inferReturnTypes` reads the same data as ops built by the
/// per-arg `build()` overloads, then defers to `finalize_checked`.
///
/// Mirrors the tail of `genInferredTypeCollectiveParamBuilder` in
/// `mlir-tblgen` (which calls `report_fatal_error` on conversion
/// failure — we raise a Python `TypeError` instead, via
/// `RaisingScopedEmitFn`).
template <typename Op>
Op finalize_checked_collective(mlir::OpBuilder &builder,
                               mlir::OperationState &state) {
  if (!state.attributes.empty()) {
    (void)state.getOrAddProperties<typename Op::Properties>();
    mlir::PropertyRef props = state.getRawProperties();
    auto info = state.name.getRegisteredInfo();
    if (info) {
      RaisingScopedEmitFn emit;
      (void)info->setOpPropertiesFromAttribute(
          state.name, props, state.attributes.getDictionary(state.getContext()),
          emit);
    }
  }
  return finalize_checked<Op>(builder, state);
}

/// Lift a per-op "state populator" into a Python builder binding.
///
/// The codegen for InferTypeOp builders only differs op-to-op in how
/// each argument is dropped into `OperationState` — operand → addOperands,
/// attr → Properties-field assignment, etc. Everything else (cast the
/// builder handle, construct `OperationState`, run `inferReturnTypes`,
/// dyn_cast the result) is the same. Hand that boilerplate to this
/// template and let `nanobind.py` codegen just the populator body.
///
/// Use `+[](...){}` (the unary `+` forces a non-capturing lambda to decay
/// to a function pointer) so the populator type matches `void(*)(...)`.
template <typename Op, typename... Args>
auto bind_infer_type_op(void (*pop)(mlir::OperationState &, Args...)) {
  return [pop](nanobind::handle_t<mlir::OpBuilder> builder_handle,
               mlir::Location location, Args... args) {
    auto &builder = *nanobind::cast<mlir::OpBuilder *>(builder_handle);
    mlir::OperationState state(location, Op::getOperationName());
    pop(state, std::forward<Args>(args)...);
    return finalize_checked<Op>(builder, state);
  };
}

/// Same as `bind_infer_type_op` but also threads a `mlir::Builder` into
/// the populator, for the **unwrapped-attr** variant whose body uses
/// `odsBuilder.getXxxAttr(rawValue)` (the per-attr `constBuilderCall`
/// template) to wrap raw scalars into the storage `Attribute` type.
template <typename Op, typename... Args>
auto bind_infer_type_op_unwrapped(void (*pop)(mlir::OperationState &,
                                              mlir::Builder &, Args...)) {
  return [pop](nanobind::handle_t<mlir::OpBuilder> builder_handle,
               mlir::Location location, Args... args) {
    auto &builder = *nanobind::cast<mlir::OpBuilder *>(builder_handle);
    mlir::Builder odsBuilder(builder.getContext());
    mlir::OperationState state(location, Op::getOperationName());
    pop(state, odsBuilder, std::forward<Args>(args)...);
    return finalize_checked<Op>(builder, state);
  };
}

/// Fully generic binding for the collective `(operands, attributes)`
/// form of an InferTypeOp builder. The body has no per-op pieces; bind
/// it directly with `nb::new_(&create_collective_attrdict<Op>)`.
template <typename Op>
Op create_collective_attrdict(
    nanobind::handle_t<mlir::OpBuilder> builder_handle, mlir::Location location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes) {
  auto &builder = *nanobind::cast<mlir::OpBuilder *>(builder_handle);
  mlir::OperationState state(location, Op::getOperationName());
  state.addOperands(operands);
  if (attributes)
    state.addAttributes(attributes.getValue());
  return finalize_checked_collective<Op>(builder, state);
}

/// Fully generic binding for the collective
/// `(operands, properties, discardable_attrs)` form. Mirrors
/// `genInferredTypeCollectiveParamBuilder(PropStruct)`, with `properties`
/// arriving as a `DictionaryAttr` (Python's natural carrier for the
/// typed `Op::Properties` struct, which isn't bound) and converted via
/// `Op::setPropertiesFromAttr`.
template <typename Op>
Op create_collective_propstruct(
    nanobind::handle_t<mlir::OpBuilder> builder_handle, mlir::Location location,
    mlir::ValueRange operands, mlir::DictionaryAttr properties,
    mlir::DictionaryAttr discardable_attributes) {
  auto &builder = *nanobind::cast<mlir::OpBuilder *>(builder_handle);
  typename Op::Properties props{};
  if (properties) {
    RaisingScopedEmitFn emit;
    (void)Op::setPropertiesFromAttr(props, properties, emit);
  }
  mlir::OperationState state(location, Op::getOperationName());
  state.addOperands(operands);
  state.useProperties(props);
  if (discardable_attributes)
    state.addAttributes(discardable_attributes.getValue());
  return finalize_checked<Op>(builder, state);
}

} // namespace M::Graph::Python

#endif // SDK_GRAPHAPI_PYTHON_INFER_TYPE_OP_BUILDERS_H
