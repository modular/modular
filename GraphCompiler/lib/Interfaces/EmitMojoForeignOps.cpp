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

#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoTraits.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/POPDialect/POPOps.h"

using namespace M;

using namespace M;

namespace M::MOGG {

//===----------------------------------------------------------------------===//
// index dialect implementations
//===----------------------------------------------------------------------===//

LogicalResult emitMojo(mlir::index::AddOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] + operands[1]));
  return success();
}

LogicalResult emitMojo(mlir::index::CastSOp op, EmitMojoState &state) {
  // TODO: Verify. This op exists in MGP only because pop ops need pop types,
  // not index, so keeping Int is likely fine for MGP's shape-function index
  // math.
  state.setMojoValueFor(op.getResult(), state.getMojoValueFor(op.getInput()));
  return success();
}

LogicalResult emitMojo(mlir::index::CmpOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  MojoValue replacementValue;
  switch (op.getPred()) {
  case mlir::index::IndexCmpPredicate::EQ:
    replacementValue = operands[0].eq(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::NE:
    replacementValue = operands[0].neq(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::SGE:
    replacementValue = operands[0].ge(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::SGT:
    replacementValue = operands[0].gt(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::SLE:
    replacementValue = operands[0].le(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::SLT:
    replacementValue = operands[0].lt(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::UGE:
    replacementValue = operands[0].ge(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::UGT:
    replacementValue = operands[0].gt(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::ULE:
    replacementValue = operands[0].le(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::ULT:
    replacementValue = operands[0].lt(operands[1]);
    break;
  }
  state.setMojoValueFor(op.getResult(), state.builder.var(replacementValue));

  return success();
}

LogicalResult emitMojo(mlir::index::ConstantOp op, EmitMojoState &state) {
  state.setMojoValueFor(op.getResult(), op.getValue().getSExtValue());
  return success();
}

LogicalResult emitMojo(mlir::index::BoolConstantOp op, EmitMojoState &state) {
  state.setMojoValueFor(op.getResult(), op.getValue());
  return success();
}

LogicalResult emitMojo(mlir::index::DivSOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0].idiv(operands[1])));
  return success();
}

LogicalResult emitMojo(mlir::index::DivUOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0].idiv(operands[1])));
  return success();
}

LogicalResult emitMojo(mlir::index::CeilDivSOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0].sdiv(operands[1])));
  return success();
}

LogicalResult emitMojo(mlir::index::CeilDivUOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0].sdiv(operands[1])));
  return success();
}

LogicalResult emitMojo(mlir::index::MinSOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  static const MojoValue minFunc = "min";
  state.setMojoValueFor(op.getResult(), state.builder.var(minFunc(
                                            {}, {operands[0], operands[1]})));
  return success();
}

LogicalResult emitMojo(mlir::index::MaxSOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  static const MojoValue maxFunc = kFnMax;
  state.setMojoValueFor(op.getResult(), state.builder.var(maxFunc(
                                            {}, {operands[0], operands[1]})));
  return success();
}

LogicalResult emitMojo(mlir::index::MulOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] * operands[1]));
  return success();
}

LogicalResult emitMojo(mlir::index::RemSOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] % operands[1]));
  return success();
}

LogicalResult emitMojo(mlir::index::RemUOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] % operands[1]));
  return success();
}

//===----------------------------------------------------------------------===//
// POP dialect implementations
//===----------------------------------------------------------------------===//

LogicalResult emitMojo(KGEN::POP::AndOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(
      op.getResult(),
      state.builder.var(operands[0].bitwise_logical_and(operands[1])));
  return success();
}

LogicalResult emitMojo(KGEN::POP::AddOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] + operands[1]));
  return success();
}

LogicalResult emitMojo(KGEN::POP::MulOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] * operands[1]));
  return success();
}

LogicalResult emitMojo(KGEN::POP::DivOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  if (op.getType().isFloat()) {
    state.setMojoValueFor(op.getResult(),
                          state.builder.var(operands[0].div(operands[1])));
  } else {
    state.setMojoValueFor(op.getResult(),
                          state.builder.var(operands[0].idiv(operands[1])));
  }
  return success();
}

LogicalResult emitMojo(KGEN::POP::CastFromBuiltinOp op, EmitMojoState &state) {
  state.setMojoValueFor(op.getResult(), state.getMojoValueFor(op.getInput()));
  return success();
}

LogicalResult emitMojo(KGEN::POP::CastToBuiltinOp op, EmitMojoState &state) {
  state.setMojoValueFor(op.getResult(), state.getMojoValueFor(op.getInput()));
  return success();
}

LogicalResult emitMojo(KGEN::POP::CmpOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  MojoValue replacementValue;
  switch (op.getPred()) {
  case KGEN::CmpPredicate::EQ:
    replacementValue = operands[0].eq(operands[1]);
    break;
  case KGEN::CmpPredicate::NE:
    replacementValue = operands[0].neq(operands[1]);
    break;
  case KGEN::CmpPredicate::GE:
    replacementValue = operands[0].ge(operands[1]);
    break;
  case KGEN::CmpPredicate::GT:
    replacementValue = operands[0].gt(operands[1]);
    break;
  case KGEN::CmpPredicate::LE:
    replacementValue = operands[0].le(operands[1]);
    break;
  case KGEN::CmpPredicate::LT:
    replacementValue = operands[0].lt(operands[1]);
    break;
  }
  state.setMojoValueFor(op.getResult(), state.builder.var(replacementValue));

  return success();
}

LogicalResult emitMojo(KGEN::POP::RemOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] % operands[1]));
  return success();
}

LogicalResult emitMojo(KGEN::POP::SelectOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  auto primDecl = state.registry.lookupPrimitive(op.getOperationName());
  // sym_name on PrimitiveDeclOp is the fully-qualified Mojo source name.
  MojoValue selectFunc(primDecl.getSymNameAttr().strref());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(selectFunc({}, operands)));
  return success();
}

LogicalResult emitMojo(KGEN::POP::ShlOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] << operands[1]));
  return success();
}

LogicalResult emitMojo(KGEN::POP::ShrOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] >> operands[1]));
  return success();
}

LogicalResult emitMojo(KGEN::POP::MinOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  static const MojoValue minFunc = "min";
  state.setMojoValueFor(op.getResult(), state.builder.var(minFunc(
                                            {}, {operands[0], operands[1]})));
  return success();
}

LogicalResult emitMojo(KGEN::POP::MaxOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  static const MojoValue maxFunc = kFnMax;
  state.setMojoValueFor(op.getResult(), state.builder.var(maxFunc(
                                            {}, {operands[0], operands[1]})));
  return success();
}

LogicalResult emitMojo(KGEN::POP::OrOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(
      op.getResult(),
      state.builder.var(operands[0].bitwise_logical_or(operands[1])));
  return success();
}

LogicalResult emitMojo(KGEN::POP::XOrOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(
      op.getResult(),
      state.builder.var(operands[0].bitwise_logical_xor(operands[1])));
  return success();
}

LogicalResult emitMojo(KGEN::POP::SIMDAndOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(
      op.getResult(),
      state.builder.var(operands[0].bitwise_logical_and(operands[1])));
  return success();
}

LogicalResult emitMojo(KGEN::POP::SIMDSelectOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  auto primDecl = state.registry.lookupPrimitive(op.getOperationName());
  // sym_name on PrimitiveDeclOp is the fully-qualified Mojo source name.
  MojoValue selectFunc(primDecl.getSymNameAttr().strref());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(selectFunc({}, operands)));
  return success();
}

LogicalResult emitMojo(KGEN::POP::SIMDOrOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(
      op.getResult(),
      state.builder.var(operands[0].bitwise_logical_or(operands[1])));
  return success();
}

LogicalResult emitMojo(KGEN::POP::SIMDXOrOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(
      op.getResult(),
      state.builder.var(operands[0].bitwise_logical_xor(operands[1])));
  return success();
}

//===----------------------------------------------------------------------===//
// KGEN::ParamConstantOp
//===----------------------------------------------------------------------===//

LogicalResult emitMojo(KGEN::ParamConstantOp op, EmitMojoState &state) {
  if (auto vals = dyn_cast<KGEN::SIMDAttr>(op.getValue())) {
    ASSERT_STREAM(vals, << "Expected a SIMD parameter constant");
    ASSERT_STREAM(vals.getValues().size() == 1,
                  << "Only scalars are supported");
    KGEN::DTypeValue val = vals.getValues().front();
    KGEN::KGENDType valDType = val.getDType();
    if (valDType.isBool())
      state.setMojoValueFor(op.getResult(), val.getBoolVal());
    else
      state.setMojoValueFor(op.getResult(), val.getIntVal());
  } else if (auto val = dyn_cast<IntegerAttr>(op.getValue())) {
    auto apInt = val.getValue();
    if (val.getValue().getBitWidth() == 1) {
      state.setMojoValueFor(op.getResult(), apInt.getBoolValue());
    } else {
      state.setMojoValueFor(
          op.getResult(),
          APSInt(apInt, cast<IntegerType>(val.getType()).isSigned()));
    }
  } else {
    ASSERT_STREAM(false, << "Unsupported param constant type "
                         << op.getValue().getType());
    llvm_unreachable("Unsupported mlir type");
  }

  return success();
}
} // namespace M::MOGG
