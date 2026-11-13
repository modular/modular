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

void EmitMojoState::emitOutputStoreFusionDefsOnce() {
  if (emittedOutputStoreFusionArgDefs)
    return;
  structDefsBuilder.verbatim(MojoValue(kOutputStoreFusionDefs.str()));
  emittedOutputStoreFusionArgDefs = true;
}

#define EMITS_MOJO_EXTERNALLY(OpTy)                                            \
  LogicalResult emitMojo(OpTy op, EmitMojoState &state);
#define EMIT_MOJO_EXTERNALLY(OpTy)                                             \
  if (auto asOp = dyn_cast<OpTy>(op)) {                                        \
    if (emitMojo(asOp, state).failed()) {                                      \
      return op->emitError()                                                   \
             << "'" << op->getName() << "' could not be lowered to Mojo";      \
    }                                                                          \
    return success();                                                          \
  }

EMITS_MOJO_EXTERNALLY(mlir::index::AddOp);
EMITS_MOJO_EXTERNALLY(mlir::index::CastSOp);
EMITS_MOJO_EXTERNALLY(mlir::index::CmpOp);
EMITS_MOJO_EXTERNALLY(mlir::index::ConstantOp);
EMITS_MOJO_EXTERNALLY(mlir::index::BoolConstantOp);
EMITS_MOJO_EXTERNALLY(mlir::index::DivSOp);
EMITS_MOJO_EXTERNALLY(mlir::index::DivUOp);
EMITS_MOJO_EXTERNALLY(mlir::index::CeilDivSOp);
EMITS_MOJO_EXTERNALLY(mlir::index::CeilDivUOp);
EMITS_MOJO_EXTERNALLY(mlir::index::MinSOp);
EMITS_MOJO_EXTERNALLY(mlir::index::MaxSOp);
EMITS_MOJO_EXTERNALLY(mlir::index::MulOp);
EMITS_MOJO_EXTERNALLY(mlir::index::RemSOp);
EMITS_MOJO_EXTERNALLY(mlir::index::RemUOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::AndOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::AddOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::MulOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::DivOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::CastFromBuiltinOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::CastToBuiltinOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::CmpOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::RemOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::SelectOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::ShlOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::ShrOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::MinOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::MaxOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::OrOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::XOrOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::SIMDAndOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::SIMDSelectOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::SIMDOrOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::SIMDXOrOp);
EMITS_MOJO_EXTERNALLY(KGEN::ParamConstantOp);

LogicalResult EmitMojoState::lowerOpToMojo(Operation *op) {
  if (auto emitMojoInterface = dyn_cast<MOGG::EmitsMojoInterface>(op)) {
    if (enableOpTraceComments)
      builder.comment(llvm::formatv("Emitting Mojo for: {0}", op->getName()));

    if (emitMojoInterface.emitMojo(*this).failed()) {
      return emitMojoInterface->emitError()
             << "'" << op->getName() << "' could not be lowered to Mojo";
    }
    return success();
  }

  auto &state = *this;

  EMIT_MOJO_EXTERNALLY(mlir::index::AddOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::CastSOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::CmpOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::ConstantOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::BoolConstantOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::DivSOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::DivUOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::CeilDivSOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::CeilDivUOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::MinSOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::MaxSOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::MulOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::RemSOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::RemUOp);

  EMIT_MOJO_EXTERNALLY(KGEN::POP::AndOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::AddOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::MulOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::DivOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::CastFromBuiltinOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::CastToBuiltinOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::CmpOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::RemOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::SelectOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::ShlOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::ShrOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::MinOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::MaxOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::OrOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::XOrOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::SIMDAndOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::SIMDSelectOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::SIMDOrOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::SIMDXOrOp);

  EMIT_MOJO_EXTERNALLY(KGEN::ParamConstantOp);

  return op->emitError()
         << "'" << op->getName()
         << "' must implement 'EmitsMojoInterface' to lower to Mojo; it is not "
            "a supported 'index' or 'pop' dialect operation either";
}

MojoValue EmitMojoState::getShapeAsIndexListMojoValue(MOSH::ShapeAttr shapeAttr,
                                                      ValueRange dynamicDims) {
  SmallVector<MojoValue> mojoDims;
  unsigned nextDymDim = 0;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (auto staticDim = MOSH::ShapeAttr::getStaticDim(dim)) {
      mojoDims.push_back(*staticDim);
    } else {
      ASSERT_STREAM(nextDymDim < dynamicDims.size(),
                    "Missing dynamic dimension");
      mojoDims.push_back(getMojoValueFor(dynamicDims[nextDymDim++]));
    }
  }
  if (mojoDims.empty())
    mojoDims.push_back(MojoValue(static_cast<int64_t>(1)));

  MojoValue rankParam =
      MojoValue(shapeAttr.getRank() == 0 ? 1 : shapeAttr.getRank());

  return MojoValue(kStructIndexList)({rankParam}, mojoDims);
}

MojoValue EmitMojoState::getShapeAsCoordMojoValue(MOSH::ShapeAttr shapeAttr,
                                                  ValueRange dynamicDims) {
  SmallVector<MojoValue> coordElems;
  unsigned nextDynDim = 0;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (auto staticDim = MOSH::ShapeAttr::getStaticDim(dim)) {
      coordElems.push_back(getComptimeIdx(*staticDim));
    } else {
      // Scalar[DType.int], not Int, to match what the ManagedTensorSlice
      // layout machinery rebinds to.
      ASSERT_STREAM(nextDynDim < dynamicDims.size(),
                    "Missing dynamic dimension");
      MojoValue scalarInt =
          MojoValue(kStructScalar)[MojoValue(kStructDType).get(kEnumDTypeInt)];
      coordElems.push_back(
          scalarInt({}, {getMojoValueFor(dynamicDims[nextDynDim++])}));
    }
  }
  // Rank-0 tensors are represented as rank-1 with a single static `1`.
  if (coordElems.empty())
    coordElems.push_back(getComptimeIdx(1));

  return MojoValue(kStructCoord)({}, coordElems);
}

void EmitMojoState::addDebugPrintTensorSpecs(const MojoValue &tensorValue,
                                             const MojoValue &indicesValue,
                                             bool isInput,
                                             unsigned operandIndex) {
  if (!enableDebugDumpFusedTensorSpecs)
    return;

  // We check if all indices are 0 to only print once.
  ManagedTensorSliceRef mts(tensorValue);
  builder.if_(MojoValue(kFnAllZeros)({}, indicesValue), [&]() {
    builder.call(
        kFnPrint, {},
        {MojoValue(isInput ? kParameterInput : kParameterOutput).quote(),
         MojoValue("#" + std::to_string(operandIndex) + ":").quote(),
         MojoValue("shape =").quote(), mts.staticShapeTuple(),
         MojoValue("strides =").quote(), mts.staticStridesTuple(),
         MojoValue("alignment =").quote(), mts.alignment()});
  });
}

void EmitMojoState::addDebugPrintLambdaParam(MOGG::BindLambdaKind lambdaKind,
                                             const MojoValue &widthParam,
                                             const MojoValue &alignmentParam,
                                             const MojoValue &indicesValue) {
  if (!enableDebugDumpFusedTensorSpecs)
    return;

  // We check if all indices are 0 to only print once.
  builder.if_(MojoValue(kFnAllZeros)({}, indicesValue), [&]() {
    builder.call(kFnPrint, {},
                 {
                     MojoValue(stringifyBindLambdaKind(lambdaKind)).quote(),
                     MojoValue("width =").quote(),
                     widthParam,
                     MojoValue("alignment =").quote(),
                     alignmentParam,
                 });
  });
}

MojoValue EmitMojoState::mlirOpaqueTypeToMojoType(MO::OpaqueType opaqueType) {
  // ImportPackagesPass already rewrote the symbol to a qualified Mojo name.
  return MojoValue(opaqueType.getSymbol().getValue());
}

SmallVector<MojoValue>
EmitMojoState::getOpaqueTypeParameters(MO::OpaqueType opaqueType) {
  SmallVector<MojoValue> params;
  DictionaryAttr paramDict = opaqueType.getParameters();
  if (!paramDict || paramDict.empty())
    return params;

  for (NamedAttribute param : paramDict) {
    // Emit named parameters like SIMDPair[S0=4, S1=8].
    StringRef name = param.getName().getValue();
    std::string valueStr;
    if (auto intAttr = dyn_cast<IntegerAttr>(param.getValue())) {
      // getInt() asserts signless or index; this may be signed (e.g. si32).
      valueStr = std::to_string(intAttr.getValue().getSExtValue());
    } else {
      // Fallback: use the attribute's string representation.
      llvm::raw_string_ostream ss(valueStr);
      param.getValue().print(ss);
    }
    params.push_back(MojoValue(name.str() + "=" + valueStr));
  }
  return params;
}
} // namespace M::MOGG
