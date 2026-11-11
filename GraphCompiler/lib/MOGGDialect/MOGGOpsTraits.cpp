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
// This file implements the small interfaces (effects, successors, chains)
// of the MOGG dialect operations.
//
//===----------------------------------------------------------------------===//
#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/AsmSyntax.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "GraphCompiler/MOGGDialect/Support/MojoTraits.h"
#include "GraphCompiler/MOGGDialect/Support/ProfileScopeLabels.h"
#include "GraphCompiler/MOGGDialect/Support/SymbolNames.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/Support/NameMangling.h"
#include "Support/AssertStream.h"
#include "Support/DeviceSpecs.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include <cmath>
#include <cstdlib>
#include <future>
#include <vector>

using namespace M;
using namespace MOGG;

Value RoutineOp::getOutputChain() {
  mlir::ResultRange outputs = getOutputs();
  if (outputs.empty())
    return nullptr;
  Value last = outputs.back();
  return isa<MO::ChainType>(last.getType()) ? last : Value();
}

bool CallBuildGraphOp::isThrows() { return getIsThrows(); }

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

OutputOp FuncOp::getBodyTerminator() {
  return cast<OutputOp>(getBodyBlock().getTerminator());
}

bool FuncOp::isThrows() {
  // `AddTryExceptBlocks` always gives a MOGG stub the `mogg.raise` that renders
  // its error context. Answering without a body walk also keeps
  // `mogg.call`'s `isThrows` off a stub another thread is rewriting, and
  // prevents deep recursion through CallOp chains in sequence-fused
  // kernels.
  if (getAbi() == ABIKind::MOGG)
    return true;
  return llvm::any_of(getBodyBlock().getOps<MOGG::EmitsMojoInterface>(),
                      [](auto emitMojoOp) { return emitMojoOp.isThrows(); });
}

bool CallOpaqueOp::isThrows() {
  // Raises iff the callee's execute signature does, as in
  // `CallExecuteOp::isThrows`.
  auto decl = mlir::SymbolTable::lookupNearestSymbolFrom<KernelDeclOp>(
      *this, getCalleeAttr());
  if (!decl)
    return false;
  KernelDeclAdaptor adaptor(decl);
  // A declarative launch compiles and enqueues the kernel, which raises.
  if (adaptor.isDeclarative())
    return true;
  if (!adaptor.executeFunction || !adaptor.executeFunction->attr)
    return false;
  // `isThrows` is non-const on the signature; copy it to a local to call it.
  auto sig = adaptor.executeFunction->attr.getSignature();
  return sig.isThrows();
}

bool CallExecuteDeviceOp::isThrows() { return true; }

bool GraphBlobOp::isThrows() { return true; }

bool CallRowwiseOp::isThrows() { return true; }

RowwiseCacheOp CallRowwiseOp::getCache() {
  auto loads = getBody().front().getOps<RowwiseCacheOp>();
  return loads.empty() ? RowwiseCacheOp() : *loads.begin();
}

bool CallForeachTileOp::isThrows() { return true; }

bool CallForeachOp::isThrows() { return true; }

void RoutineOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  // Same rule as `RoutineOp::getEffects` above, on the MAP-lowered kernel.
  if (!getOutputParamDecls().empty())
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());

  bool addedEffect = false;
  for (auto &&opOperand : getOperation()->getOpOperands()) {
    if (!MO::isMutableOperand(opOperand.get()))
      continue;
    if (getHasWriteEffect())
      effects.emplace_back(mlir::MemoryEffects::Write::get(), &opOperand);
    if (getHasReadEffect())
      effects.emplace_back(mlir::MemoryEffects::Read::get(), &opOperand);
    addedEffect = true;
  }

  if (!addedEffect && (getHasReadEffect() || getHasWriteEffect()))
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

//===----------------------------------------------------------------------===//
// CompiledCallOp
//===----------------------------------------------------------------------===//

void CompiledCallOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  if (!getOutputParamDecls().empty())
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
  effects.emplace_back(mlir::MemoryEffects::Read::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get());
}

//===----------------------------------------------------------------------===//
// AsScalarOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AsyncRoutineOp
//===----------------------------------------------------------------------===//

OutputOp AsyncRoutineOp::getBodyTerminator() {
  return cast<OutputOp>(getBodyBlock().getTerminator());
}

InOutSignatureAttr::InOutSemantics
AsyncRoutineOp::getInOutSemantics(size_t index) {
  ASSERT_STREAM(index < getNumOperands(),
                << "AsyncRoutineOp::getInOutSemantics index out of range.");
  return getInOutSignature()[index];
}

//===----------------------------------------------------------------------===//
// AsTupleOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AsyncUnpackOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncUnpackOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto asyncType = dyn_cast<AsyncType>(adaptor.getInput().getType());
  if (!asyncType)
    return emitOptionalError(location,
                             "async.unpack input must be a mogg.async type");
  inferredReturnTypes.push_back(asyncType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

FuncOp CallOp::getCallee() {
  mlir::SymbolTable symTable(getOperation()->getParentOfType<ModuleOp>());
  return dyn_cast_or_null<FuncOp>(symTable.lookup(getName()));
}

bool CallOp::isThrows() {
  auto callee = getCallee();
  return callee && callee.isThrows();
}

//===----------------------------------------------------------------------===//
// CallAsyncOp
//===----------------------------------------------------------------------===//

FuncOp CallAsyncOp::getCallee() {
  mlir::SymbolTable symTable(getOperation()->getParentOfType<ModuleOp>());
  return dyn_cast_or_null<FuncOp>(symTable.lookup(getName()));
}

//===----------------------------------------------------------------------===//
// AsyncJoinOp
//===----------------------------------------------------------------------===//

bool AsyncJoinOp::isThrows() { return true; }

bool CallShapeOp::isThrows() { return getIsThrows(); }

//===----------------------------------------------------------------------===//
// WithTraceOp
//===----------------------------------------------------------------------===//

bool WithTraceOp::isThrows() { return true; }

//===----------------------------------------------------------------------===//
// ProfileScopeBeginOp / ProfileScopeEndOp
//===----------------------------------------------------------------------===//

bool ProfileScopeBeginOp::isThrows() { return true; }

bool ProfileScopeEndOp::isThrows() { return true; }

//===----------------------------------------------------------------------===//
// RaiseOp
//===----------------------------------------------------------------------===//

bool RaiseOp::isThrows() { return true; }

bool TryOp::isThrows() {
  auto regionThrows = [](Region &region) {
    return llvm::any_of(region.front().getOps<MOGG::EmitsMojoInterface>(),
                        [](auto op) { return op.isThrows(); });
  };
  return regionThrows(getTryRegion()) && regionThrows(getExceptRegion());
}
