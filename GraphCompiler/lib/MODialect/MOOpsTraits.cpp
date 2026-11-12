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
// of the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GraphCompiler/Matchers.h"
#include "GraphCompiler/Support/Dims.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/ParserUtils.h"
#include "Support/ML/Permutation.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

using namespace M;
using namespace MO;
namespace M::MO {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static SmallVector<ChainValue> getChainValues(ValueRange values) {
  SmallVector<ChainValue> results;

  for (auto value : values) {
    if (auto chain = dyn_cast<ChainValue>(value))
      results.push_back(chain);
  }

  return results;
}
} // namespace M::MO

bool CallOp::isInPlace() {
  return !getInChains().empty() && !getOutChains().empty();
}

SmallVector<ChainValue> CallOp::getInChains() {
  return getChainValues(getOperands());
}

SmallVector<ChainValue> CallOp::getOutChains() {
  return getChainValues(getResults());
}

OutputOp GraphOp::getOutputOp() {
  return cast<OutputOp>(getBodyBlock()->getTerminator());
}

/// Name of the optional attribute holding the `#M.device_spec_collection`
/// describing the (virtual) devices available to the graph.
constexpr StringRef kGraphDeviceSpecCollectionAttrName = "device_specs";

void GraphOp::setDeviceSpecs(DeviceSpecCollectionAttr deviceSpecs) {
  getOperation()->setAttr(kGraphDeviceSpecCollectionAttrName, deviceSpecs);
}

void GraphOp::setDeviceTargetInfo(TargetInfoAttr targetInfo) {
  getOperation()->setAttr(kGraphTargetInfoAttrName, targetInfo);
}

ArrayRef<ParamDeclAttr> GraphOp::getInputParams() {
  return getInputParameters();
}

SmallVector<ParamDeclAttr> GraphOp::getAllParams() {
  SmallVector<ParamDeclAttr> params;
  params.append(getInputParameters().begin(), getInputParameters().end());
  params.append(getResultParameters().begin(), getResultParameters().end());
  return params;
}

void OutputOp::walkDefinitions(
    function_ref<void(KGEN::ParamDeclAttr, const KGEN::ParamDefValue &)>
        walkDef) {
  auto graph = cast<GraphOp>(getOperation()->getParentOp());
  ArrayRef<KGEN::ParamDeclAttr> decls = graph.getResultParameters();
  ArrayRef<TypedAttr> defs = this->getParameters();

  // Doesn't make sense to walk, graph is in a bad state.
  if (decls.size() != defs.size()) {
    emitError("[OutputOp::walkDefinitions] output bindings and result "
              "parameters must have the same size");
    return;
  }

  for (auto [decl, def] : llvm::zip(decls, defs))
    walkDef(decl, def);
}

void YieldOp::walkDefinitions(
    function_ref<void(KGEN::ParamDeclAttr, const KGEN::ParamDefValue &)>
        walkDef) {

  Operation *parentOp = getOperation()->getParentOp();

  if (auto ifOp = dyn_cast<IfOp>(parentOp)) {
    ArrayRef<KGEN::ParamDeclAttr> decls = ifOp.getOutputParamDecls();
    ArrayRef<TypedAttr> defs = this->getParameters();
    for (auto [decl, def] : llvm::zip(decls, defs))
      walkDef(decl, def);
  }
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

ArrayRef<ParamDeclAttr> IfOp::getInputParams() { return {}; }

SmallVector<ParamDeclAttr> IfOp::getAllParams() {
  SmallVector<ParamDeclAttr> params;
  params.append(getOutputParamDecls().begin(), getOutputParamDecls().end());
  return params;
}

void IfOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.emplace_back(mlir::RegionSuccessor(getOperation()));
    return;
  }

  regions.emplace_back(&getThenRegion());
  regions.emplace_back(&getElseRegion());
}

ValueRange IfOp::getSuccessorInputs(mlir::RegionSuccessor successor) {
  return successor.isOperation() ? getOperation()->getResults() : ValueRange();
}

void IfOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  regions.emplace_back(&getThenRegion());
  regions.emplace_back(&getElseRegion());
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

ArrayRef<ParamDeclAttr> WhileOp::getInputParams() { return {}; }

SmallVector<ParamDeclAttr> WhileOp::getAllParams() { return {}; }

void WhileOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // The parent op always branches to the condition region.
  if (point.isParent()) {
    regions.emplace_back(&getCondRegion());
    return;
  }

  assert(llvm::is_contained(
             {&getCondRegion(), &getBodyRegion()},
             point.getTerminatorPredecessorOrNull()->getParentRegion()) &&
         "there are only two regions in a WhileOp");
  // The body region always branches back to the condition region.
  if (point.getTerminatorPredecessorOrNull()->getParentRegion() ==
      &getBodyRegion()) {
    regions.emplace_back(&getCondRegion());
    return;
  }

  regions.emplace_back(mlir::RegionSuccessor(getOperation()));
  regions.emplace_back(&getBodyRegion());
}

ValueRange WhileOp::getSuccessorInputs(mlir::RegionSuccessor successor) {
  if (successor.isOperation())
    return getOperation()->getResults();
  if (successor == &getCondRegion())
    return getCondRegion().getArguments();
  if (successor == &getBodyRegion())
    return getBodyRegion().getArguments();
  llvm_unreachable("invalid region successor");
}

WhileConditionOp WhileOp::getCondTerminator() {
  return cast<WhileConditionOp>(getCondRegion().front().getTerminator());
}

YieldOp WhileOp::getBodyTerminator() {
  return cast<YieldOp>(getBodyRegion().front().getTerminator());
}

SmallVector<ChainValue> WhileOp::getInChains() {
  return getChainValues(getOperands());
}

SmallVector<ChainValue> WhileOp::getOutChains() {
  return getChainValues(getResults());
}

//===----------------------------------------------------------------------===//
// WhileConditionOp
//===----------------------------------------------------------------------===//

void WhileConditionOp::getSuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  WhileOp whileOp = getParentOp();

  // Branches to the after region if true, back to the parent op otherwise.
  regions.emplace_back(&whileOp.getBodyRegion());
  regions.emplace_back(mlir::RegionSuccessor(whileOp.getOperation()));
}

ArrayRef<ParamDeclAttr> ParallelOp::getInputParams() { return {}; }

SmallVector<ParamDeclAttr> ParallelOp::getAllParams() { return {}; }

SmallVector<ChainValue> ParallelOp::getInChains() {
  if (auto chain = getInChain())
    return {cast<ChainValue>(chain)};
  return {};
}

SmallVector<ChainValue> ParallelOp::getOutChains() {
  if (hasChain())
    return {getOutChain()};
  return {};
}

void ParallelOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  Block &body = getBodyRegion().front();

  SmallVector<mlir::MemoryEffects::EffectInstance> bodyEffects;
  for (Operation &op : body) {
    auto nested = mlir::getEffectsRecursively(&op);
    if (!nested)
      return;
    bodyEffects.append(nested->begin(), nested->end());
  }

  size_t numBundleArgs = getInputs().size();

  for (auto &effect : bodyEffects) {
    Value val = effect.getValue();
    auto blockArg = val ? dyn_cast<BlockArgument>(val) : BlockArgument{};
    if (!blockArg || blockArg.getOwner() != &body) {
      effects.push_back(effect);
      continue;
    }

    unsigned argIdx = blockArg.getArgNumber();
    if (argIdx < numBundleArgs) {
      // Remap bundle block arg effects to the bundle operand. If the bundle
      // is defined by a tensor.bundle op, trace through to individual
      // tensor operands.
      Value bundleVal = getInputs()[argIdx];
      Operation *defOp = bundleVal.getDefiningOp();
      if (defOp && isa<TensorBundleOp>(defOp)) {
        for (auto &operand : defOp->getOpOperands())
          effects.emplace_back(effect.getEffect(), &operand);
      } else {
        effects.emplace_back(effect.getEffect(), &getInputsMutable()[argIdx]);
      }
    } else {
      // The trailing chain block arg is not a memory resource. Buffers are
      // captured operands, so their effects are already keyed on the captured
      // SSA values.
      effects.push_back(effect);
    }
  }
}

//===----------------------------------------------------------------------===//
// SequenceOp
//===----------------------------------------------------------------------===//

ArrayRef<ParamDeclAttr> SequenceOp::getInputParams() { return {}; }

SmallVector<ParamDeclAttr> SequenceOp::getAllParams() { return {}; }

// `axis`, made non-negative.
static std::optional<int64_t> normalizedReductionAxis(TensorValue input,
                                                      int64_t axis) {
  if (axis < 0)
    axis += input.getType().getRank();
  return axis;
}

std::optional<int64_t> ArgMaxOp::getReductionAxis() {
  return normalizedReductionAxis(getInput(), getAxisAttr().getInt());
}

std::optional<int64_t> ArgMinOp::getReductionAxis() {
  return normalizedReductionAxis(getInput(), getAxisAttr().getInt());
}

LogicalResult
GuardOp::inferReturnTypes(MLIRContext *context,
                          std::optional<Location> location, Adaptor adaptor,
                          SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes = llvm::to_vector(TypeRange(adaptor.getInputs()));
  return success();
}

static TypedAttr emitStaticCall(TypedAttr chain, TypedAttr cond,
                                const Twine &message) {
  return emitStaticCallImpl(cond.getContext(), {chain, cond}, message);
}

static void linkAssertPair(mlir::RewriterBase &rewriter,
                           Operation *prevAssertOp,
                           Operation *currentAssertOp) {

  MLIRContext *ctx = currentAssertOp->getContext();

  auto prevAssertDeclareOp = dyn_cast<KGEN::ParamDeclareOp>(prevAssertOp);
  auto currentAssertDeclareOp = dyn_cast<KGEN::ParamDeclareOp>(currentAssertOp);

  // Grab the names of some relevant attrs so we don't hard code them later on.
  auto paramDeclOpName =
      mlir::OperationName(KGEN::ParamDeclareOp::getOperationName(), ctx);
  auto assertOpName = mlir::OperationName(AssertOp::getOperationName(), ctx);

  auto prevChain = prevAssertDeclareOp.getValue();

  auto operatorAttr = cast<KGEN::ParamOperatorAttr>(prevChain);
  auto operands = operatorAttr.getOperands();

  auto mlirOp = cast<KGEN::MLIROpAttr>(operands[0]);
  auto dictAttr = mlirOp.getAttrs();
  auto message = dictAttr.get(AssertOp::getMessageAttrName(assertOpName));

  StringAttr messageAttr = cast<mlir::StringAttr>(message);
  auto errorMsg = messageAttr.strref();

  auto currentOperands =
      cast<KGEN::ParamOperatorAttr>(currentAssertDeclareOp.getValue())
          .getOperands()
          .drop_front();

  // The new assert consumes prevAssertOp's output chain.
  auto newAttr = emitStaticCall(
      KGEN::ParamDeclRefAttr::get(prevAssertDeclareOp.getParamDecl()),
      currentOperands[0], errorMsg);

  rewriter.startOpModification(currentAssertDeclareOp);
  currentAssertDeclareOp->setAttr(
      KGEN::ParamDeclareOp::getValueAttrName(paramDeclOpName), newAttr);
  rewriter.finalizeOpModification(currentAssertDeclareOp);
}

static void linkAssertSequence(mlir::RewriterBase &rewriter,
                               ArrayRef<KGEN::ParamDeclareOp> asserts) {

  ASSERT_STREAM(!asserts.empty(),
                << "linkAssertSequence received an empty sequence of asserts!");

  KGEN::ParamDeclareOp prevAssert = asserts.front();

  for (auto assert : asserts.drop_front()) {
    linkAssertPair(rewriter, prevAssert, assert);
    prevAssert = assert;
  }
}

void AssertOp::materializeAllAssertLogic(
    mlir::RewriterBase &rewriter, Operation *op,
    ArrayRef<KGEN::ParamDeclareOp> asserts) {

  if (asserts.empty())
    return;

  linkAssertSequence(rewriter, asserts);

  mlir::RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  // Materialize the KGEN param-level chains as SSA values.
  SmallVector<Value> allChains;
  for (auto assert : asserts) {
    auto declRefAttr = KGEN::ParamDeclRefAttr::get(
        cast<KGEN::ParamDeclAttr>(assert.getParamDecl()));

    allChains.push_back(
        MOSH::ParamToValueOp::create(rewriter, op->getLoc(), declRefAttr));
  }

  // fan in all the assertion chains as the guard op takes a single chain
  auto guardChain =
      MO::ChainCreateOp::create(rewriter, op->getLoc(), ValueRange{allChains});

  auto guardOp = MO::GuardOp::create(rewriter, op->getLoc(), guardChain,
                                     op->getOperands());

  rewriter.modifyOpInPlace(op, [&guardOp, &op] {
    for (auto &&[replacement, old] :
         llvm::zip(guardOp->getResults(), op->getOpOperands())) {
      old.set(replacement);
    }
  });
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

/// Checks if an op's operands contains any mutable types: opaque or buffer.
static bool hasMutableOperand(OperandRange operands) {
  return llvm::any_of(operands, [](Value val) {
    return isa<MO::BufferType, MO::OpaqueType>(val.getType());
  });
}

bool CustomOp::isInPlace() {
  // TODO(GEX-1181): design kernel API/registration mechanism to refine this.
  return hasMutableOperand(getOperands()) && maybeGetInChain().has_value() &&
         maybeGetOutChain().has_value();
}

SmallVector<ChainValue> CustomOp::getInChains() {
  return getChainValues(getOperands());
}

SmallVector<ChainValue> CustomOp::getOutChains() {
  return getChainValues(getResults());
}

std::optional<ChainValue> CustomOp::maybeGetInChain() {
  if (getNumOperands() == 0)
    return std::nullopt;

  auto lastOperand = getOperands().back();
  if (auto chain = dyn_cast<ChainValue>(lastOperand))
    return chain;

  return std::nullopt;
}

std::optional<ChainValue> CustomOp::maybeGetOutChain() {
  if (getNumResults() == 0)
    return std::nullopt;

  auto finalResult = getResults().back();
  if (auto chain = dyn_cast<ChainValue>(finalResult))
    return chain;

  return std::nullopt;
}

mlir::SmallVector<Value> CustomOp::getBufferOperands() {
  SmallVector<Value> result;

  for (Value val : getOperands()) {
    if (isa<MO::BufferType>(val.getType()))
      result.push_back(val);
  }

  return result;
}

void CustomOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {

  bool isInPlace = this->isInPlace();

  // The mutable spec isn't a required attr at the moment. If it's not set
  // conservatively assume that there's a read and write effect on all mutable
  // operands.
  if (isInPlace) {
    for (auto &&opOperand : getOperation()->getOpOperands()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), &opOperand);
      effects.emplace_back(mlir::MemoryEffects::Write::get(), &opOperand);
    }
  }

  // Handle generic parameterization logic.
  if (!getOutputParamDecls().empty())
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
}

//===----------------------------------------------------------------------===//
// MO Mutable tensor operations
//===----------------------------------------------------------------------===//

void BufferTransferOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {

  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSrcMutable());

  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getDstMutable());
}

void MutableLoadOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInBufferMutable());
}

void MutableStoreOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getInBufferMutable());
}

void MutableStoreSliceOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getInBufferMutable());
}

void AllReduceSumOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

LogicalResult AllReduceSumOp::replaceWithBundledOps(IRRewriter &rewriter) {
  Location loc = getLoc();
  Value inChain = getInChain();

  // Every copy reduces over the whole world rather than its own group: the
  // kernel indexes the peer arrays by global rank, which is `peer_rank` here,
  // so slicing a grouped allreduce down to its group's operands would make
  // each device outside the first group read past the end of its arrays.
  SmallVector<Value> outputs, outChains;
  for (auto [rank, origOutput] : llvm::enumerate(getOutputs())) {
    auto bundled = BundledAllReduceSumOp::create(
        rewriter, loc, origOutput.getType(), inChain.getType(), getInputs(),
        getSignalBuffers(), inChain, getGroupSizeAttr(),
        rewriter.getI64IntegerAttr(rank));
    outputs.push_back(bundled.getOutput());
    outChains.push_back(bundled.getOutChain());
  }
  ::Impl::replaceDistributedOp(rewriter, getOperation(), getOutputs(),
                               getOutChain(), outputs, outChains);
  return success();
}

void BundledAllReduceSumOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  for (auto &sig : getSignalBuffersMutable())
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

void BundledBroadcastOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  for (auto &sig : getSignalBuffersMutable())
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

void BundledAllGatherOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  for (auto &sig : getSignalBuffersMutable())
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// BundledAllReduceAddRMSNormQuantFP8Op
//===----------------------------------------------------------------------===//

void Composite::BundledAllReduceAddRMSNormQuantFP8Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  for (auto &sig : getSignalBuffersMutable())
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// MegaFFNNvfp4Op
//===----------------------------------------------------------------------===//

void Composite::MegaFFNNvfp4Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  // The fused kernel writes the persistent `arrival_count` pool-slot buffer in
  // place (cross-CTA scheduler hand-off). Surfacing the Write lets the
  // buffer/chain machinery sequence the in-place mutation.
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getArrivalCountMutable());
}

void DistributedScatterOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

void ReduceScatterSumOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

void AllToAllOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

void EPDispatchOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange atomicCounters = getAtomicCountersMutable();
  for (auto &counter : atomicCounters)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &counter);
}

void EPDispatchFP8Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange atomicCounters = getAtomicCountersMutable();
  for (auto &counter : atomicCounters)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &counter);
}

void EPDispatchBlockScaledNVOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange atomicCounters = getAtomicCountersMutable();

  for (auto &counter : atomicCounters)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &counter);
}

void EPDispatchBlockScaledNVDynGlobalScalesOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange atomicCounters = getAtomicCountersMutable();

  for (auto &counter : atomicCounters)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &counter);
}

void EPDispatchMXFP4Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange atomicCounters = getAtomicCountersMutable();

  for (auto &counter : atomicCounters)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &counter);
}

void EPCombineOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange atomicCounters = getAtomicCountersMutable();
  for (auto &counter : atomicCounters)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &counter);
}

void Composite::DistributedMatmulReduceScatterSumOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

void AllGatherOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

LogicalResult AllGatherOp::replaceWithBundledOps(IRRewriter &rewriter) {
  ValueRange inputs = getInputs();
  auto numDevices = static_cast<int64_t>(inputs.size());
  int64_t groupSize = getGroupSize();
  if (groupSize != numDevices)
    return emitOpError()
           << "a device graph supports only a full-world allgather "
              "(group_size equal to the number of inputs, "
           << numDevices << "), but group_size is " << groupSize
           << ": a grouped allgather routes part of a shard through a paired "
              "group's buffers, which a device graph cannot express";
  if (getSliceRows())
    return emitOpError()
           << "a device graph has no per-device form of a row-sliced "
              "allgather (slice_rows)";

  Location loc = getLoc();
  Value inChain = getInChain();

  // The distributed op's outputs are device-major: device `d` owns the run
  // `[d * N, (d + 1) * N)`, one gathered piece per peer.
  ValueRange origOutputs = getOutputs();
  SmallVector<Value> outputs, outChains;
  for (auto rank : llvm::seq(numDevices)) {
    auto types =
        llvm::map_to_vector(origOutputs.slice(rank * numDevices, numDevices),
                            [](Value v) { return v.getType(); });
    auto bundled = BundledAllGatherOp::create(
        rewriter, loc, types, inChain.getType(), inputs, getSignalBuffers(),
        inChain, rewriter.getI64IntegerAttr(rank));
    llvm::append_range(outputs, bundled.getOutputs());
    outChains.push_back(bundled.getOutChain());
  }
  ::Impl::replaceDistributedOp(rewriter, getOperation(), origOutputs,
                               getOutChain(), outputs, outChains);
  return success();
}

void AllGatherConcatOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

void DistributedBroadcastOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

LogicalResult
DistributedBroadcastOp::replaceWithBundledOps(IRRewriter &rewriter) {
  Location loc = getLoc();
  Value inChain = getInChain();
  Value input = getInput();

  // Every device's copy, the root's included, reads the root tensor; stable
  // marking routes that read through a slot afterwards.
  SmallVector<Value> outputs, outChains;
  for (auto [rank, origOutput] : llvm::enumerate(getOutputs())) {
    auto bundled = BundledBroadcastOp::create(
        rewriter, loc, origOutput.getType(), inChain.getType(), input,
        getSignalBuffers(), inChain, getRootAttr(),
        rewriter.getI64IntegerAttr(rank));
    outputs.push_back(bundled.getOutput());
    outChains.push_back(bundled.getOutChain());
  }
  ::Impl::replaceDistributedOp(rewriter, getOperation(), getOutputs(),
                               getOutChain(), outputs, outChains);
  return success();
}

//===----------------------------------------------------------------------===//
// AllReduceAddRMSNormQuantFP8Op
//===----------------------------------------------------------------------===//

void Composite::AllReduceAddRMSNormQuantFP8Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// DistributedReduceScatterRMSNormOp
//===----------------------------------------------------------------------===//

void Composite::DistributedReduceScatterRMSNormOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// DistributedAllGatherRMSNormOp
//===----------------------------------------------------------------------===//

void Composite::DistributedAllGatherRMSNormOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// DistributedAllGatherGatedResidualRMSNormOp
//===----------------------------------------------------------------------===//

void Composite::DistributedAllGatherGatedResidualRMSNormOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// DistributedAllGatherRMSNormQuantMXFP8Op
//===----------------------------------------------------------------------===//

void Composite::DistributedAllGatherRMSNormQuantMXFP8Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// DistributedAllGatherRMSNormQuantMXFP6Op
//===----------------------------------------------------------------------===//

void Composite::DistributedAllGatherRMSNormQuantMXFP6Op::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// DistributedAllReduceRMSNormOp
//===----------------------------------------------------------------------===//

void Composite::DistributedAllReduceRMSNormOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// DebugTensorPrintOp
//===----------------------------------------------------------------------===//

bool DebugTensorPrintOp::isDataTypeAgnostic() { return false; }

//===----------------------------------------------------------------------===//
// ReduceAddOp, ReduceMinOp, ReduceMaxOp, ReduceMulOp

static std::optional<int64_t> implicitReductionAxis(TensorValue input,
                                                    TensorValue result) {
  ErrorOr<int64_t> axis = getReductionAxis(input, result.getType());
  if (axis.isError())
    return std::nullopt;
  return *axis;
}

std::optional<int64_t> ReduceAddOp::getReductionAxis() {
  return implicitReductionAxis(getInput(), getResult());
}

std::optional<int64_t> ReduceMinOp::getReductionAxis() {
  return implicitReductionAxis(getInput(), getResult());
}

std::optional<int64_t> ReduceMaxOp::getReductionAxis() {
  return implicitReductionAxis(getInput(), getResult());
}

std::optional<int64_t> ReduceMulOp::getReductionAxis() {
  return implicitReductionAxis(getInput(), getResult());
}
