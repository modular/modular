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

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/AsmSyntax.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/AssertStream.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace mlir;
using namespace M::MAP;

// The declarative formats name these unqualified.
using M::MOGG::parseKernelParamsBody;
using M::MOGG::printKernelParamsBody;

using namespace M;
using namespace mlir;
using namespace M::MAP;
using M::MOGG::parseKernelParamsBody;
using M::MOGG::printKernelParamsBody;

Value IterOpaqueOp::getOpaqueResult() {
  for (Value result : getResults())
    if (isa<MO::OpaqueType>(result.getType()))
      return result;
  return {};
}

unsigned IterOpaqueOp::getNumNonChainResults() {
  return getNumResults() - (getOutputChain() ? 1 : 0) -
         (getOpaqueResult() ? 1 : 0);
}

Block::BlockArgListType IterOpaqueOp::getOutputBlockArgs() {
  return getBody().front().getArguments().take_back(getNumNonChainResults());
}

Block::BlockArgListType IterOpaqueOp::getInputBlockArgs() {
  return getBody().front().getArguments().drop_back(getNumNonChainResults());
}

Value IterOpaqueOp::getInputFor(BlockArgument arg) {
  ValueRange inputs = getInputs();
  Block::BlockArgListType inputArgs = getInputBlockArgs();
  for (unsigned i = 0, e = inputs.size(); i < e; ++i)
    if (inputArgs[i] == arg)
      return inputs[i];
  return {};
}

IterEpilogueOp IterOpaqueOp::getEpilogueForResult(unsigned resultIdx) {
  Value outArg = getOutputBlockArgs()[resultIdx];
  for (auto epilogueOp : getBody().front().getOps<IterEpilogueOp>())
    if (llvm::is_contained(epilogueOp.getCaptures(), outArg))
      return epilogueOp;
  return nullptr;
}

void IterOpaqueOp::insertInputsAt(unsigned idx, ValueRange vals) {
  ASSERT_STREAM(idx <= getInputs().size(),
                << "idx (" << idx << ") must stay within the inputs ("
                << getInputs().size() << "), before outputs");
  SmallVector<Value> ops(getInputs());
  ops.insert(ops.begin() + idx, vals.begin(), vals.end());
  getInputsMutable().assign(ops);

  for (auto v : llvm::reverse(vals))
    getBody().front().insertArgument(idx, v.getType(), v.getLoc());
}

void IterOpaqueOp::eraseInputsAt(unsigned idx, size_t len) {
  ASSERT_STREAM(idx + len <= getInputs().size(), << "[idx, idx + len) = ["
                                                 << idx << ", " << (idx + len)
                                                 << "), before outputs");
  getInputsMutable().erase(idx, len);

  for (unsigned i = 0; i < len; ++i)
    getBody().front().eraseArgument(idx);
}

Value IterOpaqueOp::getOutputChain() {
  ResultRange results = getResults();
  if (results.empty())
    return nullptr;
  Value last = results.back();
  return llvm::isa<MO::ChainType>(last.getType()) ? last : Value();
}

void IterOpaqueOp::setChain(Value chain) {
  if (chain)
    getInputChainMutable().assign(chain);
}

// Reports hasReadEffect/hasWriteEffect per mutable operand, falling back to an
// unattributed write when no operand is mutable.
static void
collectKernelEffects(Operation *op, bool hasReadEffect, bool hasWriteEffect,
                     SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  bool addedEffect = false;
  for (OpOperand &opOperand : op->getOpOperands()) {
    if (!MO::isMutableOperand(opOperand.get()))
      continue;
    if (hasWriteEffect)
      effects.emplace_back(MemoryEffects::Write::get(), &opOperand);
    if (hasReadEffect)
      effects.emplace_back(MemoryEffects::Read::get(), &opOperand);
    addedEffect = true;
  }

  if (!addedEffect && (hasReadEffect || hasWriteEffect))
    effects.emplace_back(MemoryEffects::Write::get());
}

void IterOpaqueOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  collectKernelEffects(getOperation(), getHasReadEffect(), getHasWriteEffect(),
                       effects);
}

Value IterElementwiseOp::getOutputChain() {
  ResultRange results = getResults();
  if (results.empty())
    return nullptr;
  Value last = results.back();
  return llvm::isa<MO::ChainType>(last.getType()) ? last : Value();
}

void IterElementwiseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  collectKernelEffects(getOperation(), getHasReadEffect(), getHasWriteEffect(),
                       effects);
}

// The first device annotation on `results`, then `inputs`.
static llvm::SmallVector<M::DeviceRefAttr>
getFirstExecutionDevice(ValueRange results, ValueRange inputs) {
  for (ValueRange values : {results, inputs})
    for (Value v : values)
      if (auto dt =
              llvm::dyn_cast<M::MO::MOHasDeviceTypeInterface>(v.getType()))
        if (M::DeviceRefAttr device = dt.getDeviceRef())
          return {device};
  return {M::DeviceRefAttr{}};
}

llvm::SmallVector<M::DeviceRefAttr> IterElementwiseOp::getExecutionDevices() {
  return getFirstExecutionDevice(getResults(), getInputs());
}

Block::BlockArgListType IterElementwiseOp::getInputBlockArgs() {
  return getBody().front().getArguments().take_front(getInputs().size());
}

Block::BlockArgListType IterElementwiseOp::getOutputBlockArgs() {
  unsigned numOutputs = getNumResults() - (getOutputChain() ? 1 : 0);
  return getBody().front().getArguments().slice(getInputs().size(), numOutputs);
}

bool MAP::computesIndex(Operation *op) {
  auto isIndex = [](Type type) { return isa<IndexType>(type); };
  return llvm::any_of(op->getResultTypes(), isIndex) ||
         llvm::any_of(op->getRegions(), [&](Region &region) {
           return llvm::any_of(region.getArgumentTypes(), isIndex);
         });
}

TypedAttr MAP::getMaxSimdWidth(Region &region) {
  MLIRContext *ctx = region.getContext();
  SmallVector<TypedAttr> widths;
  for (Operation &op : region.getOps()) {
    if (auto constrained = dyn_cast<MaxSimdWidthInterface>(&op)) {
      widths.push_back(constrained.getMaxSimdWidth());
      continue;
    }
    // An index this op computes isn't the loop's, so what its readers need
    // can't be stated for the loop's index; width 1 is always valid.
    if (computesIndex(&op))
      return MOSH::ShapeAttr::getDim(ctx, 1);
    for (Region &nested : op.getRegions())
      widths.push_back(getMaxSimdWidth(nested));
  }
  return GCDAttr::getSimplified(ctx, widths);
}

CapturingFusionClosure
IterElementwiseOp::createBodyClosure(OpBuilder &builder, Location loc,
                                     ValueRange captures) {
  return IterElementwiseBodyOp::create(builder, loc, getShape(), captures,
                                       /*max_simd_width=*/nullptr);
}

void IterElementwiseOp::insertInputsAt(unsigned idx, ValueRange vals) {
  ASSERT_STREAM(idx <= getInputs().size(),
                << "idx (" << idx << ") must stay within the inputs ("
                << getInputs().size() << "), before outputs/index block args");
  SmallVector<Value> ops(getInputs());
  ops.insert(ops.begin() + idx, vals.begin(), vals.end());
  getInputsMutable().assign(ops);

  for (auto v : llvm::reverse(vals))
    getBody().insertArgument(idx, v.getType(), v.getLoc());
}

void IterElementwiseOp::eraseInputsAt(unsigned idx, size_t len) {
  ASSERT_STREAM(idx + len <= getInputs().size(),
                << "[idx, idx + len) = [" << idx << ", " << (idx + len)
                << ") must stay within the inputs (" << getInputs().size()
                << "), before outputs/index block args");
  getInputsMutable().erase(idx, len);

  for (unsigned i = 0; i < len; ++i)
    getBody().eraseArgument(idx);
}

void IterElementwiseOp::setChain(Value chain) {
  if (chain)
    getChainMutable().assign(chain);
}

Block::BlockArgListType IterRowwiseOp::getInputBlockArgs() {
  return getBody().front().getArguments().take_front(getInputs().size());
}

Block::BlockArgListType IterRowwiseOp::getOutputBlockArgs() {
  unsigned numOutputs = getNumResults() - (getOutputChain() ? 1 : 0);
  return getBody().front().getArguments().slice(getInputs().size(), numOutputs);
}

CapturingFusionClosure IterRowwiseOp::createBodyClosure(OpBuilder &builder,
                                                        Location loc,
                                                        ValueRange captures) {
  return IterRowwiseBodyOp::create(builder, loc, getShape(), getAxisAttr(),
                                   captures);
}

BlockArgument IterRowwiseOp::getRowArg() {
  return getBody().front().getArguments().back();
}

void IterRowwiseOp::insertInputsAt(unsigned idx, ValueRange vals) {
  ASSERT_STREAM(idx <= getInputs().size(),
                << "idx (" << idx << ") must stay within the inputs ("
                << getInputs().size() << "), before outputs/row block args");
  SmallVector<Value> ops(getInputs());
  ops.insert(ops.begin() + idx, vals.begin(), vals.end());
  getInputsMutable().assign(ops);

  for (auto v : llvm::reverse(vals))
    getBody().insertArgument(idx, v.getType(), v.getLoc());
}

void IterRowwiseOp::eraseInputsAt(unsigned idx, size_t len) {
  ASSERT_STREAM(idx + len <= getInputs().size(),
                << "[idx, idx + len) = [" << idx << ", " << (idx + len)
                << ") must stay within the inputs (" << getInputs().size()
                << "), before outputs/row block args");
  getInputsMutable().erase(idx, len);

  for (unsigned i = 0; i < len; ++i)
    getBody().eraseArgument(idx);
}

llvm::SmallVector<M::DeviceRefAttr> IterRowwiseOp::getExecutionDevices() {
  return getFirstExecutionDevice(getResults(), getInputs());
}

Value IterRowwiseOp::getOutputChain() {
  ResultRange results = getResults();
  if (results.empty())
    return nullptr;
  Value last = results.back();
  return llvm::isa<MO::ChainType>(last.getType()) ? last : Value();
}

void IterRowwiseOp::setChain(Value chain) {
  if (chain)
    getChainMutable().assign(chain);
}

void IterRowwiseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  collectKernelEffects(getOperation(), getHasReadEffect(), getHasWriteEffect(),
                       effects);
}

YieldOp IterReduceOp::getYield() {
  Block &body = getBody().front();
  return body.empty() ? YieldOp() : llvm::dyn_cast<YieldOp>(body.back());
}

Value SliceIndexOp::createStridedView(OpBuilder &builder, Value tensor,
                                      const IRMapping &outerValues) {
  auto inputTy = M::MO::castFromTensorLike(tensor.getType());
  auto resultTy = M::MO::TensorType::get(getSpec().getSignature().getOut(),
                                         inputTy.getElementType());
  return SliceStrideOp::create(builder, getLoc(), resultTy, getSpec(), tensor,
                               outerValues.lookupOrDefault(getStarts()),
                               outerValues.lookupOrDefault(getStops()),
                               outerValues.lookupOrDefault(getSteps()));
}

SmallVector<MOSH::ShapeAttr> SliceSelectOp::getDynamicShapes() {
  MAP::ViewSignatureAttr sig = getSpec().getSignature();
  if (sig.getIn().isStatic() && sig.getOut().isStatic())
    return {};
  return {sig.getIn(), sig.getOut()};
}

Value BroadcastIndexOp::createStridedView(OpBuilder &builder, Value tensor,
                                          const IRMapping &) {
  auto inputTy = M::MO::castFromTensorLike(tensor.getType());
  auto resultTy = M::MO::TensorType::get(getSpec().getSignature().getOut(),
                                         inputTy.getElementType());
  return BroadcastStrideOp::create(builder, getLoc(), resultTy, getSpec(),
                                   tensor);
}

//===----------------------------------------------------------------------===//
// map.index.reshape
//===----------------------------------------------------------------------===//

SmallVector<MOSH::ShapeAttr> ReshapeIndexOp::getDynamicShapes() {
  MAP::ViewSignatureAttr sig = getSpec().getSignature();
  if (sig.getIn().isStatic() && sig.getOut().isStatic())
    return {};
  // The lowered op names its index's space first: the out-shape.
  return {sig.getOut(), sig.getIn()};
}

Value ReshapeIndexOp::createStridedView(OpBuilder &builder, Value tensor,
                                        const IRMapping &) {
  auto inputTy = M::MO::castFromTensorLike(tensor.getType());
  auto resultTy = M::MO::TensorType::get(getSpec().getSignature().getOut(),
                                         inputTy.getElementType());
  return ReshapeStrideOp::create(builder, getLoc(), resultTy, getSpec(),
                                 tensor);
}

Value TransposeIndexOp::createStridedView(OpBuilder &builder, Value tensor,
                                          const IRMapping &outerValues) {
  auto inputTy = M::MO::castFromTensorLike(tensor.getType());
  auto resultTy = M::MO::TensorType::get(getSpec().getSignature().getOut(),
                                         inputTy.getElementType());
  return TransposeStrideOp::create(
      builder, getLoc(), resultTy, getSpec(), tensor,
      outerValues.lookupOrDefault(getPermutations()));
}
