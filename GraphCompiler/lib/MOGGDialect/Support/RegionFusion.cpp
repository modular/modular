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
// Implementation of fuseAndReplaceCompiledRegions.
//
// This lives in its own translation unit (rather than in FusionUtils.cpp or
// SequenceFusion.cpp) so it can be linked by both GraphCompilerInterfaces and
// MOGGTransforms without introducing a circular Bazel dependency.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"

#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SetVector.h"

using namespace M;

namespace M::MOGG {

AsyncRoutineOp fuseAndReplaceCompiledRegions(bool allowDuplication,
                                             ArrayRef<Operation *> toFuse) {
  ASSERT_STREAM(
      llvm::all_of(toFuse,
                   [](auto op) { return isa<MOGG::AsyncRoutineOp>(op); }),
      << "Expected all operations to be "
      << MOGG::AsyncRoutineOp::getOperationName());
  llvm::SetVector<Value> allResults;
  llvm::SetVector<Value> allOperands;
  for (Operation *op : toFuse) {
    auto results = op->getResults();
    auto operands = op->getOperands();
    allResults.insert(results.begin(), results.end());
    allOperands.insert(operands.begin(), operands.end());
  }

  // The fused op's operands will be allOperands - allResults.
  llvm::SetVector<Value> fusedRegionOperands = allOperands;
  fusedRegionOperands.set_subtract(allResults);

  // Identify results that must stay outputs after fusion.
  llvm::SetVector<Value> resultsThatShouldBeReplaced;
  SmallVector<Type> fusedRegionOutputTypes;
  DenseSet<Operation *> compiledRegionOpsToFuseTogetherAsSet(toFuse.begin(),
                                                             toFuse.end());

  SmallVector<Value> chainsThatShouldBeRemapped;
  for (auto result : allResults) {
    // We keep at least one chain output such that error could be propagated
    // properly. Duplicated chain outputs and operands are removed later in the
    // pipeline by the AsyncRoutineOp canonicalization.
    bool isChain = isa<MEF::ChainType>(result.getType());
    if (isChain) {
      chainsThatShouldBeRemapped.push_back(result);
      continue;
    }

    if (allowDuplication) {
      // If none of the users of this result are in the fusion set, then it is
      // something that is solely used from the outside.
      if (llvm::none_of(result.getUsers(), [&](auto user) {
            return compiledRegionOpsToFuseTogetherAsSet.contains(user);
          })) {
        resultsThatShouldBeReplaced.insert(result);
        fusedRegionOutputTypes.push_back(result.getType());
      }
    } else {
      // If any of the users of this result are used outside, then it must be
      // output too.
      if (llvm::any_of(result.getUsers(), [&](auto user) {
            return !compiledRegionOpsToFuseTogetherAsSet.contains(user);
          })) {
        resultsThatShouldBeReplaced.insert(result);
        fusedRegionOutputTypes.push_back(result.getType());
      }
    }
  }

  if (!chainsThatShouldBeRemapped.empty()) {
    fusedRegionOutputTypes.insert(fusedRegionOutputTypes.begin(),
                                  chainsThatShouldBeRemapped.front().getType());
  }

  // Maintain attributes.
  auto firstRegionOp = cast<MOGG::AsyncRoutineOp>(toFuse.front());
  // One deferred part makes the fused region deferred.
  const bool deferred = llvm::any_of(toFuse, [&](auto region) {
    return cast<MOGG::AsyncRoutineOp>(region).getDispatch() ==
           MOGG::DispatchKind::Deferred;
  });
  llvm::SmallSetVector<Attribute, 4> resources;
  for (Operation *regionOp : toFuse) {
    auto res = cast<MOGG::AsyncRoutineOp>(regionOp)
                   .getConstantResourcesAttr()
                   .getValue();
    resources.insert(res.begin(), res.end());
  }
  const bool canFuseIntoOther = llvm::all_of(toFuse, [&](auto region) {
    return cast<MOGG::AsyncRoutineOp>(region).getCanFuseIntoOther();
  });
  const bool canBeFusedInto = llvm::all_of(toFuse, [&](auto region) {
    return cast<MOGG::AsyncRoutineOp>(region).getCanBeFusedInto();
  });
  const bool canDuplicate = llvm::all_of(toFuse, [&](auto region) {
    return cast<MOGG::AsyncRoutineOp>(region).getCanDuplicate();
  });
  const bool isStructured = llvm::any_of(toFuse, [&](auto region) {
    return cast<MOGG::AsyncRoutineOp>(region).getIsStructured();
  });
  DeviceRefAttr device =
      cast<MOGG::AsyncRoutineOp>(toFuse.front()).getExecutionDevice();
  for (auto region : toFuse) {
    auto regionDevice = cast<MOGG::AsyncRoutineOp>(region).getExecutionDevice();
    if (regionDevice.getLabel() == kGPULabel)
      device = regionDevice;
  }

  auto ctx = firstRegionOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointAfter(toFuse.back());

  SmallVector<Location> locs;
  for (auto region : toFuse)
    locs.push_back(region->getLoc());
  auto loc = mlir::FusedLoc::get(ctx, locs);

  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics;
  auto fusedCompiledRegionOp = MOGG::AsyncRoutineOp::create(
      builder, loc, fusedRegionOutputTypes, fusedRegionOperands.getArrayRef(),
      KGEN::ParamDeclArrayAttr::get(ctx, {}),
      InOutSignatureAttr::get(ctx, operandSemantics),
      /*fusionInfo=*/
      MOGG::createFusionInfo<MOGG::FusionSequenceAttr>(ctx, toFuse),
      /*dispatch=*/
      deferred ? MOGG::DispatchKind::Deferred : MOGG::DispatchKind::Immediate,
      /*constantResources=*/builder.getArrayAttr(resources.getArrayRef()),
      canFuseIntoOther, canBeFusedInto, canDuplicate, isStructured);
  fusedCompiledRegionOp.setDeviceRef(device);
  Block &newBody = fusedCompiledRegionOp.getBody().emplaceBlock();
  builder.setInsertionPoint(&newBody, newBody.begin());

  // Body block arguments are `!mogg.async<T>` wrappers around the operand
  // types.
  DenseMap<Value, Value> outsideValueToInternalValue;
  for (Value operand : fusedRegionOperands) {
    auto newArg =
        newBody.addArgument(MOGG::AsyncType::get(operand.getType()), loc);
    outsideValueToInternalValue[operand] = newArg;
  }

  SmallVector<Value> valuesToOutput;

  IRMapping mapper;
  for (auto op : toFuse) {
    auto asRegionOp = cast<MOGG::AsyncRoutineOp>(op);
    for (auto [operand, argument] :
         llvm::zip(asRegionOp->getOperands(),
                   asRegionOp.getBody().front().getArguments())) {
      mapper.map(argument, outsideValueToInternalValue[operand]);
    }

    for (auto &internalOp : asRegionOp.getBody().front()) {
      auto newOp = builder.clone(internalOp, mapper);
      auto newOutputOp = dyn_cast<MOGG::OutputOp>(newOp);
      if (!newOutputOp)
        continue;

      for (auto [oldResult, internalValue] :
           llvm::zip(asRegionOp->getResults(), newOutputOp.getOperands())) {
        outsideValueToInternalValue[oldResult] = internalValue;
        if (resultsThatShouldBeReplaced.contains(oldResult))
          valuesToOutput.push_back(internalValue);
      }

      newOutputOp->erase();
    }
  }

  auto newOutputOp = MOGG::OutputOp::create(builder, loc, valuesToOutput);
  if (!chainsThatShouldBeRemapped.empty()) {
    auto firstReadyOp = *newBody.getOps<MOGG::AsyncReadyOp>().begin();
    firstReadyOp->moveBefore(newOutputOp);
    newOutputOp->insertOperands(0, firstReadyOp.getResult());

    for (auto &&chain : chainsThatShouldBeRemapped)
      chain.replaceAllUsesWith(fusedCompiledRegionOp.getResult(0));
  }

  for (auto [oldValue, newValue] :
       llvm::zip(resultsThatShouldBeReplaced.takeVector(),
                 fusedCompiledRegionOp.getResults().drop_front(
                     !chainsThatShouldBeRemapped.empty()))) {
    oldValue.replaceAllUsesWith(newValue);
  }

  return fusedCompiledRegionOp;
}

} // namespace M::MOGG
