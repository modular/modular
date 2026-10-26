//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "FrameData.h"
#include "Mojo/CODialect/COOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"

using namespace M;
using namespace KGEN;
using namespace POP;
using namespace CO;

//===----------------------------------------------------------------------===//
// Frame argument cloning
//===----------------------------------------------------------------------===//

static bool needsStateClone(Operation *operation) {
  return operation->hasTrait<OpTrait::ConstantLike>() ||
         isa<KGEN::StructGEPOp, POP::OffsetOp>(operation);
}

namespace {
struct CloneFrameArgs {
  CloneFrameArgs(ImplicitLocOpBuilder &b, FrameInterface &frame,
                 mlir::DominanceInfo &dominanceInfo)
      : builder(b), frame(frame), dominanceInfo(dominanceInfo) {}
  void cloneFrameArgsOf(Operation *user) {
    for (OpOperand &use : user->getOpOperands()) {
      Operation *definingOp = use.get().getDefiningOp();
      if (!definingOp)
        continue;
      if (!needsStateClone(definingOp))
        continue;
      if (!frame.crossesSuspend(use))
        continue;

      SmallVector<ClonedFor> &clones = clonesOf[definingOp];
      auto existingClone =
          llvm::find_if(llvm::reverse(clones), [&](const ClonedFor &existing) {
            return !frame.suspendBetween(existing.user, user);
          });
      Operation *clonedDefOp;
      if (existingClone == llvm::reverse(clones).end()) {
        builder.setInsertionPoint(user);
        clonedDefOp = builder.clone(*definingOp);
        frame.registerClone(clonedDefOp, definingOp, user);
        clones.push_back({user, clonedDefOp});
        if (clonedDefOp->getNumOperands() > 0)
          cloneFrameArgsOf(clonedDefOp);
      } else {
        clonedDefOp = existingClone->clone;
        if (!dominanceInfo.dominates(clonedDefOp, user)) {
          // We have two uses with no suspend between them where one does not
          // dominate the other. This implies that the first instance of the
          // usage is in a nested region.
          Operation *parent = clonedDefOp->getParentOp();
          while (!dominanceInfo.dominates(parent, user))
            parent = parent->getParentOp();
          clonedDefOp->moveBefore(parent);
        }
      }
      use.set(clonedDefOp->getResult(0));
    }
  }
  /// A clone of a constant, and the op it was first cloned for.
  struct ClonedFor {
    Operation *user;
    Operation *clone;
  };

  ImplicitLocOpBuilder &builder;
  FrameInterface &frame;
  DenseMap<Operation *, SmallVector<ClonedFor>> clonesOf;
  mlir::DominanceInfo &dominanceInfo;
};
} // namespace

void M::KGEN::cloneFrameArgs(FuncOp funcOp, ImplicitLocOpBuilder &b,
                             mlir::DominanceInfo &domInfo,
                             FrameInterface &frame) {
  auto insertPoint = b.saveInsertionPoint();
  CloneFrameArgs cloner(b, frame, domInfo);
  funcOp.walk([&](Operation *user) {
    if (user->getNumOperands() > 0)
      cloner.cloneFrameArgsOf(user);
  });
  b.restoreInsertionPoint(insertPoint);
}

//===----------------------------------------------------------------------===//
// CoTypes
//===----------------------------------------------------------------------===//

COTypes::COTypes(MLIRContext *cxt, FrameData &&frameData,
                 StructType promiseType)
    : cxt(cxt), frameData(std::move(frameData)), promiseType(promiseType) {
  opaquePointerType = PointerType::get(KGEN::NoneType::get(cxt));
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  inputs.push_back(opaquePointerType);
  FunctionType resumeFunctionType = FunctionType::get(cxt, inputs, results);
  resumeSignatureType =
      FuncTypeGeneratorType::get(/*inputParamTypes=*/{}, resumeFunctionType);
  FunctionType callbackFunctionType =
      FunctionType::get(cxt, opaquePointerType, results);
  callbackSignature =
      FuncTypeGeneratorType::get(/*inputParamTypes=*/{}, callbackFunctionType);

  // Build Continuation Type.
  size_t size = Promise;
  SmallVector<Type> types(size);
  types[State] = typeForField(State);
  types[ResumeFunction] = typeForField(ResumeFunction);
  types[CallbackFn] = typeForField(CallbackFn);
  types[ClosureState] = typeForField(ClosureState);
  types[ErrorSlot] = typeForField(ErrorSlot);
  types[ResultSlot] = typeForField(ResultSlot);

  // Header type omits the variable sized frame and promise.
  headerType = StructType::get(cxt, types);

  // Only create continuationType if promiseType is valid. Some callers
  // pass a null promiseType when they only need the headerType.
  if (promiseType) {
    types.push_back(typeForField(Promise));
    for (auto [index, frameVariableType] :
         llvm::enumerate(this->frameData.frameTypes))
      types.push_back(frameVariableType);
    continuationType = StructType::get(cxt, types);
  }
}

//===----------------------------------------------------------------------===//
// FrameVariables
//===----------------------------------------------------------------------===//

void FrameVariables::eraseReusable(SmallVector<ExtractedValue> &extracted,
                                   Operation *op) {
  FrameInterface &frame = *frameData->frame;
  llvm::erase_if(extracted, [&](const ExtractedValue &existing) {
    return !frame.suspendBetween(existing.extractedFor, op);
  });
}

Value FrameVariables::getFrameValueForOperand(Value continuation, Value operand,
                                              Operation *opWithUse) {
  auto entry = frameData->valueToIndexInFrame.find(operand);
  if (entry == frameData->valueToIndexInFrame.end() && operand != errorValue &&
      operand != resultValue)
    return {};
  SmallVector<ExtractedValue> &extracted = frameVariables[operand];

  // Reuse existing extracted value if possible.
  FrameInterface &frame = *frameData->frame;
  auto existingImage = llvm::find_if(
      llvm::reverse(extracted), [&](const ExtractedValue &existing) {
        return !frame.suspendBetween(existing.extractedFor, opWithUse);
      });
  // TODO: can this be parent region also?
  bool wasExtractedInThisRegion =
      existingImage != llvm::reverse(extracted).end() &&
      existingImage->image.getParentRegion() == opWithUse->getParentRegion();
  Value image;
  if (wasExtractedInThisRegion) {
    image = existingImage->image;
  } else {
    builder.setInsertionPoint(opWithUse);
    if (operand == errorValue) {
      Value dataSlot = StructGEPOp::create(builder, continuation, ErrorSlot);
      Value ptr = LoadOp::create(builder, dataSlot);
      image = PointerBitcastOp::create(builder, errorValue.getType(), ptr);
    } else if (operand == resultValue) {
      Value dataSlot = StructGEPOp::create(builder, continuation, ResultSlot);
      Value ptr = LoadOp::create(builder, dataSlot);
      image = PointerBitcastOp::create(builder, resultValue.getType(), ptr);
    } else {
      unsigned frameIndex = entry->getSecond();
      Value dataSlot =
          StructGEPOp::create(builder, continuation, Frame + frameIndex);
      if (operand.getDefiningOp() &&
          isa<StackAllocationOp>(operand.getDefiningOp())) {
        auto stackAlloc = dyn_cast<StackAllocationOp>(operand.getDefiningOp());
        if (cast<IntegerAttr>(stackAlloc.getCount()).getInt() == 1) {
          image = dataSlot;
        } else {
          image =
              PointerBitcastOp::create(builder, stackAlloc.getType(), dataSlot);
        }
      } else {
        image = LoadOp::create(builder, dataSlot);
      }
    }
    eraseReusable(extracted, opWithUse);
    extracted.push_back({opWithUse, image});
  }
  return image;
}

void FrameVariables::overwriteValue(Operation *op, Value value) {
  SmallVector<ExtractedValue> &extracted = frameVariables[value];
  eraseReusable(extracted, op);
  extracted.push_back({op, value});
}
