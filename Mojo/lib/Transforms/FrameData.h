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

#ifndef KGEN_TRANSFORMS_FRAMEDATA_H
#define KGEN_TRANSFORMS_FRAMEDATA_H

#include "Mojo/CODialect/COOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/TransformUtils/AsyncUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace M::KGEN {

struct FrameData;

using VirtualBlock = Operation *;

/// The queries the lowering makes about the coroutine frame, so each frame
/// evaluation can answer them from its own analysis.
class FrameInterface {
public:
  virtual ~FrameInterface() = default;

  /// Returns true if this op existed in the function when the frame data was
  /// calculated.
  virtual bool isTracked(Operation *op) = 0;

  /// Returns true if `use` reads its value after a suspend that follows the
  /// value's definition.
  virtual bool crossesSuspend(OpOperand &use) = 0;

  /// Returns true if some path from `from` to `to` passes a suspend. Ops with
  /// no suspend between them can share a frame load or a clone.
  virtual bool suspendBetween(Operation *from, Operation *to) = 0;

  /// Returns true if some path from the entry to `op` passes a suspend.
  virtual bool isAfterSuspend(Operation *op) = 0;

  /// Returns the first ops of the virtual blocks that no path from the entry
  /// through a suspend reaches, in dominance order. A hot resume removes them.
  virtual ArrayRef<VirtualBlock> firstStateVirtualBlocks() = 0;

  /// Returns the suspends reachable from the entry without passing another
  /// suspend.
  // TODO(stumpOS): Remove once the liveness frame evaluation is adopted. The
  // hot resume can walk every suspend and get the same parents to clear.
  virtual const DenseSet<CO::SuspendOp> &firstSuspends() = 0;

  /// Records that `clone`, a copy of `original`, was inserted for `user`.
  virtual void registerClone(Operation *clone, Operation *original,
                             Operation *user) = 0;
};

using FrameStateTransform = function_ref<void(FuncOp, FrameInterface &)>;

/// Populates `frameData` for `originalFunction`. Implementations invoke
/// `rematerializeFrameValues` with their answers to the cloner's queries, lay
/// out the frame, and store the frame interface in `frameData`.
using FrameEvaluator = void (*)(FrameData &frameData, FuncOp originalFunction,
                                mlir::DominanceInfo &domInfo, Value errorValue,
                                Value resultValue,
                                FrameStateTransform rematerializeFrameValues,
                                bool isHot);

/// Frame Data stores any metadata necessary to transform the async function
/// into a suspendable procedure. This includes indexing information into the
/// frame type so that we can generate loads and stores, and the frame
/// interface that answers where the suspends are.
struct FrameData {
  /// Error value and result value are excluded from the frame
  FrameData(FuncOp originalFunction, mlir::DominanceInfo &domInfo,
            Value errorValue, Value resultValue,
            FrameStateTransform rematerializeFrameValues, bool isHot,
            FrameEvaluator evaluateFrame) {
    evaluateFrame(*this, originalFunction, domInfo, errorValue, resultValue,
                  rematerializeFrameValues, isHot);
  }
  FrameData(const FrameData &) = delete;
  FrameData(FrameData &&) = default;

  FrameData() {}
  /// pairs index of argument from original function with its index in the
  /// frame.
  struct ArgInFrame {
    ArgInFrame(int index, int frameIndex)
        : argIndex(index), frameIndex(frameIndex) {}
    ArgInFrame() {}
    int argIndex = -1;
    int frameIndex = -1;
  };

  SmallVector<Type> frameTypes;
  DenseMap<Value, unsigned> valueToIndexInFrame;
  DenseMap<Operation *, unsigned> operationToIndexInFrame;
  SmallVector<ArgInFrame> argsInFrame;
  std::unique_ptr<FrameInterface> frame;
};

/// Clone constant-like ops (and struct GEPs / offsets) before every use that
/// crosses a suspend so they never need to be stored in the frame.
void cloneFrameArgs(FuncOp funcOp, ImplicitLocOpBuilder &b,
                    mlir::DominanceInfo &domInfo, FrameInterface &frame);

struct COTypes {
  Type typeForField(AsyncContinuationField field) {
    switch (field) {
    case State:
      return IntegerType::get(cxt, 32);
    case CallbackFn:
      return callbackSignature;
    case Promise:
      return promiseType;
    case ResumeFunction:
    case ClosureState:
    case ErrorSlot:
    case ResultSlot:
      return opaquePointerType;
    case Frame:
      return StructType::get(cxt, frameData.frameTypes);
    }
    llvm_unreachable("invalid AsyncContinuationField value");
  }
  COTypes(MLIRContext *cxt, FrameData &&frameData, StructType promiseType);
  COTypes(COTypes &&other)
      : continuationType(other.continuationType),
        resumeSignatureType(other.resumeSignatureType),
        opaquePointerType(other.opaquePointerType),
        callbackSignature(other.callbackSignature),
        headerType(other.headerType), cxt(other.cxt),
        frameData(std::move(other.frameData)), promiseType(other.promiseType) {}
  COTypes(const COTypes &) = delete;
  COTypes &operator=(const COTypes &) = delete;

public:
  Type getContinuationType() const { return continuationType; }
  FrameData *getFrameData() { return &frameData; }
  StructType getHeaderType() const { return headerType; }
  Type getResumeSignatureType() const { return resumeSignatureType; }
  StructType getPromiseType() const { return promiseType; }

private:
  Type continuationType;
  Type resumeSignatureType;
  Type opaquePointerType;
  Type callbackSignature;
  StructType headerType;
  MLIRContext *cxt;
  FrameData frameData;
  StructType promiseType;
};

/// Frame Variables is a Cache of extracted frame variables. We may for example
/// reference a frame variable multiple times within a virtual block. We should
/// only extract that variable once while no suspend separates its uses.
class FrameVariables {
public:
  FrameVariables(ImplicitLocOpBuilder &builder, const FrameData *frameData,
                 Value errorValue, Value resultValue)
      : builder(builder), frameData(frameData), errorValue(errorValue),
        resultValue(resultValue) {}

  /// Given the original operand, return the value extracted from the frame. Use
  /// a previously extracted value if available.
  Value getFrameValueForOperand(Value continuation, Value operand,
                                Operation *opWithUse);

  /// Overwrite the cached values for the variable that `op` could reuse. Uses
  /// with no suspend between them can be in nested control flow, resulting in
  /// frame variables extracted in nested blocks. Sometime we could optimize
  /// this so that frame variables used multiple times are extracted in the
  /// first shared parent block, thus removing the need to overwrite them.
  void overwriteValue(Operation *op, Value value);

private:
  /// A value extracted from the frame, and the op it was extracted for.
  struct ExtractedValue {
    Operation *extractedFor;
    Value image;
  };

  /// Removes the entries of `extracted` that `op` could reuse.
  void eraseReusable(SmallVector<ExtractedValue> &extracted, Operation *op);

  ImplicitLocOpBuilder &builder;
  const FrameData *frameData;
  DenseMap<Value, SmallVector<ExtractedValue>> frameVariables;
  Value errorValue;
  Value resultValue;
};

} // namespace M::KGEN

#endif // KGEN_TRANSFORMS_FRAMEDATA_H
