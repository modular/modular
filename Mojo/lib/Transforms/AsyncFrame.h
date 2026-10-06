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

#ifndef KGEN_TRANSFORMS_ASYNCFRAME_H
#define KGEN_TRANSFORMS_ASYNCFRAME_H

#include "Mojo/CODialect/COOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/TransformUtils/AsyncUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace M::KGEN {

/// Frame Data stores any metadata necessary to transform the async function
/// into a suspendable procedure. This includes indexing information into the
/// frame type so that we can generate loads and stores and state information so
/// we can reuse loaded frame variables when legal.
struct FrameData {
  /// Error value and result value are excluded from the frame
  FrameData(FuncOp originalFunction, mlir::DominanceInfo &domInfo,
            Value errorValue, Value resultValue,
            function_ref<void(FuncOp, DenseMap<Operation *, int> &)> transform,
            bool isHot);
  FrameData(const FrameData &) = delete;
  FrameData(const FrameData &&other)
      : frameTypes(std::move(other.frameTypes)),
        valueToIndexInFrame(std::move(other.valueToIndexInFrame)),
        operationToIndexInFrame(std::move(other.operationToIndexInFrame)),
        opToState(std::move(other.opToState)),
        virtualBlocksFirstState(std::move(other.virtualBlocksFirstState)),
        argsInFrame(std::move(other.argsInFrame)),
        firstSuspends(std::move(other.firstSuspends)) {}

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

  /// Given a value, determine the state of its defining op or block argument.
  int getDefinitionStateForValue(Value operand, bool isHot) const;

  /// Update the ops in this virtual block.
  void updateVirtualBlock(Operation *virtualBlock, int newState);

  SmallVector<Type> frameTypes;
  DenseMap<Value, unsigned> valueToIndexInFrame;
  DenseMap<Operation *, unsigned> operationToIndexInFrame;
  DenseMap<Operation *, int> opToState;
  SmallVector<Operation *> virtualBlocksFirstState;
  SmallVector<ArgInFrame> argsInFrame;
  /// A suspend op 'S' is in this data structure if it is possible to reach 'S'
  /// from the entry point without passing through another suspend op. This is
  /// used to clear the state 0 blocks in hot resumes.
  DenseSet<CO::SuspendOp> firstSuspends;
};

/// Clone constant-like ops (and struct GEPs / offsets) into every state that
/// uses them so they never need to be stored in the frame.
void cloneFrameArgs(FuncOp funcOp, ImplicitLocOpBuilder &b,
                    mlir::DominanceInfo &domInfo, FuncOp originalFunction,
                    DenseMap<Operation *, int> &opToState);

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
  COTypes(const COTypes &&other)
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

using VirtualBlock = Operation *;

/// Frame Variables is a Cache of extracted frame variables. We may for example
/// reference a frame variable multiple times within a virtual block. We should
/// only extract that variable once for that state.
class FrameVariables {
public:
  FrameVariables(ImplicitLocOpBuilder &builder, const FrameData *frameData,
                 Value errorValue, Value resultValue)
      : builder(builder), frameData(frameData), errorValue(errorValue),
        resultValue(resultValue) {}

  /// Given the original operand, return the value extracted from the frame. Use
  /// a previously extracted value if available.
  Value getFrameValueForOperand(Value continuation, Value operand,
                                Operation *opWithUse, int useState);

  /// Overwrite the cached value for the variable in this state. A state can
  /// contain nested control flow, resulting in frame variables extracted in
  /// nested blocks. Sometime we could optimize this so that frame variables
  /// used multiple times in the same state are extracted in the first shared
  /// parent block, thus removing the need to overwrite state.
  void overwriteValue(int state, Value value);

private:
  ImplicitLocOpBuilder &builder;
  const FrameData *frameData;
  DenseMap<Value, DenseMap<int, Value>> frameVariables;
  Value errorValue;
  Value resultValue;
};

} // namespace M::KGEN

#endif // KGEN_TRANSFORMS_ASYNCFRAME_H
