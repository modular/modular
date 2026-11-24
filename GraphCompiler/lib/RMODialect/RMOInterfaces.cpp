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
// This file implements the RMO dialect interfaces.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/RMODialect/RMOInterfaces.h"
#include "GraphCompiler/MODialect/MOOps.h"

using namespace M;
using namespace RMO;

namespace M::RMO {
LogicalResult verifyRMOOp(RMOOp op) {
  auto parent = op->getParentOfType<MO::GraphOp>();
  if (!parent)
    return mlir::emitError(op.getLoc(), "Do not have a parent graph op!");

  return success();
}

LogicalResult verifySameDTypeOperandsAndResults(RMOOp op) {
  DType targetDType = DType::invalid;

  auto run = [&targetDType](ValueRange values) -> ErrorOrSuccess {
    for (const Value &v : values) {
      auto asTensor = dyn_cast<TensorValue>(v);
      if (!asTensor)
        continue;

      DType curDType = asTensor.getType().getDtype();
      if (targetDType == DType::invalid)
        targetDType = curDType;

      if (targetDType != curDType)
        return Error("Not all TensorValues have the same dtype.");
    }

    return success();
  };

  ErrorOrSuccess operandOutcome = run(op->getOperands());
  if (operandOutcome.isError())
    return mlir::emitError(op.getLoc(), operandOutcome.getError());

  ErrorOrSuccess resultOutcome = run(op->getResults());
  if (resultOutcome.isError())
    return mlir::emitError(op.getLoc(), resultOutcome.getError());

  if (targetDType == DType::invalid)
    return mlir::emitError(op.getLoc(), "No TensorValues found!");

  return success();
}
} // namespace M::RMO

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/RMODialect/RMOInterfaces.cpp.inc"
