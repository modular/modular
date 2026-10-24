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

#include "KGENToLLVM/Target/max_license/GPUCommon.h"

#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/POPDialect/POPOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Operation.h"

#include <utility>

namespace M::KGEN {

mlir::Type lowerAggregateArgToMemory(mlir::Type type) {
  auto structType = mlir::dyn_cast<StructType>(type);
  if (!structType || structType.getIsParamPack())
    return {};

  bool hasPointer = false;
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&hasPointer](PointerType) {
    hasPointer = true;
    return nullptr;
  });
  replacer.replace(type);
  if (!hasPointer)
    return {};

  return PointerType::get(type);
}

mlir::LogicalResult verifyNoGPUHeapAllocation(mlir::Operation *op) {
  if (mlir::isa<POP::AlignedAllocOp>(op))
    return op->emitError("heap allocation is not supported on GPU");
  if (mlir::isa<POP::AlignedFreeOp>(op))
    return op->emitError("heap deallocation is not supported on GPU");

  mlir::AttrTypeReplacer replacer;
  bool hasHeapAllocation = false;
  replacer.addReplacement([&](MemorySpaceAttr space)
                              -> std::pair<mlir::Attribute, mlir::WalkResult> {
    for (MemoryBlobAttr blob : space) {
      if (isGlobalBlob(blob))
        continue;
      if (blob.getKind() != MemoryKind::Stack) {
        hasHeapAllocation = true;
        return {nullptr, mlir::WalkResult::interrupt()};
      }
    }
    return {nullptr, mlir::WalkResult::advance()};
  });
  replacer.replaceElementsIn(op, /*replaceAttrs=*/true, /*replaceLocs=*/false,
                             /*replaceTypes=*/false);
  if (hasHeapAllocation)
    return op->emitError("heap allocation is not supported on GPU");
  return mlir::success();
}

} // namespace M::KGEN
