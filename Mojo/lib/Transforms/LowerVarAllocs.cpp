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

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERVARALLOCS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerVarAllocsPass : impl::LowerVarAllocsBase<LowerVarAllocsPass> {
  using LowerVarAllocsBase::LowerVarAllocsBase;

  void runOnOperation() override;

  /// Replaces `op`, whose address space is `addressSpace`, with its storage.
  /// Fails if the target has no storage for that address space.
  LogicalResult lowerVarAlloc(VarAllocOp op, int64_t addressSpace);
};
} // namespace

LogicalResult LowerVarAllocsPass::lowerVarAlloc(VarAllocOp op,
                                                int64_t addressSpace) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto ptrType = cast<PointerType>(op.getType());
  TypedAttr alignment = op.getAlignmentAttr();

  Value replacement;
  if (sharedAddressSpace < 0 || addressSpace == 0) {
    // Without shared memory, each thread has its own copy of every variable.
    // A variable in the default address space is a stack slot on every target.
    replacement =
        StackAllocationOp::create(b, loc, ptrType, /*count=*/1, alignment);
  } else if (addressSpace == stackAddressSpace) {
    // An `alloca` must be in the data layout's alloca address space, which can
    // differ from the stack address space. A default stack slot lowers to
    // that, so cast one.
    auto genericType =
        PointerType::get(ptrType.getElementType(),
                         /*addressSpace=*/0u, ptrType.getIsNonNull());
    Value slot =
        StackAllocationOp::create(b, loc, genericType, /*count=*/1, alignment);
    replacement = PointerBitcastOp::create(b, loc, ptrType, slot);
  } else if (addressSpace == sharedAddressSpace) {
    MLIRContext *ctx = op.getContext();
    replacement = GlobalAllocOp::create(
        b, loc, ptrType, StringAttr::get(op.getName(), StringType::get(ctx)),
        b.getIndexAttr(1),
        GlobalAllocAddressSpaceAttr::get(ctx,
                                         GlobalAllocAddressSpace::GPU_SHARED),
        alignment, /*initializer=*/TypedAttr());
  } else {
    return op.emitError("variable '")
           << op.getName() << "' in address space " << addressSpace
           << " is not supported on this target";
  }

  op.replaceAllUsesWith(replacement);
  op.erase();
  return success();
}

void LowerVarAllocsPass::runOnOperation() {
  bool hadError = false;
  getOperation()->walk([&](VarAllocOp op) {
    std::optional<unsigned> addressSpace =
        cast<PointerType>(op.getType()).getAddrSpace();
    if (!addressSpace) {
      op.emitOpError("address space is not known after elaboration");
      hadError = true;
      return;
    }
    if (failed(lowerVarAlloc(op, *addressSpace)))
      hadError = true;
  });
  if (hadError)
    signalPassFailure();
}
