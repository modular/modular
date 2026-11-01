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

#include "GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h" // IWYU: needed for type declarations used via MOGG::
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_ADDTRYEXCEPTBLOCKS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

struct AddTryExceptBlocks
    : public MOGG::impl::AddTryExceptBlocksBase<AddTryExceptBlocks> {
  using AddTryExceptBlocksBase::AddTryExceptBlocksBase;

  void runOnOperation() override {
    auto kernelOp = getOperation();
    OpBuilder builder(kernelOp->getContext());
    auto &kernelBody = kernelOp.getBodyBlock();
    auto terminator = kernelBody.getTerminator();
    Location loc = kernelOp->getLoc();

    // MOGG ABI: wrap every op except the terminator in a TryOp whose except
    // block raises with the kernel context (name, fusion info, traceback). The
    // terminator stays outside, so its values are forwarded through.
    if (kernelOp.getAbi() == MOGG::ABIKind::MOGG) {
      builder.setInsertionPointToStart(&kernelBody);
      auto tryOp =
          MOGG::TryOp::create(builder, loc, terminator->getOperandTypes());
      Block &tryBlock = tryOp.getTryRegion().emplaceBlock();
      for (Operation &op :
           llvm::make_early_inc_range(kernelBody.getOperations())) {
        if (!isa<MOGG::TryOp, MOGG::OutputOp>(op))
          op.moveBefore(&tryBlock, tryBlock.end());
      }
      builder.setInsertionPointToEnd(&tryBlock);
      MOGG::YieldOp::create(builder, loc, terminator->getOperands());

      Block &exceptBlock = tryOp.getExceptRegion().emplaceBlock();
      Value errArg =
          exceptBlock.addArgument(MOGG::ErrorType::get(loc->getContext()), loc);
      builder.setInsertionPointToStart(&exceptBlock);
      MOGG::RaiseOp::create(builder, loc, errArg);

      terminator->setOperands(tryOp.getOutputs());
      return;
    }

    if (kernelOp.getAbi() != MOGG::ABIKind::MGP || !kernelOp.isThrows())
      return;

    // Local destructors still run during unwinding, so cleanup needs no
    // explicit handling on either path.
    builder.setInsertionPointToStart(&kernelBody);
    auto tryOp =
        MOGG::TryOp::create(builder, loc, terminator->getOperandTypes());
    Block &tryBlock = tryOp.getTryRegion().emplaceBlock();
    for (Operation &op :
         llvm::make_early_inc_range(kernelBody.getOperations())) {
      if (!isa<MOGG::TryOp, MOGG::OutputOp>(op))
        op.moveBefore(&tryBlock, tryBlock.end());
    }

    builder.setInsertionPointToEnd(&tryBlock);
    MOGG::YieldOp::create(builder, loc, terminator->getOperands());

    // The except block returns a chain error and dummy values for other
    // outputs.
    Block &exceptBlock = tryOp.getExceptRegion().emplaceBlock();
    Value errArg =
        exceptBlock.addArgument(MOGG::ErrorType::get(loc->getContext()), loc);
    builder.setInsertionPointToStart(&exceptBlock);

    // One AsyncErrorOp per output, chain and non-chain.
    SmallVector<Value> exceptYieldValues;
    exceptYieldValues.reserve(tryOp.getOutputs().size());
    for (Value tryYieldOperand : tryOp.getOutputs()) {
      Value asyncError = MOGG::AsyncErrorOp::create(
          builder, loc, tryYieldOperand.getType(), errArg);
      exceptYieldValues.push_back(asyncError);
    }

    MOGG::YieldOp::create(builder, loc, exceptYieldValues);

    terminator->setOperands(tryOp.getOutputs());
  }
};

} // namespace

} // namespace M::MOGG
