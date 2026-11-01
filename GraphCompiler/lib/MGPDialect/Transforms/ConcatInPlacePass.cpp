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

#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MGPDialect/Transforms/FixupIsolatedFromAbove.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/AssertStream.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace M::MGP {

// Generate support code for ConcatInPlacePass.
#define GEN_PASS_DEF_CONCATINPLACEPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

/// Whether `v` is provably a multiple of `alignment`, recognising only the
/// forms the buffer-size lowering emits: a constant, a product with a proven
/// factor, or a sum of proven terms. A dynamic extent times a static per-row
/// byte stride is the common provable case.
bool isProvableMultipleOf(Value v, uint64_t alignment) {
  if (alignment <= 1)
    return true;
  if (auto cst = v.getDefiningOp<mlir::index::ConstantOp>())
    return cst.getValue().getSExtValue() % static_cast<int64_t>(alignment) == 0;
  if (auto mul = v.getDefiningOp<mlir::index::MulOp>())
    return isProvableMultipleOf(mul.getLhs(), alignment) ||
           isProvableMultipleOf(mul.getRhs(), alignment);
  if (auto add = v.getDefiningOp<mlir::index::AddOp>())
    return isProvableMultipleOf(add.getLhs(), alignment) &&
           isProvableMultipleOf(add.getRhs(), alignment);
  return false;
}

//===----------------------------------------------------------------------===//
// ConcatInPlacePass
//===----------------------------------------------------------------------===//

class ConcatInPlacePass
    : public impl::ConcatInPlacePassBase<ConcatInPlacePass> {
  using ConcatInPlacePassBase::ConcatInPlacePassBase;

  static SmallVector<Value>
  computeOffsets(OpBuilder &builder, Location loc,
                 ArrayRef<TypedValue<IndexType>> sliceSizes) {
    SmallVector<Value> result;
    result.reserve(sliceSizes.size());

    if (sliceSizes.empty())
      return result;

    // Equal sizes allow independent multiplies instead of a chain of adds.
    if (llvm::all_equal(sliceSizes)) {
      Value sliceSize = sliceSizes.front();

      for (auto index : llvm::seq(sliceSizes.size())) {
        auto indexCst = mlir::index::ConstantOp::create(
            builder, loc, static_cast<int64_t>(index));

        Value offset =
            builder.createOrFold<mlir::index::MulOp>(loc, indexCst, sliceSize);
        result.push_back(offset);
      }

    } else {
      Value offset = mlir::index::ConstantOp::create(builder, loc, 0);
      result.push_back(offset);

      for (Value sliceSize : sliceSizes.drop_back()) {
        offset =
            builder.createOrFold<mlir::index::AddOp>(loc, offset, sliceSize);
        result.push_back(offset);
      }
    }

    return result;
  }

  void runOnOperation() override {
    ModelOp modelOp = getOperation();
    BufferBuilderContext builderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(modelOp);
    ExecuteState execState(modelOp->getContext(), modelOp);

    modelOp.getExecute().walk([&](BufferConcatOp concatOp) {
      if (concatOp.getInBuffers().empty())
        // Ignore trivial concats.
        return;

      if (!llvm::all_of(concatOp.getInBuffers(), [&](Value buffer) {
            return buffer.getDefiningOp<MGPBufferAllocOpInterface>();
          }))
        // The concat input buffers are not explicitly allocated.
        return;

      auto allocOutputOp =
          concatOp.getOutBuffer().getDefiningOp<MGPBufferAllocOpInterface>();
      if (!allocOutputOp)
        // The concat output buffer is not explicitly allocated.
        return;

      DenseSet<Value> concatBuffers;
      for (Value inBuffer : concatOp.getInBuffers())
        concatBuffers.insert(inBuffer);

      if (concatBuffers.size() != concatOp.getInBuffers().size())
        // The concat input buffers are not distinct.
        return;

      execState.builder.setInsertionPoint(concatOp);
      TypedValue<MGP::DeviceBufferType> concatBuffer = concatOp.getOutBuffer();

      auto loc = concatOp.getLoc();

      MGPBufferBuilder &bufferBuilder =
          builderContext.getBufferBuilder(concatBuffer.getType().getDevice());
      OpBuilder &builder = execState.builder;

      auto sliceSizes =
          llvm::map_to_vector(concatOp.getInBuffers(), [&](Value inBuffer) {
            auto alloc = inBuffer.getDefiningOp<MGP::BufferAllocOp>();
            ASSERT_STREAM(alloc, << "unexpected buffer allocation op "
                                 << *inBuffer.getDefiningOp());
            return alloc.getRuntimeSize();
          });

      // A slice is only aligned when the preceding sizes sum to a provable
      // multiple of the alignment. Kernels assume every tensor base has the
      // device's preferred alignment (stamped into `mogg_tensor_init` by
      // `MOGGOps.cpp`), so an unprovable offset is a misaligned-address fault,
      // not a missed optimization. The last size feeds no offset.
      uint64_t alignment = M::Driver::getPreferredAlignment(
          concatBuffer.getType().getDevice().getLabel().str());
      if (!llvm::all_of(llvm::drop_end(sliceSizes), [&](Value size) {
            return isProvableMultipleOf(size, alignment);
          }))
        return;

      auto offsets = computeOffsets(builder, loc, sliceSizes);

      SmallVector<Value> outBuffers;
      for (auto [inBuffer, sliceSize, offset] :
           llvm::zip_equal(concatOp.getInBuffers(), sliceSizes, offsets)) {
        // Place each slice at its producer, for readability.
        if (Operation *op = inBuffer.getDefiningOp())
          execState.builder.setInsertionPoint(op);

        Value outSlice = bufferBuilder.bufferSlice(
            builder, loc, concatBuffer, offset, sliceSize, concatOp.getCtx());
        outBuffers.push_back(outSlice);
      }

      // Move the combined alloc to the top for readability.
      concatBuffer.getDefiningOp()->moveBefore(
          outBuffers.front().getDefiningOp());

      for (auto [origInBuffer, newOutBuffer] :
           llvm::zip(concatOp.getInBuffers(), outBuffers)) {
        origInBuffer.replaceAllUsesWith(newOutBuffer);
        origInBuffer.getDefiningOp()->erase();
      }

      concatOp.getOutChain().replaceAllUsesWith(concatOp.getInChain());

      concatOp->erase();
    });
  }
};

} // namespace
} // namespace M::MGP
