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

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_LOOKUPCONSTANTRESOURCES
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

struct LookupConstantResources
    : public MOGG::impl::LookupConstantResourcesBase<LookupConstantResources> {
  using LookupConstantResourcesBase::LookupConstantResourcesBase;

  void runOnOperation() override {
    auto kernelStubOp = getOperation();
    auto constantResources = kernelStubOp.getConstantResources();
    auto ctx = kernelStubOp->getContext();
    bool wasInterrupted =
        kernelStubOp
            ->walk([&](MGP::BufferConstantOp bufferConstantOp) {
              auto elementsAttr = bufferConstantOp.getValue();
              auto it = llvm::find(constantResources, elementsAttr);
              if (it == constantResources.end()) {
                bufferConstantOp->emitError(
                    "The constant resource can't be found in the kernel");
                signalPassFailure();
                return mlir::WalkResult::interrupt();
              }
              uint64_t resourceIndex =
                  std::distance(constantResources.begin(), it);

              OpBuilder builder(bufferConstantOp);
              auto newConstantResource = MGP::ResourceConstantOp::create(
                  builder, bufferConstantOp->getLoc(),
                  bufferConstantOp.getType(),
                  IntegerAttr::get(IndexType::get(ctx), resourceIndex));
              bufferConstantOp->replaceAllUsesWith(newConstantResource);
              bufferConstantOp->erase();

              return mlir::WalkResult::advance();
            })
            .wasInterrupted();

    if (wasInterrupted)
      return;

    // FuncOp::emitMojo reads only the resource count, so keep an array
    // of the right size with placeholder entries.
    SmallVector<Attribute> dummyResources(constantResources.size(),
                                          UnitAttr::get(ctx));
    kernelStubOp.setConstantResourcesAttr(
        OpBuilder(kernelStubOp).getArrayAttr(dummyResources));
  }
};

} // namespace
} // namespace M::MOGG
