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

#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_LOADMOJO
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

struct LoadMojo : public impl::LoadMojoBase<LoadMojo> {
  using Base::Base;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    static constexpr llvm::StringLiteral mojoResourceName = "mojo";
    auto resourceManager = DenseResourceElementsHandle::getManagerInterface(
        moduleOp->getContext());
    auto mojoTextResource =
        resourceManager.getBlobManager().lookup(mojoResourceName);
    if (!mojoTextResource) {
      moduleOp->emitError(
          "Can't load Mojo before emitting Mojo or after JIT compilation");
      return;
    }

    auto memBufferOr = toModularErrorOr(
        llvm::MemoryBuffer::getFile(this->path, /*IsText=*/true,
                                    /*RequiresNullTerminator=*/false));
    if (memBufferOr.isError()) {
      moduleOp->emitError(llvm::formatv("Could not load the Mojo file {} : {}",
                                        this->path, memBufferOr.getError()));
      signalPassFailure();
      return;
    }

    auto buffer = memBufferOr->get()->getBuffer();
    auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<char>(
        {buffer.data(), buffer.size()});
    auto &blobManager = resourceManager.getBlobManager();
    blobManager.update(mojoResourceName, std::move(blob));
  }
};

} // namespace M::MOGG
