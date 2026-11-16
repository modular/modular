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

#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"

using namespace M;
using namespace M::MOGG;
using namespace mlir;

namespace M::MOGG {

#define GEN_PASS_DEF_VALIDATEKERNELREGISTRY
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct ValidateKernelRegistryPass
    : public impl::ValidateKernelRegistryBase<ValidateKernelRegistryPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MojoRegistry registry(module);
    SmallVector<std::string> ambiguities;
    ErrorOrSuccess error =
        registry.findAmbiguousSelections(builtinsOnly, ambiguities);
    // An error does not discard the ambiguities: report both.
    if (error.isError())
      module.emitError(error.getError());
    for (const std::string &ambiguity : ambiguities)
      module.emitError(ambiguity);
    if (error.isError() || !ambiguities.empty())
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
