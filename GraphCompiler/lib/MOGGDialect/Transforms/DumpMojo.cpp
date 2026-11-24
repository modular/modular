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

#define GEN_PASS_DEF_DUMPMOJO
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

struct DumpMojo : public impl::DumpMojoBase<DumpMojo> {
  using Base::Base;

  void runOnOperation() override {
    auto resourceManager = DenseResourceElementsHandle::getManagerInterface(
        getOperation()->getContext());
    auto mojoTextResource = resourceManager.getBlobManager().lookup("mojo");
    if (!mojoTextResource)
      return;

    auto mojoTextRef = mojoTextResource->getBlob()->getData();
    auto mojoText = StringRef(mojoTextRef.data(), mojoTextRef.size());
    llvm::errs() << mojoText << "\n";
  }
};

} // namespace M::MOGG
