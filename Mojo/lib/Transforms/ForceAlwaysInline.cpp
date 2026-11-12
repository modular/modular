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
#include "Mojo/ToolCommon/KGENPasses.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_FORCEALWAYSINLINE
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct ForceAlwaysInline : impl::ForceAlwaysInlineBase<ForceAlwaysInline> {
  using ForceAlwaysInlineBase::ForceAlwaysInlineBase;

  explicit ForceAlwaysInline(std::function<void(FuncOp)> onAnnotated)
      : onAnnotated(std::move(onAnnotated)) {}

  void runOnOperation() override {
    FuncOp func = getOperation();
    // An external function has no body to inline, and the inliner cannot
    // inline a declaration.
    if (func.isExternal())
      return;
    if (inlineLevelOrAutomatic(func.getInlineLevel()) !=
            InlineLevel::Automatic &&
        onAnnotated) {
      // Callback to emit remark/warning/error if necessary.
      onAnnotated(func);
    }
    func.setInlineLevelAttr(
        getInlineLevelAttr(func.getContext(), InlineLevel::Always));
  }

  // Invoked per function carrying non-automatic inline level
  std::function<void(FuncOp)> onAnnotated;
};
} // namespace

std::unique_ptr<mlir::Pass>
KGEN::createForceAlwaysInline(std::function<void(FuncOp)> onAnnotated) {
  return std::make_unique<ForceAlwaysInline>(std::move(onAnnotated));
}
