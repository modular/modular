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

#include "mlir/IR/Threading.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_ANONYMIZEDIMNAMES
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

// Drops the param-decl-ref name from every `!mogg.dim<name>` block arg in
// `stub`, nested blocks included, and from its function signature. Idempotent.
void stripStubDimNames(MOGG::FuncOp stub) {
  MLIRContext *ctx = stub.getContext();
  auto anonymous = MOGG::DimType::get(ctx, /*name=*/{});
  stub->walk([&](Block *block) {
    for (BlockArgument arg : block->getArguments())
      if (auto dim = dyn_cast<MOGG::DimType>(arg.getType());
          dim && dim.getName())
        arg.setType(anonymous);
  });
  FunctionType funcType = stub.getFunctionType();
  stub.setFunctionTypeAttr(TypeAttr::get(FunctionType::get(
      ctx, stub.getBodyBlock().getArgumentTypes(), funcType.getResults())));
}

struct AnonymizeDimNamesPass
    : public MOGG::impl::AnonymizeDimNamesBase<AnonymizeDimNamesPass> {
  using AnonymizeDimNamesBase::AnonymizeDimNamesBase;

  void runOnOperation() override {
    auto stubsRange = getOperation().getOps<MOGG::FuncOp>();
    SmallVector<MOGG::FuncOp> stubs(stubsRange.begin(), stubsRange.end());
    mlir::parallelForEach(&getContext(), stubs, stripStubDimNames);
  }
};

} // namespace

} // namespace M::MOGG
