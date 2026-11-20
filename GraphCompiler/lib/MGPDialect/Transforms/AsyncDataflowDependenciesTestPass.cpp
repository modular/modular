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
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include "GraphCompiler/MGPDialect/Transforms/AsyncDataflowDependencies.h"

using namespace mlir;

namespace M::MGP {

// Generate support code for AsyncDataflowDependenciesTestPass
#define GEN_PASS_DEF_ASYNCDATAFLOWDEPENDENCIESTESTPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// AsyncDataflowDependenciesTestPass
//===----------------------------------------------------------------------===//

class AsyncDataflowDependenciesTestPass
    : public impl::AsyncDataflowDependenciesTestPassBase<
          AsyncDataflowDependenciesTestPass> {
  using AsyncDataflowDependenciesTestPassBase::
      AsyncDataflowDependenciesTestPassBase;

  void runOnOperation() override {
    ModelOp modelOp = getOperation();
    IndexSet<Value> valueIndex = createValueIndexSet(modelOp);
    IndexSet<Operation *> operationIndex = createOperationIndexSet(modelOp);
    AsyncDataflowDependencies dependencies(
        &valueIndex, &operationIndex, AsyncDataflowDependencies::kExclusive);

    valueIndex.printModel(llvm::outs());
    dependencies.print(llvm::outs());
  }
};

} // namespace
} // namespace M::MGP
