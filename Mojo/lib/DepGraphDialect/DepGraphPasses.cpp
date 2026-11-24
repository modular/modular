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

#include "Mojo/DepGraphDialect/DepGraphPasses.h"

#include "Mojo/DepGraphDialect/DepGraphAnalysis.h"
#include "Mojo/DepGraphDialect/DepGraphOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace DepGraph;

namespace {
/// Runs the dirty-propagation analysis against the on-disk sources and marks
/// each dirty module node with a `depgraph.dirty` unit attribute: a node is
/// dirty when its own source changed (or cannot be verified) or when any
/// transitive dependency is dirty.
struct MarkDirtyPass
    : public mlir::PassWrapper<MarkDirtyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkDirtyPass)

  llvm::StringRef getArgument() const override { return "depgraph-mark-dirty"; }
  llvm::StringRef getDescription() const override {
    return "Mark dependency-graph module nodes whose sources changed";
  }

  void runOnOperation() override {
    mlir::Builder b(&getContext());
    getOperation().walk([&](GraphOp graph) {
      SourceHashes hashes = hashModuleSources(graph);
      for (mlir::Operation *dirty : computeDirtyModules(graph, hashes))
        dirty->setAttr("depgraph.dirty", b.getUnitAttr());
    });
  }
};
} // namespace

void DepGraph::registerDepGraphPasses() {
  mlir::PassRegistration<MarkDirtyPass>{};
}
