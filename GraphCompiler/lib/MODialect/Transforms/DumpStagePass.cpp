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

#include "GraphCompiler/Driver/Common/Utils.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace MO;

namespace {

/// Dump file stem from the module's non-subgraph GraphOp names joined by "+",
/// or "graph" when there are none. Mirrors the `<names>.mo.mlir` stem
/// pipeline_to_mo_graph.py writes, so a graph's stages land beside its source.
std::string stageDumpBaseName(ModuleOp moduleOp) {
  std::string baseFileName;
  for (Operation &op : moduleOp.getOps()) {
    if (auto graphOp = dyn_cast<GraphOp>(op)) {
      if (!graphOp.getIsSubgraph()) {
        if (!baseFileName.empty())
          baseFileName += "+";
        baseFileName += graphOp.getSymName().str();
      }
    }
  }
  if (baseFileName.empty())
    baseFileName = "graph";
  return baseFileName;
}

/// Writes the current IR as a stage dump named `<baseName>.<suffix>.mlir` from
/// inside the pipeline, so phase boundaries dump without the caller holding the
/// module in hand. No-op unless an IR dump dir is set (getMaxIRDumpDir).
struct DumpStagePass
    : mlir::PassWrapper<DumpStagePass, mlir::OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DumpStagePass)

  std::string suffix;
  // Filled by the first dump while GraphOps still exist; later dumps reuse it,
  // since lowering to MGP removes the GraphOps the stem derives from.
  std::shared_ptr<std::string> baseName;
  DumpStagePass(std::string suffix, std::shared_ptr<std::string> baseName)
      : suffix(std::move(suffix)), baseName(std::move(baseName)) {}

  llvm::StringRef getArgument() const final { return "gc-dump-stage"; }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    if (baseName->empty())
      *baseName = stageDumpBaseName(moduleOp);
    M::writeIntermediateFileIfRequested(suffix, moduleOp, /*options=*/nullptr,
                                        BufferRef(), *baseName);
  }
};

} // namespace

namespace M::MO {

std::unique_ptr<mlir::Pass>
createDumpStagePass(std::string suffix, std::shared_ptr<std::string> baseName) {
  return std::make_unique<DumpStagePass>(std::move(suffix),
                                         std::move(baseName));
}

} // namespace M::MO
