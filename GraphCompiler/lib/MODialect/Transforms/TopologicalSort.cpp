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

#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"

namespace M::MO {

#define GEN_PASS_DEF_TOPOLOGICALSORT
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

struct TopologicalSort : public impl::TopologicalSortBase<TopologicalSort> {

  TopologicalSort() : TopologicalSortBase() {}

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ParametricTopologyAnalysis analysis(moduleOp);
    if (failed(analysis.topologicallySortAllOpsWithin(&moduleOp.getRegion())))
      return signalPassFailure();
  }
};

} // namespace M::MO
