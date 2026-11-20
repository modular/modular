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

#include "GraphCompiler/MOGGDialect/Support/ModuleHasher.h"

#include "mlir/Analysis/SymbolTableAnalysis.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_HASHMODULECONTENTS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct HashModuleContents
    : public impl::HashModuleContentsBase<HashModuleContents> {
  using HashModuleContentsBase::HashModuleContentsBase;

  void runOnOperation() override {
    auto op = getOperation();
    mlir::SymbolTableAnalysis &analysis =
        getAnalysis<mlir::SymbolTableAnalysis>();
    SymbolTable &symtab = analysis.getTopLevelSymbolTable();
    hashModuleContents(op, symtab);
  }
};

} // namespace

} // namespace M::MOGG
