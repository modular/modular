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

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"

using namespace M;
using namespace KGEN;
using namespace M::MOGG;

namespace M::MOGG {

#define GEN_PASS_DEF_DUMPKERNELS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct DumpKernelsPass : public impl::DumpKernelsBase<DumpKernelsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    for (KernelDeclOp decl : moduleOp.getOps<KernelDeclOp>())
      llvm::errs() << KernelDeclAdaptor(decl) << "\n";
    for (PrimitiveDeclOp decl : moduleOp.getOps<PrimitiveDeclOp>())
      llvm::errs() << KernelFunctionDeclAdaptor(decl) << "\n";
  }
};

} // namespace

} // namespace M::MOGG
