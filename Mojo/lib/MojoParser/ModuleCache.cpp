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
//
// Implementation of the incremental parsing cache helpers.
//
//===----------------------------------------------------------------------===//

#include "Mojo/MojoParser/ModuleCache.h"

#include "Mojo/DepGraphDialect/DepGraphOps.h"
#include "Mojo/MojoParser/ModuleLoader.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/Support/MojoPrecompiledFile.h"
#include "Mojo/ToolCommon/CompilationOptions.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

void LIT::eraseIncrParsingCacheResumeAttrs(ModuleOp module) {
  module->walk([](Operation *op) {
    op->removeAttr(kIncrParsingCacheResumeInfoAttrName);
  });
}

/// Erase the trailing `depgraph.graph` op `writeIncrParsingCache` appended (a
/// direct pop of the last top-level op).
static void eraseDepGraph(ModuleOp module) {
  Block *body = module.getBody();
  if (!body->empty())
    if (auto graphOp = dyn_cast<DepGraph::GraphOp>(&body->back()))
      graphOp->erase();
}

void LIT::writeIncrParsingCache(SharedState &sharedState, ModuleOp module,
                                const ASTDecl &moduleDecl) {
  StringRef cachePath = sharedState.options.incrParsingCacheFile;

  // Mark declarations the lazy parse left unresolved so a later cache-served
  // compilation can resume them from source instead of consuming garbage.
  sharedState.prepareModuleForIncrParsingCache();

  // Embed the dependency graph in the module so graph and sections are
  // written and read as one file (no sidecar-skew window).
  module.getBody()->push_back(sharedState.getModuleLoader()
                                  .buildDepGraph(/*rootDecl=*/&moduleDecl)
                                  .release());

  // A failed write is non-fatal, but a truncated cache file must not survive
  // to poison the next run.
  std::error_code ec;
  llvm::raw_fd_ostream os(cachePath, ec, llvm::sys::fs::OF_None);
  if (!ec) {
    bool wrote = succeeded(KGEN::writePrecompiledFile(module, os));
    os.flush();
    if (os.has_error()) {
      os.clear_error();
      wrote = false;
    }
    if (!wrote) {
      os.close();
      llvm::sys::fs::remove(cachePath);
    }
  }

  eraseDepGraph(module);
}
