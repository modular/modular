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

#ifndef KGEN_LIB_MOJOTOOLING_REPLINTERNALS_H
#define KGEN_LIB_MOJOTOOLING_REPLINTERNALS_H

#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/MojoREPL/REPL.h"
#include "Mojo/MojoTooling/ParserDriver.h"

namespace M::KGEN::LIT {

struct MojoREPLContext::Impl {
  explicit Impl(MojoParserContext &parserContext)
      : parserContext(parserContext), locMapper(parserContext.getSourceMgr()) {}

  MojoParserContext &parserContext;
  LocMapper locMapper;

  llvm::MapVector<ASTDecl *, ASTDecl *> prevReplModuleDecls;
  SmallVector<ASTDecl *> replModuleDecls;

  struct CompletionCache {
    llvm::SourceMgr sourceMgr;
    std::unique_ptr<MojoParserContext> context;
    ASTDecl *lastResolvedDecl = nullptr;
    size_t replayedModuleCount = 0;
    ASTDecl *replDeclKey = nullptr;
    size_t completionsSinceRebuild = 0;

    explicit CompletionCache(const SharedState &mainState) {
      sourceMgr.setDiagHandler([](const llvm::SMDiagnostic &, void *) {});
      sourceMgr.setIncludeDirs(mainState.getSourceMgr().getIncludeDirs());
      KGEN::FrontendOptions frontendOptions = mainState.frontendOptions;
      frontendOptions.autoFixItHandler = nullptr;
      frontendOptions.useMLIRDiagnostics = false;
      frontendOptions.parserListener = nullptr;
      frontendOptions.maxNotesPerDiagnostic = 0;
      context = std::make_unique<MojoParserContext>(
          sourceMgr, mainState.getContext(), mainState.options,
          frontendOptions);
    }
  };
  std::unique_ptr<CompletionCache> completionCache;
};

} // namespace M::KGEN::LIT

#endif // KGEN_LIB_MOJOTOOLING_REPLINTERNALS_H
