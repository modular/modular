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

#ifndef KGEN_LIB_MOJOTOOLING_MOJOTOOLINGINTERNALS_H
#define KGEN_LIB_MOJOTOOLING_MOJOTOOLINGINTERNALS_H

#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/MojoTooling/ParserDriver.h"

namespace M::KGEN::LIT {
class ASTDecl;
class DeclResolver;
class TopLevelModuleASTDecl;
} // namespace M::KGEN::LIT

namespace M {

/// Resolves an unparsed decl enough for the language server to operate. Fully
/// body-resolves all decls descended from the root, validates their doc
/// strings, and leaves everything else unparsed.
void resolveForLSP(KGEN::LIT::DeclResolver &resolver, KGEN::LIT::ASTDecl &decl);

/// Signature-resolves all parsed decls that are not yet resolved, skipping
/// lazy named imports and body-resolving non-FnOp bytecode decls. This is the
/// second stage of LSP resolution: after resolveForLSP covers the direct
/// children of a container, this step covers transitive dependencies that were
/// pulled in during resolution.
void resolveSignaturesForLSP(KGEN::LIT::DeclResolver &resolver);

} // namespace M

namespace M::KGEN::LIT {

/// This class represents the internal implementation of the parser driver.
struct MojoParserContext::Impl {
  Impl(llvm::SourceMgr &sourceMgr, MLIRContext *context,
       const KGEN::CompilationOptions &options,
       const KGEN::FrontendOptions &frontendOptions);

  //===--------------------------------------------------------------------===//
  // General State
  //===--------------------------------------------------------------------===//

  /// The shared state for the parser.
  KGEN::LIT::SharedState sharedState;

  /// The top level decl for everything being parsed.
  KGEN::LIT::TopLevelModuleASTDecl *topLevelDecl = nullptr;

  /// The main module we are parsing into.
  mlir::OwningOpRef<ModuleOp> module;
};

} // namespace M::KGEN::LIT

#endif // KGEN_LIB_MOJOTOOLING_MOJOTOOLINGINTERNALS_H
