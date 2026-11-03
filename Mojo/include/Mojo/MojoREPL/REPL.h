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

#ifndef KGEN_MOJOREPL_REPL_H
#define KGEN_MOJOREPL_REPL_H

#include "Support/LLVMCompilerForwardDecls.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace llvm {
class SMDiagnostic;
class SourceMgr;
} // namespace llvm

namespace M::KGEN {
namespace LIT {
class ASTDecl;
class MojoParserContext;
} // namespace LIT
namespace Mojo {
struct CodeCompletionResult;
struct SignatureHelpResult;
} // namespace Mojo
} // namespace M::KGEN

namespace M::KGEN::LIT {

/// This class provides callbacks for clients that evaluate REPL expressions.
class MojoREPLListener {
public:
  virtual ~MojoREPLListener() = default;

  virtual void notifyWrappedExpr(StringRef wrappedExpr) = 0;
  virtual void notifyFixedExpr(StringRef fixedExpr) = 0;
  virtual void notifyDiagnostics(ArrayRef<llvm::SMDiagnostic> diagnostics) = 0;
  virtual bool shouldPersistVariable(StringRef name, Type type) = 0;
};

/// Manages the state needed to parse and complete a sequence of REPL
/// expressions using a parser context.
class MojoREPLContext {
public:
  explicit MojoREPLContext(MojoParserContext &parserContext);
  ~MojoREPLContext();

  class LocMapper {
  public:
    class ExprLocMapper;

    ~LocMapper();

    llvm::SMLoc mapLocation(llvm::SMLoc loc) const;
    llvm::SMRange mapRange(llvm::SMRange range) const;
    llvm::SMDiagnostic mapDiagnostic(const llvm::SMDiagnostic &diag);

  private:
    explicit LocMapper(llvm::SourceMgr &sourceMgr);
    friend class MojoREPLContext;

    llvm::SourceMgr &sourceMgr;
    std::vector<std::unique_ptr<ExprLocMapper>> exprMappers;
  };

  struct ParsedExpression {
    bool isValid() const { return exprFnDecl; }

    ASTDecl *moduleDecl;
    ASTDecl *exprFnDecl;
  };

  LocMapper &getLocMapper();

  ParsedExpression
  parseExpression(MojoREPLListener &listener, unsigned exprFileId,
                  StringRef replExprFnName,
                  ArrayRef<std::pair<StringRef, Type>> replVariables);

  ParsedExpression
  parseExpression(MojoREPLListener &listener, unsigned exprFileId,
                  StringRef exprText, StringRef replExprFnName,
                  ArrayRef<std::pair<StringRef, Type>> replVariables,
                  ASTDecl *prevReplExpr, bool parseForLSP);

  std::vector<KGEN::Mojo::CodeCompletionResult>
  codeComplete(StringRef exprText, uint64_t completionPosition,
               ArrayRef<std::pair<StringRef, Type>> replVariables);
  std::vector<KGEN::Mojo::CodeCompletionResult>
  codeComplete(StringRef exprText, uint64_t completionPosition,
               ArrayRef<std::pair<StringRef, Type>> replVariables,
               ASTDecl *replDecl);

  std::optional<KGEN::Mojo::SignatureHelpResult>
  signatureHelp(StringRef exprText, uint64_t position,
                ArrayRef<std::pair<StringRef, Type>> replVariables,
                ASTDecl *replDecl);

  void removeLastExpression();

  static bool isHiddenPersistentVariable(StringRef name);

private:
  StringRef prepareCompletionCache(ASTDecl *replDecl);

  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace M::KGEN::LIT

namespace M {
using KGEN::LIT::MojoREPLContext;
using KGEN::LIT::MojoREPLListener;
} // namespace M

#endif // KGEN_MOJOREPL_REPL_H
