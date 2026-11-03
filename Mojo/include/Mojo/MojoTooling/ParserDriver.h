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

#ifndef KGEN_MOJOTOOLING_PARSERDRIVER_H
#define KGEN_MOJOTOOLING_PARSERDRIVER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include <filesystem>
#include <memory>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace M::KGEN {
class CompilationOptions;
struct FrontendOptions;
namespace LIT {
class ASTDecl;
class SharedState;
} // namespace LIT
} // namespace M::KGEN

namespace M::KGEN::LIT {

//===----------------------------------------------------------------------===//
// MojoParserContext
//===----------------------------------------------------------------------===//

/// This class provides a context for parsing and interacting with Mojo
/// modules.
class MojoParserContext {
public:
  MojoParserContext(llvm::SourceMgr &sourceMgr, MLIRContext *context,
                    const KGEN::CompilationOptions &options,
                    const FrontendOptions &frontendOptions);
  ~MojoParserContext();

  /// Return the current module being parsed.
  ModuleOp getModule();

  /// Return the source manager used by the parser.
  llvm::SourceMgr &getSourceMgr();

  /// Return the shared state user by the parser.
  SharedState &getSharedState();

  /// Return the compilation options used by the parser.
  const KGEN::CompilationOptions &getCompilationOptions();

  /// Parse a SourceMgr file given its id as a module.
  ///
  /// In the case of success, the decl corresponding to the module is returned.
  /// In the case of an error, a null decl is returned. If `eraseUnparsedDecls`
  /// is true, any unparsed decls are removed from the module.
  ASTDecl *parseFile(unsigned fileId, bool eraseUnparsedDecls = true);

  /// Parse a SourceMgr file given its ID as a module. This is a specialized
  /// variant of parseFile that does as little work as possible while still
  /// producing a decl that is usable for the language server.
  ASTDecl *parseFileForLSP(unsigned fileId);

  /// Ensures that all parsed decls have been signature-resolved. This is a
  /// required step to ensure the IR is well-formed.
  void ensureSignaturesResolved();

  /// Parse a package with the given path.
  ///
  /// In the case of success, the decl corresponding to the package is returned.
  /// In the case of an error, a null decl is returned.
  ASTDecl *parsePackage(const std::filesystem::path &path);

  /// Parse a module or package with the given path.
  ///
  /// In the case of success, the corresponding decl is returned.
  /// In the case of an error, a null decl is returned.
  ASTDecl *parseFileOrPackage(const std::filesystem::path &path);

  /// Parse a module or package with the given path, without resolving any of
  /// the nested decls. The returned decl provides only a partial view of the
  /// module or package, and does not contain information for nested decls.
  ///
  /// In the case of success, the corresponding decl is returned.
  /// In the case of an error, a null decl is returned.
  ASTDecl *parseFileOrPackageNonRecursive(const std::filesystem::path &path);

  /// Parse a module or package with the given path, without resolving
  /// "external" decls that originate from other modules/packages.
  ///
  /// In the case of success, the corresponding decl is returned.
  /// In the case of an error, a null decl is returned.
  ASTDecl *parseIsolatedFileOrPackage(const std::filesystem::path &path);

  /// Returns true if an error occurred during parsing.
  bool wasErrorEmitted() const;

protected:
  /// A struct representing the internal state of the parser.
  struct Impl;

  /// The internal state of the parser.
  std::unique_ptr<Impl> impl;
};

} // namespace M::KGEN::LIT

namespace M {
using KGEN::LIT::MojoParserContext;
} // namespace M

#endif // KGEN_MOJOTOOLING_PARSERDRIVER_H
