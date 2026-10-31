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

#ifndef KGEN_MOJOTOOLING_PublicASTDecl_H
#define KGEN_MOJOTOOLING_PublicASTDecl_H

#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/SignatureModel.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <string>
namespace llvm {
namespace json {
class Object;
} // namespace json
} // namespace llvm

namespace M::KGEN {
enum class PassingKind : uint32_t;
enum class VariadicKind : uint32_t;
} // namespace M::KGEN

namespace M::KGEN::LIT {

class ASTDecl;
class AliasASTDecl;
class DefASTDecl;
class FnSignatureASTDecl;
class MojoParserContext;
class PackageASTDecl;
class SharedState;
class StructASTDecl;
class StructFieldASTDecl;
class TraitASTDecl;
class VarASTDecl;
class FnTypeGeneratorType;

namespace {
/// Typical max parameter count for Mojo functions, structs, and aliases.
/// Currently about 98% of declarations use ≤2 parameters.
static constexpr size_t kTypicalParameterCount = 2;

/// Typical max argument count for Mojo functions.
/// Currently about 95% of functions use ≤2 arguments.
static constexpr size_t kTypicalArgumentCount = 2;
} // namespace
enum class PublicDeclKind;

std::optional<PublicDeclKind> getPublicDeclKind(ASTDecl *decl);
std::optional<PublicDeclKind> getApproximatePublicDeclKind(ASTDecl *decl);
StringRef getPublicDeclKindAsString(PublicDeclKind kind);
std::string getPublicDeclDeclarationSnippet(ASTDecl &decl,
                                            MojoParserContext &ctx);
std::string getPublicDeclMarkdownDocString(ASTDecl &decl);
std::string getPublicDeclFullMarkdownString(ASTDecl &decl,
                                            MojoParserContext &ctx);
/// Serialize a supported AST declaration. Returns nullopt for declarations
/// outside the public tooling model.
std::optional<llvm::json::Object> serializePublicDecl(ASTDecl &decl,
                                                      MojoParserContext &ctx);
std::string getToolingAliasValue(AliasASTDecl &decl);
SmallVector<KGEN::ParameterInfo, kTypicalParameterCount>
getStructParameters(StructASTDecl &decl);
std::string getStructSignature(
    StructASTDecl &decl,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets = nullptr);
std::string getStructDeclarationSnippet(
    StructASTDecl &decl, MojoParserContext &ctx,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets = nullptr);

/// Rendered `where` conditions of a decl's conditional trait conformances,
/// keyed by the conformed trait's symbol. Unconditional conformances are
/// absent. The key is the symbol rather than the trait's short name because
/// distinct symbols can share one name.
using ConformanceConditionMap =
    llvm::DenseMap<KGEN::TraitSymbolAttr, std::string>;

//===----------------------------------------------------------------------===//
// Public declaration rendering
//
// These helpers provide structured documentation and code snippets for the
// most common Mojo AST declarations.
//===----------------------------------------------------------------------===//

/// Stability information extracted from the @stable decorator on a decl.
/// isStable=false means no @stable decorator. sinceVersion is empty when
/// the decorator was bare (@stable with no arguments).
/// isStabilityTracked=true means the symbol is in a package that has opted
/// into stability tracking (currently only "std").
struct StableInfo {
  bool isStable = false;
  bool isStabilityTracked = false;
  std::string sinceVersion;
};

/// The different kinds of public decls.
enum class PublicDeclKind {
  DK_PublicAliasDecl,
  DK_PublicArgumentDecl,
  DK_PublicModuleDecl,
  DK_PublicPackageDecl,
  DK_PublicParameterDecl,
  DK_PublicStructDecl,
  DK_PublicStructFieldDecl,
  DK_PublicTraitDecl,
  DK_PublicVariableDecl,
};

/// Function tooling helpers operate on function signatures.
SmallVector<KGEN::ParameterInfo, kTypicalParameterCount>
getFunctionParameters(FnSignatureASTDecl &decl);
std::string getFunctionDeclarationSnippet(
    DefASTDecl &decl, MojoParserContext &ctx,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets = nullptr,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets = nullptr);
std::string getFunctionSignature(
    DefASTDecl &decl, MojoParserContext &ctx,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets = nullptr,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets = nullptr,
    unsigned *returnOffset = nullptr);
std::string getFunctionMarkdownDocString(DefASTDecl &decl);
std::string getFunctionFullMarkdownString(DefASTDecl &decl,
                                          MojoParserContext &ctx);
llvm::json::Object serializeFunctionDecl(DefASTDecl &decl,
                                         MojoParserContext &ctx);

/// A collection of overloaded functions in the same scope.
class FunctionDeclOverloadSet {
public:
  /// Create a list of function overload sets by grouping the given functions by
  /// their name. It's assumed that the input list is sorted by name.
  static SmallVector<FunctionDeclOverloadSet, 2>
  fromSortedFunctions(SmallVector<DefASTDecl *, 2> &&functions);

  /// Get the common name of the functions in this overload set.
  StringRef getBaseName() const { return baseName; }

  /// Get the functions in this overload set.
  ArrayRef<DefASTDecl *> getFunctions() const { return functions; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "function",
  ///   "name": string,
  ///   "overloads": function declarations
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const;

private:
  FunctionDeclOverloadSet(StringRef baseName) : baseName(baseName) {}

  void append(DefASTDecl *function) { functions.push_back(function); }

  std::string baseName;
  SmallVector<DefASTDecl *, 2> functions;
};

} // namespace M::KGEN::LIT

namespace M {
using KGEN::LIT::ConformanceConditionMap;
using KGEN::LIT::FunctionDeclOverloadSet;
using KGEN::LIT::PublicDeclKind;
using KGEN::LIT::StableInfo;
} // namespace M

#endif // KGEN_MOJOTOOLING_PublicASTDecl_H
