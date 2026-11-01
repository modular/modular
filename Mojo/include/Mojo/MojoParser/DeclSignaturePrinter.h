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

#ifndef KGEN_MOJOPARSER_DECLSIGNATUREPRINTER_H
#define KGEN_MOJOPARSER_DECLSIGNATUREPRINTER_H

#include "Mojo/MojoParser/ASTDecl.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

namespace M {
namespace KGEN {

/// Optional output ranges to capture during signature emission. Useful for
/// downstream consumers (e.g. mojo-doc) that need to highlight or navigate to
/// individual signature components. All offsets are measured from the start of
/// the underlying string buffer being written into.
struct SignatureOffsets {
  /// Half-open `[start, end)` byte ranges for each documented parameter, in
  /// emission order.
  llvm::SmallVectorImpl<std::pair<unsigned, unsigned>> *parameters = nullptr;
  /// Half-open `[start, end)` byte ranges for each documented argument, in
  /// emission order.
  llvm::SmallVectorImpl<std::pair<unsigned, unsigned>> *arguments = nullptr;
  /// Byte offset where the return-type clause begins. Set even when there is
  /// no return type, so callers can splice in extra text (e.g. ` raises`)
  /// just before the result.
  unsigned *returnTypeStart = nullptr;
};

/// Print a Mojo-syntax signature for the given function op.
///
/// The declaration is installed as the current diagnostic context so dependent
/// parameter references render with their source names.
void printFunctionSignature(LIT::DefASTDecl &decl, llvm::raw_string_ostream &os,
                            const SignatureOffsets &offsets = {});

/// Print a Mojo-syntax signature for the given struct op.
void printStructSignature(LIT::StructASTDecl &decl,
                          llvm::raw_string_ostream &os,
                          const SignatureOffsets &offsets = {});

/// Print a Mojo-syntax signature for the given alias op.
///
/// No leading `comptime` keyword is emitted.
void printAliasSignature(LIT::AliasASTDecl &decl, llvm::raw_string_ostream &os,
                         const SignatureOffsets &offsets = {});

/// Print a Mojo-syntax signature for the given trait op.
void printTraitSignature(LIT::TraitASTDecl &decl, llvm::raw_string_ostream &os,
                         const SignatureOffsets &offsets = {});

std::string getFunctionSignature(
    LIT::DefASTDecl &decl,
    llvm::SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets =
        nullptr,
    llvm::SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets =
        nullptr,
    unsigned *returnOffset = nullptr);
std::string getFunctionDeclarationSnippet(
    LIT::DefASTDecl &decl,
    llvm::SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets =
        nullptr,
    llvm::SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets =
        nullptr);
std::string getStructSignature(
    LIT::StructASTDecl &decl,
    llvm::SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets =
        nullptr);

/// Return a source-facing declaration snippet for any supported declaration.
///
/// Unlike the signature-specific routines above, this includes declaration
/// keywords and variable/field type annotations. It returns an empty string
/// for declarations with no source-facing representation.
std::string getDeclarationSnippet(LIT::ASTDecl &decl);

/// Return the source-facing type for a parameter.
std::string getParameterTypeString(LIT::ASTDecl &decl);

/// Return the source-facing type and convention keyword for an argument.
std::string getArgumentTypeString(LIT::ASTDecl &decl);
llvm::StringRef getArgumentConventionString(LIT::ASTDecl &decl);

/// Return this argument or parameter's default with parameter references
/// rebound to their source-level names.
mlir::TypedAttr getReboundPogDefault(LIT::ASTDecl &decl);

/// Print one parameter or argument as it appears in a declaration signature.
void printParameter(LIT::ASTDecl &decl, llvm::raw_ostream &os);
void printArgument(LIT::ASTDecl &decl, llvm::raw_ostream &os);

/// Render a constraint proposition as Mojo source-level syntax.
std::string renderConstraintProposition(mlir::TypedAttr proposition,
                                        ParameterEvaluator *evaluator,
                                        LIT::SharedState &shared);

std::string generatePValueString(LIT::SharedState &shared, LIT::PValue value);
std::string getSignatureOrigin(LIT::SharedState &shared, mlir::TypedAttr origin,
                               LIT::FnTypeGeneratorType signature,
                               bool isRefResult);
std::string getRefPrefixAsString(LIT::SharedState &shared, LIT::RefType refType,
                                 LIT::FnTypeGeneratorType signature,
                                 bool isRefResult);
std::string
generateTypeString(LIT::SharedState &shared, LIT::ASTType type,
                   VariadicKind varKind,
                   std::optional<LIT::ASTType> selfType = std::nullopt,
                   std::optional<ArgConvention> convention = std::nullopt);
llvm::Twine prependVariadicIdentifiers(const llvm::Twine &identifier,
                                       VariadicKind varKind);
void dumpIdentifierWithType(llvm::raw_ostream &os, llvm::StringRef identifier,
                            llvm::StringRef type,
                            VariadicKind varKind = VariadicKind::None,
                            bool elideType = false);
std::string getDefaultValueString(mlir::TypedAttr defaultValue,
                                  LIT::SharedState &shared);

/// Remove implicit `argName.member` parameters from a rendered argument type.
std::string stripImplicitArgParams(llvm::StringRef typeStr,
                                   llvm::StringRef argName);

//===----------------------------------------------------------------------===//
// Diagnostic-oriented helpers
//===----------------------------------------------------------------------===//

/// Synthesize a Mojo-source-syntax signature for the given declaration.
/// Dispatches to the concrete declaration printer and returns an empty string
/// for unsupported declaration kinds.
///
/// Intended for diagnostics that need to identify a decl when a source-locator
/// won't reach the user (see `hasReadableSourceLocation`). Inexpensive enough
/// to call unconditionally; callers can fall back to other phrasings on an
/// empty return.
std::string synthesizeDeclSignature(const LIT::ASTDecl &decl);

/// True when `loc` resolves to a `FileLineColLoc` whose file is readable from
/// `sourceMgr`. False for `UnknownLoc`, locations with only `NameLoc`
/// metadata, or `FileLineColLoc`s referring to files the manager doesn't have
/// (e.g. decls loaded from a bytecode package). Walks `FusedLoc`/`NameLoc`/
/// `CallSiteLoc` chains.
///
/// Diagnostic emitters can use this to choose between attaching a normal
/// source-pointer note and synthesizing a self-describing note via
/// `synthesizeDeclSignature`.
bool hasReadableSourceLocation(Location loc, LIT::SharedState &shared);

} // namespace KGEN
} // namespace M

#endif // KGEN_MOJOPARSER_DECLSIGNATUREPRINTER_H
