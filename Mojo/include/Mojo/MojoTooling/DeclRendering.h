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

#ifndef KGEN_MOJOTOOLING_DECLRENDERING_H
#define KGEN_MOJOTOOLING_DECLRENDERING_H

#include "Mojo/MojoParser/SignatureModel.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace M::KGEN::LIT {

class ASTDecl;
class AliasASTDecl;
class DefASTDecl;
class FnSignatureASTDecl;
class MojoParserContext;
class StructASTDecl;

StringRef getPublicDeclKindName(ASTDecl &decl);
std::string getPublicDeclDeclarationSnippet(ASTDecl &decl,
                                            MojoParserContext &ctx);
std::string getPublicDeclMarkdownDocString(ASTDecl &decl);
std::string getPublicDeclFullMarkdownString(ASTDecl &decl,
                                            MojoParserContext &ctx);

std::string getToolingAliasValue(AliasASTDecl &decl);

SmallVector<KGEN::ParameterInfo, 2> getStructParameters(StructASTDecl &decl);
std::string getStructSignature(
    StructASTDecl &decl,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets = nullptr);
std::string getStructDeclarationSnippet(
    StructASTDecl &decl, MojoParserContext &ctx,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets = nullptr);

SmallVector<KGEN::ParameterInfo, 2>
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

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOTOOLING_DECLRENDERING_H
