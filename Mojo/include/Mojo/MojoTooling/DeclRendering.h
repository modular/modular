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

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace M::KGEN::LIT {

class ASTDecl;
class StructASTDecl;

std::string getDeclDeclarationSnippet(ASTDecl &decl);
std::string getDeclMarkdownDocString(ASTDecl &decl);
std::string getDeclFullMarkdownString(ASTDecl &decl);

std::string getStructDeclarationSnippet(
    StructASTDecl &decl,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets = nullptr);

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOTOOLING_DECLRENDERING_H
