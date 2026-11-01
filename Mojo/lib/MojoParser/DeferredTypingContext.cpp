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

#include "DeferredTypingContext.h"
#include "Mojo/MojoParser/ASTDecl.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

ParamDeclAttr ImplicitParamManager::declare(Type type, StringRef name,
                                            SMLoc loc) {
  StringAttr mangledName =
      declScope.mangleParamName(prefix.strref() + "." + name);
  auto decl = ParamDeclAttr::get(mangledName, type);
  pending.push_back({decl, loc});
  return decl;
}
