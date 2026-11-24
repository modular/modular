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
// This file declares the internal documentation serialization entry point.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOTOOLING_DOCGENSERIALIZATION_H
#define KGEN_LIB_MOJOTOOLING_DOCGENSERIALIZATION_H

#include "llvm/Support/JSON.h"
#include <optional>

namespace M::KGEN::LIT {
class ASTDecl;
class SharedState;

/// Serialize a supported AST declaration. Returns nullopt for declarations
/// outside the public documentation model.
std::optional<llvm::json::Object> serializeDocDecl(ASTDecl &decl,
                                                   SharedState &shared);
} // namespace M::KGEN::LIT

#endif // KGEN_LIB_MOJOTOOLING_DOCGENSERIALIZATION_H
