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
// Write side of the experimental incremental parsing cache
// (`--experimental-incr-parsing`): the completeness marks the cache writer
// records on declaration ops, and the cache-file writer itself. The read
// side (opening a cache and serving modules from it) lands separately.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_MOJOPARSER_MODULECACHE_H
#define MOJO_MOJOPARSER_MODULECACHE_H

#include "mlir/IR/BuiltinOps.h"

namespace M::KGEN::LIT {
class ASTDecl;
class SharedState;

/// Discardable attr marking a decl left unresolved; its payload [resumeOffset,
/// resumeIndentation, endOffset, declIndentation] lets a serve re-parse it.
inline constexpr llvm::StringLiteral kIncrParsingCacheResumeInfoAttrName =
    "lit.incr_parsing_cache_resume_info";

/// Serialize `module` into the incremental parsing cache file, embedding the
/// dependency graph built during this parse. A failed write is non-fatal.
void writeIncrParsingCache(SharedState &sharedState, mlir::ModuleOp module,
                           const ASTDecl &moduleDecl);

/// Erase every resume attr from the live module: only the cache file may
/// carry them, never downstream stages or other serializations.
void eraseIncrParsingCacheResumeAttrs(mlir::ModuleOp module);

} // namespace M::KGEN::LIT

#endif // MOJO_MOJOPARSER_MODULECACHE_H
