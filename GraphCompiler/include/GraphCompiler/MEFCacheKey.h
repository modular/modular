//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MEFCACHEKEY_H
#define GENERICML_MEFCACHEKEY_H

#include "Support/Context.h"
#include "Support/ErrorOr.h"

#include "llvm/Support/BLAKE3.h"

#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace M {

namespace MO {
struct GCPipelineOptions;
}

/// Incremental hasher for MEF cache keys, holding precomputed session-level
/// hash state. The key depends on
/// - The GC pass pipeline, version, and options
/// - KGEN compilation context (Mojo compile-time defines)
/// - User-provided kernel library files
/// - The built-in kernel packages and every Mojo binary package on the
///   module import path (the kernel implementations the built-in packages
///   re-export live there)
///
/// It _does not_ depend on location info or specific symbol names.
class MEFCacheKeyHasher {
public:
  static ErrorOr<MEFCacheKeyHasher>
  build(mlir::MLIRContext &mlirCtx,
        const MO::GCPipelineOptions &gcPipelineOptions, ContextRef maxContext);

  ErrorOr<std::string> keyForModule(mlir::ModuleOp module) const;

private:
  MEFCacheKeyHasher() = default;
  llvm::BLAKE3 prefixHasher;
};

ErrorOr<std::string>
computeMEFCacheKey(mlir::ModuleOp module,
                   const MO::GCPipelineOptions &gcPipelineOptions,
                   ContextRef maxContext);

} // namespace M

#endif // GENERICML_MEFCACHEKEY_H
