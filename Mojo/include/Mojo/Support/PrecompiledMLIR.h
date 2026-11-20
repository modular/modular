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

#ifndef KGEN_SUPPORT_PRECOMPILEDMLIR_H
#define KGEN_SUPPORT_PRECOMPILEDMLIR_H

#include "Support/ML/DType.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace M::KGEN {

class PrecompiledMLIRCache;

/// A dialect-agnostic registry of precompiled MLIR "snippets", keyed by a
/// caller-defined `Kind`. Each snippet is a module containing a single
/// function-like op, stored as textual MLIR (canonicalize with kgen-opt and
/// embed the printed form, e.g. as a raw string literal). A registered
/// snippet can be inserted into a module (to be called) or inlined in place
/// of an op.
///
/// Contents are fixed at construction, so a shared registry needs no
/// synchronization.
///
/// Registration is policy-free: the caller owns what a kind means, which
/// snippet to register, and when to call vs. inline. The float conversions are
/// the one built-in set; their kinds stay private. Any
/// dialects used by the snippet must be registered in the target
/// `MLIRContext`. A snippet is parsed on each use into caller-owned storage;
/// callers that need the same snippet repeatedly should share a
/// `PrecompiledMLIRCache`.
class PrecompiledMLIRRegistry {
public:
  /// Caller-defined snippet identifier.
  using Kind = uint32_t;

  /// The textual MLIR of a module containing a single function-like op named
  /// `symbolName`, and the kind it answers to.
  struct Snippet {
    Kind kind;
    llvm::StringRef symbolName;
    llvm::StringRef mlirText;
  };

  /// `snippets` is copied, so it need not outlive the registry. Kinds must be
  /// unique.
  explicit PrecompiledMLIRRegistry(llvm::ArrayRef<Snippet> snippets);
  ~PrecompiledMLIRRegistry();
  PrecompiledMLIRRegistry(const PrecompiledMLIRRegistry &) = delete;
  PrecompiledMLIRRegistry &operator=(const PrecompiledMLIRRegistry &) = delete;

  /// Parse the snippet's module into `storage` (which the caller must keep
  /// alive) and return its function-like op, or nullptr if `kind` is unknown or
  /// parsing fails. Clone the op rather than mutating it. The parse is not
  /// cached here: it is tied to `ctx`, which a registry may outlive. Use a
  /// `PrecompiledMLIRCache` to reuse a parse within one context's lifetime.
  mlir::Operation *
  getFunction(Kind kind, mlir::MLIRContext *ctx,
              mlir::OwningOpRef<mlir::Operation *> &storage) const;

  /// Ensure the snippet's function (and any symbols it transitively references)
  /// exists in `module`, cloning them in if absent (deduplicated by symbol
  /// name). Returns the function op in `module`, or failure.
  mlir::FailureOr<mlir::Operation *>
  insertIntoModule(Kind kind, mlir::ModuleOp module) const;

  /// Inline the snippet in place of `op`: the function body is cloned at `op`,
  /// its arguments are bound to `inputs`, and `op`'s results are replaced by
  /// the function's returned values. `op` is erased on success. `inputs` must
  /// match the function's argument count and types.
  mlir::LogicalResult inlineInPlace(Kind kind, mlir::Operation *op,
                                    mlir::ValueRange inputs,
                                    mlir::RewriterBase &rewriter) const;

  /// Which of a conversion's two snippets to use. Both compute the same
  /// result; they differ in whether the conversion's branches survive.
  ///
  /// The fp8 conversions leave through an early return for NaN, for infinity
  /// and for overflow, and pick between a normal and a subnormal path. `Flat`
  /// is those collapsed into one block, which a SIMD cast needs: it becomes
  /// one conversion per lane, and branches leave SLP nothing to widen back
  /// up. `Branching` keeps them, so a conversion with no lanes behind it
  /// stops paying for the paths it does not take.
  enum class FloatConversionForm { Flat, Branching };

  // The built-in snippets: conversions a target may have no instruction for
  // (fp8 <-> f32 today), keyed by dtype pair and form. Empty or null means the
  // pair has no snippet, so callers need no separate query.
  static llvm::StringRef getFloatConversionSymbolName(DType from, DType to,
                                                      FloatConversionForm form);
  static mlir::FailureOr<mlir::Operation *>
  insertFloatConversion(DType from, DType to, FloatConversionForm form,
                        mlir::ModuleOp module);

  /// Given a `cache` the parse is shared; without one it lands in `storage`,
  /// which the caller must keep alive. Clone the op rather than mutating it.
  static mlir::Operation *
  getFloatConversionFunction(DType from, DType to, FloatConversionForm form,
                             mlir::MLIRContext *ctx,
                             mlir::OwningOpRef<mlir::Operation *> &storage,
                             PrecompiledMLIRCache *cache);

private:
  struct Entry {
    std::string symbolName;
    std::string mlirText;
  };

  static const PrecompiledMLIRRegistry &
  getFloatConversions(FloatConversionForm form);
  static std::optional<Kind> getFloatConversionKind(DType from, DType to,
                                                    FloatConversionForm form);
  static llvm::DenseMap<Kind, Entry>
  buildTable(llvm::ArrayRef<Snippet> snippets);

  const llvm::DenseMap<Kind, Entry> table;
};

/// Reuses parsed snippets across uses within a single `MLIRContext`, so a
/// snippet needed once per op is parsed once instead. Caches entries from any
/// number of registries.
///
/// The cache owns the parsed IR, so it must not outlive `ctx`; scope one to
/// something the context outlives (an MLIR analysis over the module being
/// lowered, say), never to a process-lifetime registry.
class PrecompiledMLIRCache {
public:
  explicit PrecompiledMLIRCache(mlir::MLIRContext *ctx) : ctx(ctx) {}
  ~PrecompiledMLIRCache();
  PrecompiledMLIRCache(const PrecompiledMLIRCache &) = delete;
  PrecompiledMLIRCache &operator=(const PrecompiledMLIRCache &) = delete;

  /// Return `registry`'s function-like op for `kind`, parsing it on first use,
  /// or nullptr if the kind is unknown or parsing fails. The op is owned by
  /// this cache and shared by all callers: clone it rather than mutating it.
  mlir::Operation *getFunction(const PrecompiledMLIRRegistry &registry,
                               PrecompiledMLIRRegistry::Kind kind);

private:
  struct Parsed {
    mlir::OwningOpRef<mlir::Operation *> module;
    mlir::Operation *function = nullptr;
  };

  mlir::MLIRContext *ctx;
  llvm::DenseMap<
      std::pair<const PrecompiledMLIRRegistry *, PrecompiledMLIRRegistry::Kind>,
      Parsed>
      parsed;
  std::mutex mutex;
};

} // namespace M::KGEN

#endif // KGEN_SUPPORT_PRECOMPILEDMLIR_H
