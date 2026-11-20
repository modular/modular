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
// Shared doc-generation pipeline used by `mojo doc` and `kgen-translate`.
// Callers are responsible for creating the Init context, resolving the input
// path, setting up the MLIRContext (with registerAllKGENDialects), and managing
// the output file lifetime. This helper constructs a SourceMgr internally for
// the parse/serialize steps.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTOOLING_DOCGEN_H
#define KGEN_MOJOTOOLING_DOCGEN_H

#include "Support/Driver/DiagnosticFormat.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <string>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace M {
namespace KGEN {
struct FrontendOptions;
} // namespace KGEN

/// Parses the Mojo file or package at \p resolvedPath with \p frontendOptions,
/// serializes its public API documentation as JSON (including the top-level
/// version wrapper) to \p os, and returns true on success.
///
/// \p context must have all KGEN dialects already registered (e.g. via
/// registerAllKGENDialects). Imports are resolved against \p includePaths.
/// Diagnostics are written to stderr through the SourceMgr handler selected by
/// \p diagnosticFormat.
bool generateMojoDocJSON(const std::filesystem::path &resolvedPath,
                         mlir::MLIRContext &context,
                         const KGEN::FrontendOptions &frontendOptions,
                         llvm::ArrayRef<std::string> includePaths,
                         DiagnosticFormat diagnosticFormat,
                         llvm::raw_ostream &os);

} // namespace M

#endif // KGEN_MOJOTOOLING_DOCGEN_H
