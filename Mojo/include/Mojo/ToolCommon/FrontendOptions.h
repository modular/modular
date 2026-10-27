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

#ifndef KGEN_TOOLCOMMON_FRONTENDOPTIONS_H
#define KGEN_TOOLCOMMON_FRONTENDOPTIONS_H

#include "llvm/ADT/SmallVector.h"
#include <string>

namespace M {
class AutoFixItHandler;
} // namespace M

namespace M::KGEN::LIT {
class ParserListener;
} // namespace M::KGEN::LIT

namespace M::KGEN {

/// The settings the parser reads when parsing Mojo source. The parser also
/// reads the debug-info settings of the `CompilationOptions` it is given.
struct FrontendOptions {
  /// When non-null, this handler collects and applies fix-its automatically.
  AutoFixItHandler *autoFixItHandler = nullptr;

  /// When non-empty, this prefix is stripped from filenames used for
  /// diagnostics & debugging.
  std::string stripFilePrefix;

  /// When true, this prints diagnostics through MLIR (so MLIR features like
  /// -verify-diagnostics may be used). When false, this prints them through
  /// SourceMgr to get ranges and fixit hints.
  bool useMLIRDiagnostics = false;

  /// If true, this will diagnose missing pieces of documentation strings.
  bool diagnoseMissingDocStrings = false;

  /// If true, ignore any already-compiled `foo.mojoc` that appear in its import
  /// search paths. Doing so results in Mojo source packages named `foo/` being
  /// found instead, and those source packages being parsed anew.
  bool disablePrebuiltPackages = false;

  /// If true, auto-import the builtin package.
  bool useBuiltinModule = true;

  /// An optional listener that is used to inspect certain events of the parser.
  /// For simplicity it is a single item, but it could evolve into a list of
  /// listeners.
  LIT::ParserListener *parserListener = nullptr;

  /// Maximum number of notes to print per compiler error or warning.
  int maxNotesPerDiagnostic = 10;

  /// Base library path prefix for generated documentation links.
  std::string docsBasePath;

  /// Comma-separated directories that replace the default auto-import search
  /// path, set via `-mojo-search-paths`.
  std::string searchPaths;

  /// Directories searched for auto-imports in addition to `searchPaths`.
  llvm::SmallVector<std::string> extraSearchPaths;

  bool disableWarnings = false;
  bool warningsAsErrors = false;
  bool warnOnUnstableAPIs = false;
  bool ignoreIncompatiblePrecompiledFileErrors = false;

  /// Qualified names (e.g. `Foo.bar`, or just `some_fn` for a top-level
  /// declaration) of `@deprecated` declarations whose deprecation warning
  /// should be suppressed, set via `--ignore-deprecated`.
  llvm::SmallVector<std::string> ignoredDeprecations;

  /// Experimental incremental parsing cache file backing
  /// `--experimental-incr-parsing`; empty when the option is off.
  std::string incrParsingCacheFile;
};

} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_FRONTENDOPTIONS_H
