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
// Shared packaging step used by `mojo precompile` and `kgen-translate
// --precompile`. Callers parse the package (LIT::importMojoPackage), run the
// check-LIT pipeline over the parsed module so its diagnostics reach the user,
// and write the result (writePrecompiledFile, or print it); this helper turns
// the parsed IR into the module a `.mojoc` file holds.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_PRECOMPILE_PRECOMPILE_H
#define MOJO_PRECOMPILE_PRECOMPILE_H

#include "Mojo/LITDialect/LITOps.h"
#include "Support/ErrorOr.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <string>

namespace M::KGEN {

/// Builds the module written to a `.mojoc` file from \p parsedModule, the
/// parser's output for a package directory, and \p parsedPackage, the package
/// op within it. The package was parsed under \p sourceName, its directory's
/// name, and is renamed to \p outputName. Each of \p bitcodeLibs is an LLVM
/// bitcode file embedded in the package.
ErrorOr<mlir::OwningOpRef<mlir::ModuleOp>>
buildPrecompiledPackage(mlir::ModuleOp parsedModule,
                        LIT::PackageOp parsedPackage,
                        llvm::StringRef sourceName, llvm::StringRef outputName,
                        llvm::ArrayRef<std::string> bitcodeLibs);

} // namespace M::KGEN

#endif // MOJO_PRECOMPILE_PRECOMPILE_H
