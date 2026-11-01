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
// This file contains various utilities for interacting with the Mojo REPL.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTOOLING_REPLPYTHONEXPRUTILS_H
#define KGEN_MOJOTOOLING_REPLPYTHONEXPRUTILS_H

#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace M {
namespace KGEN {
namespace Mojo {

/// This struct describes an extracted Python symbol.
struct ExtractedPythonSymbol {
public:
  enum class Kind {
    /// A python import statement.
    Import,
    /// A python declaration statement.
    Decl,
  };

  ExtractedPythonSymbol(Kind kind, StringRef name, StringRef module = {})
      : kind(kind), name(name.str()), module(module.str()) {}

  /// Return the exposed name of this symbol.
  StringRef getName() const { return name; }

  /// Return the kind of this symbol.
  Kind getKind() const { return kind; }

  /// Return the module name of this import.
  StringRef getModule() const { return module; }

  Kind kind;
  std::string name;
  std::string module;
};

/// Extract the importable python symbols from the given python expression
/// string.
ErrorOr<std::vector<ExtractedPythonSymbol>>
extractPythonSymbolsFromReplExpr(StringRef pythonExpr);

} // namespace Mojo
} // namespace KGEN
} // namespace M

#endif // KGEN_MOJOTOOLING_REPLPYTHONEXPRUTILS_H
