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
// This file defines an KGEN MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_H
#define KGEN_KGENDIALECT_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringMap.h"

#include <filesystem>
#include <type_traits>

namespace M::KGEN {
class DedupContext;
class KGENParamPrinter;
} // namespace M::KGEN

// Pull in the dialect definition.
#pragma GCC visibility push(default)
#include "Mojo/KGENDialect/KGENDialect.h.inc"
#pragma GCC visibility pop

namespace M::KGEN {
/// Register KGEN dialect commandline options.
void registerKGENCommandLineOptions();

template <typename TypeT>
void KGENDialect::registerMnemonicType() {
  // A type with no parameters has nothing to thread through, so it keeps the
  // plain body printer.
  constexpr bool hasParameters =
      !std::is_same_v<typename TypeT::ImplType, mlir::TypeStorage>;
  constexpr bool hasParamPrinter = requires(
      const TypeT t, AsmPrinter &pp, KGENParamPrinter &kp) { t.print(pp, kp); };

  static_assert(!hasParameters || hasParamPrinter,
                "a type with parameters must declare "
                "print(AsmPrinter &, KGENParamPrinter &), so its nested "
                "parameters keep the caller's dedup labels and memref hashing");

  registerPrettyType(
      TypeT::getMnemonic(), +[](AsmParser &p) { return TypeT::parse(p); },
      mlir::TypeID::get<TypeT>(),
      +[](AsmPrinter &p, Type type) {
        p << TypeT::getMnemonic();
        cast<TypeT>(type).print(p);
      },
      +[](AsmPrinter &p, Type type, KGENParamPrinter &printer) {
        p << TypeT::getMnemonic();
        if constexpr (hasParamPrinter)
          cast<TypeT>(type).print(p, printer);
        else
          cast<TypeT>(type).print(p);
      });
}
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_H
