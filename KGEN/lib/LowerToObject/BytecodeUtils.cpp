//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "BytecodeUtils.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"

#include <fstream>

using namespace M;
using namespace KGEN;

FailureOr<Operation *>
M::KGEN::replaceSymbolFromBytecode(mlir::SymbolOpInterface toReplace,
                                   mlir::SymbolTable &symtab,
                                   llvm::MemoryBufferRef buf) {
  assert(mlir::isBytecode(buf) && "expected bytecode buffer");

  // Remove, but don't erase yet.
  symtab.remove(toReplace);
  // Read the `kgen.precompiled.llvm` from the cache directly into the block
  // with the function. Don't verify during the parse, the bytecode is not
  // self-contained. We will verify as soon as the module is in a state to be
  // verified.
  mlir::ParserConfig parserConfig(toReplace->getContext(),
                                  /*verifyAfterParse=*/false);
  Block *block = toReplace->getBlock();
  if (failed(mlir::readBytecodeFile(buf, block, parserConfig)))
    return failure();

  // Store the new op's name.
  std::string newOpName = toReplace.getName().str();

  // Find the op we just inserted. This is done in reverse order because the
  // bytecode parser likely added the new ops to the end of the block.
  mlir::SymbolOpInterface replacedWith;
  for (auto iter = block->rbegin(), end = block->rend(); iter != end; ++iter) {
    if (auto sym = dyn_cast<mlir::SymbolOpInterface>(*iter)) {
      if (sym != toReplace && sym.getName() == newOpName) {
        replacedWith = sym;
        break;
      }
    }
  }
  // Erase the op we're replacing.
  symtab.erase(toReplace);

  // Insert the new op into the symbol table.
  symtab.insert(replacedWith, {});

  // And verify the op we just replaced with.
  if (failed(mlir::verify(replacedWith)))
    return failure();

  return replacedWith.getOperation();
}
