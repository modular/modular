//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TRANSFORMUTILS_ELIMINATEDEADSYMBOLUTILS_H
#define KGEN_TRANSFORMUTILS_ELIMINATEDEADSYMBOLUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace mlir {
class SymbolTableAnalysis;
}

namespace M::KGEN {

DenseSet<StringAttr> getUsedSymbols(mlir::SymbolTableAnalysis &analysis,
                                    ModuleOp theModule);

}

#endif
